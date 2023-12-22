from collections.abc import Callable

import jax
import jax.numpy as jnp

from decoding.utils import BeamDecodingState, DecodingState, brevity_penalty
from type_annotations import Array, PyTree

NEG_INF = jnp.array(-1.0e7)


def _add_beam_dim(arr: Array, beams: int) -> Array:
    """Adds a beam dimension to a batch of sequences.

    Args:
        arr: An array of dimensions [batch_size, max_seq_length] containing a batch of sequences
        beams: The size of the beam dimension to create.

    Returns:
        `arr` with an added beam dimension. The result has dimensions [batch_size, beams, max_seq_length].
        Each sequence in the batch is repeated `beams` times along the new beam dimension.
    """
    expanded = jnp.expand_dims(arr, 1)
    reps = [1] * expanded.ndim
    reps[1] = beams
    return jnp.tile(expanded, reps)


def flatten_beam_dim(arr: Array) -> Array:
    """Flattens the first two (batch and beam) dimensions of an Array.

    Args:
        arr: An array to be reshaped.

    Returns:
        A form of `arr` with the first two dimensions flattened.
    """
    return arr.reshape((-1,) + arr.shape[2:])


def _unflatten_beam_dim(arr: Array, beams: int) -> Array:
    """Reshapes an Array by splitting the first dimension into batch and beam dimensions.

    Args:
        arr: The array to be reshaped.
        beams: The size of the beam dimension.

    Returns:
        The reshaped form of `arr`.
    """
    return arr.reshape((-1, beams) + arr.shape[1:])


def _gather_beams(arr: Array, batch_indices: Array, beam_indices: Array) -> Array:
    """Selects sequences from an Array according to supplied vectors of batch and beam indices.

    Args:
        arr: An Array of dimensions [batch_size, beams, max_sequence_length] containing sequences to select from.
        batch_indices: An Array of dimension [batch_size * beams] containing the batch indices for the selected
            sequences.
        beam_indices: An Array of dimension [batch_size * beams] containing the beam indices of the selected sequences.

    Returns:
         An Array of the same shape as `arr` containing the selected sequences.
         Note that some sequences may have been selected multiple times.
    """
    return arr.at[batch_indices, beam_indices].get().reshape(arr.shape)


def beam_search(
    sequences: Array,
    decoding_start_index: Array,
    sequences_to_logits: Callable[[Array], tuple[Array, PyTree]],
    tokens_to_logits: Callable[[DecodingState], tuple[Array, PyTree]],
    eos_token_id: int,
    beams: int = 4,
    alpha: float = 0.6,
) -> tuple[Array, Array]:
    """Runs beam search decoding.

    Args:
        sequences: A batch of input token id sequences. Has dimensions [batch_size, max_sequence_length].
        decoding_start_index: A column vector containing the indices (into `sequences`) where the first decoded tokens
            will be stored. For sequence i, the `beam_search` method will store the first decoded token to
            `sequences[i, decoding_start_index[i]]`.
        sequences_to_logits: A function which performs the initial forward-pass of the model on the input sequences
            (prefixes). Returns a tuple where the first element is an Array of dimensions
            [batch_size, max_sequence_length, vocab_size] containing the logits for each input token.
            and the second element is a PyTree containing the cache variables to be used for autoregressive decoding.
        tokens_to_logits: A function which performs one step of autoregressive decoding. Returns a tuple where the first
            element is an Array of dimensions [batch_size, 1, vocab_size] containing the logits for the input token and
            the second element is a PyTree containing the updated cache variables to be used in subsequent decoding
            steps.
        eos_token_id: The ID of the EOS token.
        beams: The size of the beam to use. A value of 1 results in a greedy search.
        alpha: A scaling factor for the brevity penalty.

    Returns:
        A tuple where the first element is an Array of dimensions [batch_size, beams, max_sequence_length] containing
        the top `beams` sequences found for each element in the input batch sorted in descending order of log
        likelihood, and the second element is an Array of dimensions [batch_size, beams] containing the log
        probabilities of each of the decoded sequences. Note that the log probabilities do not take the prompt tokens
        into account. Thus, the same sequences can have different log probabilities if `decoding_start_index` differs.
    """
    max_length = sequences.shape[1]

    logits, cache = sequences_to_logits(sequences)

    batch_size = sequences.shape[0]

    # beam search starts here

    log_probs = jax.nn.log_softmax(logits)

    # currently log_probs contains the model outputs for all tokens in the input sequence
    # we want just the outputs from the final input token of each sequence
    # [batch_size, vocab_size]
    first_decode_token_log_probs = log_probs.at[
        jnp.arange(batch_size), decoding_start_index.ravel() - 1
    ].get()

    # [batch_size, beams]
    topk_log_probs, topk_indices = jax.lax.top_k(
        first_decode_token_log_probs,
        k=beams,
    )

    # [batch_size, beams, max_length]
    expanded_sequences = _add_beam_dim(sequences, beams)

    # [batch_size, beams, 1]
    expanded_decoding_start_index = _add_beam_dim(decoding_start_index, beams)

    batch_indices = jnp.arange(batch_size * beams) // beams
    beam_indices = jnp.arange(batch_size * beams) % beams

    # add the first decoded token to each sequence
    expanded_sequences = expanded_sequences.at[
        batch_indices, beam_indices, expanded_decoding_start_index.ravel()
    ].set(topk_indices.ravel())

    new_decoding_start_index = flatten_beam_dim(expanded_decoding_start_index)

    decode_state = BeamDecodingState(
        cur_index=new_decoding_start_index,
        sequences=flatten_beam_dim(expanded_sequences),
        cache=jax.tree_map(lambda x: flatten_beam_dim(_add_beam_dim(x, beams)), cache),
        sequence_log_probs=flatten_beam_dim(topk_log_probs),
        sequence_is_terminated=jnp.logical_or(
            topk_indices.ravel() == eos_token_id,
            new_decoding_start_index.ravel() == max_length - 1,
        ),
        sequence_lengths=jnp.full(batch_size * beams, 1),
    )

    def loop_body_func(state: BeamDecodingState) -> BeamDecodingState:
        # logits <- [batch_size * beams, 1, vocab_size]
        logits, new_cache = tokens_to_logits(state)
        log_probs = jax.nn.log_softmax(logits)

        # add log probs for new tokens to running total for each sequence (unless the sequence is already terminated)
        # [batch_size * beams, 1, vocab_size]
        new_sequence_log_probs = state.sequence_log_probs.reshape(
            batch_size * beams, 1, 1
        ) + log_probs * ~state.sequence_is_terminated.reshape(batch_size * beams, 1, 1)

        # update the lengths of each sequence (unless the sequence is already terminated)
        # [batch_size * beams]
        new_sequence_lengths = (
            state.sequence_is_terminated * state.sequence_lengths
            + ~state.sequence_is_terminated * state.sequence_lengths
            + 1
        )

        # [batch_size * beams, 1, vocab_size]
        scores = new_sequence_log_probs / brevity_penalty(
            alpha, new_sequence_lengths
        ).reshape(-1, 1, 1)

        vocab_size = log_probs.shape[-1]

        # if a sequence is terminated, it should appear at most once in the new beam
        # we do this by setting the scores to NEG_INF for all but one of the expansions of the terminated sequence
        # [batch_size * beams, 1, vocab_size]
        mask = jnp.tile(
            jnp.array([0.0] + [NEG_INF] * (vocab_size - 1)), [batch_size * beams, 1, 1]
        )
        scores += state.sequence_is_terminated.reshape(-1, 1, 1) * mask

        # [batch_size, beams]
        _, topk_indices = jax.lax.top_k(
            (scores).reshape(batch_size, -1),
            beams,
        )

        # gather the top k sequences in the beam
        topk_beam_indices = topk_indices // vocab_size
        topk_vocab_indices = topk_indices % vocab_size

        # gather the log probs for the top k sequences in the beam
        new_sequence_log_probs = new_sequence_log_probs.at[
            jnp.arange(batch_size * beams), 1, topk_vocab_indices.ravel()
        ].get()

        # [batch_size, beams, max_length]
        sequences = _unflatten_beam_dim(state.sequences, beams)
        sequences = _gather_beams(sequences, batch_indices, topk_beam_indices.ravel())

        # gather the is_terminated values for the top k sequences in the beam
        # flatten the beam dimension
        new_sequence_is_terminated = flatten_beam_dim(
            _gather_beams(
                _unflatten_beam_dim(state.sequence_is_terminated, beams),
                batch_indices,
                topk_beam_indices.ravel(),
            )
        )

        # add the next decoded token to each (un-terminated) sequence
        # flatten the beam dimension
        new_sequences = flatten_beam_dim(
            sequences.at[batch_indices, beam_indices, state.cur_index.ravel() + 1].set(
                topk_vocab_indices.ravel() * ~new_sequence_is_terminated.ravel()
            )
        )

        # gather caches for top-k sequences in the beam
        new_cache = jax.tree_map(
            lambda x: flatten_beam_dim(
                _gather_beams(
                    _unflatten_beam_dim(x, beams),
                    batch_indices,
                    topk_beam_indices.ravel(),
                )
            ),
            new_cache,
        )

        new_cur_index = jnp.minimum(state.cur_index + 1, max_length - 1)

        # terminate sequences where an EOS token was predicted
        new_sequence_is_terminated = jnp.logical_or(
            new_sequence_is_terminated, topk_vocab_indices.ravel() == eos_token_id
        )
        # terminate sequences which have reached the max length
        new_sequence_is_terminated = jnp.logical_or(
            new_sequence_is_terminated, new_cur_index.ravel() == max_length - 1
        )

        return BeamDecodingState(
            cur_index=new_cur_index,
            sequences=new_sequences,
            cache=new_cache,
            sequence_log_probs=new_sequence_log_probs,
            sequence_is_terminated=new_sequence_is_terminated,
            sequence_lengths=new_sequence_lengths,
        )

    def loop_cond_func(state: BeamDecodingState) -> jax.Array:
        return ~jnp.all(jnp.expand_dims(state.sequence_is_terminated, axis=1))

    final_state = jax.lax.while_loop(loop_cond_func, loop_body_func, decode_state)

    return _unflatten_beam_dim(final_state.sequences, beams), _unflatten_beam_dim(
        final_state.sequence_log_probs, beams
    )
