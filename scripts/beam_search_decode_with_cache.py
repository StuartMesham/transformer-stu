import jax
import jax.numpy as jnp
import tensorflow_text as tf_text
from orbax.checkpoint import PyTreeCheckpointer
from utils import BeamDecodingState, DecodingState, brevity_penalty

from transformer import Transformer
from type_annotations import Array, PyTree

BEAMS = 20
MAX_LENGTH = 25
EOS_TOKEN_ID = 2
BREVITY_PENALTY_ALPHA = 0.6

NEG_INF = jnp.array(-1.0e7)


def add_beam_dim(arr: Array, beams: int) -> Array:
    expanded = jnp.expand_dims(arr, 1)
    reps = [1] * expanded.ndim
    reps[1] = beams
    return jnp.tile(expanded, reps)


def flatten_beam_dim(arr: Array) -> Array:
    return arr.reshape((-1,) + arr.shape[2:])


def unflatten_beam_dim(arr: Array, beams: int) -> Array:
    return arr.reshape((-1, beams) + arr.shape[1:])


def gather_beams(arr: Array, batch_indices: Array, beam_indices: Array):
    return arr.at[batch_indices, beam_indices].get().reshape(arr.shape)


def main() -> None:
    """Runs greedy decoding using an activation cache."""
    checkpointer = PyTreeCheckpointer()
    restored_params = checkpointer.restore("model_saves/70/default")

    transformer = Transformer(
        max_length=522,
        vocab_size=10149,
        emb_size=512,
        mlp_hidden_dim=1024,
        num_layers=15,
        num_heads=4,
        dropout_rate=0.1,
        decode=True,
    )

    with open("model_saves/70/en-de.model", "rb") as f:
        tokenizer = tf_text.SentencepieceTokenizer(f.read(), add_eos=True)

    token_ids_ragged_tensor = tokenizer.tokenize(
        ["i would really like a cup of tea.", "please give me one."]
    )
    batch_size = token_ids_ragged_tensor.shape[0]
    decoding_start_index = jnp.asarray(
        token_ids_ragged_tensor.row_lengths().numpy().reshape(batch_size, 1),
        dtype=jnp.int32,
    )
    sequences = jnp.asarray(
        token_ids_ragged_tensor.to_tensor(
            default_value=0, shape=[batch_size, MAX_LENGTH]
        ).numpy()
    )

    batch = {
        "token_ids": sequences,
        "position_ids": jnp.broadcast_to(
            jnp.arange(0, sequences.shape[1]), sequences.shape
        ),
        "bidirectional_attention_mask": sequences != 0,
    }

    logits, initial_variables = transformer.apply(
        {"params": restored_params}, batch, eval_mode=True, mutable=True
    )

    cache = initial_variables["cache"]
    cache = jax.tree_map(
        lambda x: dict(
            x,
            cache_index=decoding_start_index,
        ),
        cache,
        is_leaf=lambda x: "cached_value" in x,
    )

    # beam search starts here

    log_probs = jax.nn.log_softmax(logits)
    topk_log_probs, topk_indices = jax.lax.top_k(
        log_probs.at[jnp.arange(batch_size), decoding_start_index.ravel() - 1].get(),
        k=BEAMS,
    )

    expanded_sequences = add_beam_dim(sequences, BEAMS)
    expanded_decoding_start_index = add_beam_dim(decoding_start_index, BEAMS)

    sequence_log_probs = topk_log_probs

    batch_indices = jnp.arange(batch_size * BEAMS) // BEAMS
    beam_indices = jnp.arange(batch_size * BEAMS) % BEAMS

    expanded_sequences = expanded_sequences.at[
        batch_indices, beam_indices, expanded_decoding_start_index.ravel()
    ].set(topk_indices.ravel())

    cache = jax.tree_map(lambda x: flatten_beam_dim(add_beam_dim(x, BEAMS)), cache)

    decode_state = BeamDecodingState(
        cur_index=flatten_beam_dim(expanded_decoding_start_index),
        sequences=flatten_beam_dim(expanded_sequences),
        cache=cache,
        sequence_log_probs=flatten_beam_dim(sequence_log_probs),
        sequence_is_terminated=topk_indices.ravel() == EOS_TOKEN_ID,
        sequence_lengths=jnp.full(batch_size * BEAMS, 1),
    )

    def tokens_to_logits(state: DecodingState) -> tuple[Array, PyTree]:
        batch = {
            "token_ids": state.sequences[
                jnp.arange(state.sequences.shape[0]), state.cur_index.ravel()
            ].reshape(state.sequences.shape[0], 1),
            "position_ids": state.cur_index,
            "bidirectional_attention_mask": jnp.zeros(
                (state.sequences.shape[0], 1), dtype="int32"
            ),
        }
        logits, new_vars = transformer.apply(
            {"params": restored_params, "cache": state.cache},
            batch,
            mutable=["cache"],
            eval_mode=True,
        )
        return logits, new_vars["cache"]

    def loop_body_func(state: BeamDecodingState) -> BeamDecodingState:
        logits, new_cache = tokens_to_logits(state)
        log_probs = jax.nn.log_softmax(logits)
        log_probs *= ~state.sequence_is_terminated.reshape(batch_size * BEAMS, 1, 1)
        new_sequence_lengths = (
            state.sequence_is_terminated * state.sequence_lengths
            + ~state.sequence_is_terminated
            * jnp.minimum(state.cur_index.ravel() + 1, MAX_LENGTH - 1)
        )
        new_sequence_log_probs = (
            state.sequence_log_probs.reshape(batch_size * BEAMS, 1, 1) + log_probs
        )

        scores = new_sequence_log_probs * brevity_penalty(
            BREVITY_PENALTY_ALPHA, new_sequence_lengths
        ).reshape(-1, 1, 1)

        vocab_size = log_probs.shape[-1]

        mask = jnp.tile(
            jnp.array([0.0] + [NEG_INF] * (vocab_size - 1)), [batch_size * BEAMS, 1, 1]
        )

        scores += state.sequence_is_terminated.reshape(-1, 1, 1) * mask

        _, topk_indices = jax.lax.top_k(
            (scores).reshape(batch_size, -1),
            BEAMS,
        )
        new_sequence_log_probs = new_sequence_log_probs.at[
            jnp.arange(batch_size * BEAMS), 1, topk_indices.ravel()
        ].get()
        topk_beam_indices = topk_indices // vocab_size
        topk_vocab_indices = topk_indices % vocab_size

        sequences = unflatten_beam_dim(state.sequences, BEAMS)
        sequences = gather_beams(sequences, batch_indices, topk_beam_indices.ravel())

        gathered_sequence_is_terminated = flatten_beam_dim(
            gather_beams(
                unflatten_beam_dim(state.sequence_is_terminated, BEAMS),
                batch_indices,
                topk_beam_indices.ravel(),
            )
        )

        new_sequences = flatten_beam_dim(
            sequences.at[batch_indices, beam_indices, state.cur_index.ravel() + 1].set(
                topk_vocab_indices.ravel() * ~gathered_sequence_is_terminated.ravel()
            )
        )

        new_cache = jax.tree_map(
            lambda x: flatten_beam_dim(
                gather_beams(
                    unflatten_beam_dim(x, BEAMS),
                    batch_indices,
                    topk_beam_indices.ravel(),
                )
            ),
            new_cache,
        )

        new_sequence_is_terminated = jnp.logical_or(
            gathered_sequence_is_terminated, topk_vocab_indices.ravel() == EOS_TOKEN_ID
        )

        # print(jnp.array_str(new_sequences, max_line_width=59999))
        # print(new_sequence_is_terminated)
        # print('\n'.join([s.decode() for s in tokenizer.detokenize(new_sequences).numpy()]))

        return BeamDecodingState(
            cur_index=jnp.minimum(state.cur_index + 1, MAX_LENGTH - 1),
            sequences=new_sequences,
            cache=new_cache,
            sequence_log_probs=new_sequence_log_probs,
            sequence_is_terminated=jnp.logical_or(
                new_sequence_is_terminated,
                jnp.minimum(state.cur_index + 1, MAX_LENGTH - 1).ravel()
                == MAX_LENGTH - 1,
            ),
            sequence_lengths=new_sequence_lengths,
        )

    def loop_cond_func(state: BeamDecodingState) -> jax.Array:
        return ~jnp.all(
            jnp.logical_or(
                state.cur_index >= MAX_LENGTH - 1,
                jnp.expand_dims(state.sequence_is_terminated, axis=1),
            )
        )

    # with jax.disable_jit():
    final_state = jax.lax.while_loop(loop_cond_func, loop_body_func, decode_state)

    for decoded_string in tokenizer.detokenize(final_state.sequences).numpy():
        print(decoded_string.decode())
    print()


if __name__ == "__main__":
    main()
