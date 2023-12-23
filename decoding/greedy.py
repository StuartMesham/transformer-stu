from collections.abc import Callable

import jax
import jax.numpy as jnp

from decoding.utils import DecodingState
from type_annotations import Array, PyTree


def greedy_search(
    sequences: Array,
    decoding_start_index: Array,
    sequences_to_logits: Callable[[Array], tuple[Array, PyTree]],
    tokens_to_logits: Callable[[DecodingState], tuple[Array, PyTree]],
    eos_token_id: int,
) -> Array:
    max_length = sequences.shape[1]
    batch_size = sequences.shape[0]

    logits, cache = sequences_to_logits(sequences)

    sequences = sequences.at[jnp.arange(batch_size), decoding_start_index.ravel()].set(
        logits.argmax(axis=2)[jnp.arange(batch_size), decoding_start_index.ravel() - 1]
    )

    decode_state = DecodingState(
        cur_index=decoding_start_index,
        sequences=sequences,
        cache=cache,
    )

    def loop_body_func(state: DecodingState) -> DecodingState:
        logits, new_cache = tokens_to_logits(state)

        # Add current sampled tokens to recorded sequences.
        next_tokens = logits.argmax(axis=2)
        one_hot = jax.nn.one_hot(
            state.cur_index.ravel() + 1,
            state.sequences.shape[1],
            dtype=state.sequences.dtype,
        )
        new_sequences = (
            state.sequences * (1 - one_hot)
            + next_tokens
            * (jnp.sum(state.sequences == eos_token_id, axis=-1) < 2).reshape(
                batch_size, 1
            )
            * one_hot
        )

        return DecodingState(
            cur_index=jnp.minimum(state.cur_index + 1, max_length - 1),
            sequences=new_sequences,
            cache=new_cache,
        )

    def loop_cond_func(state: DecodingState) -> jax.Array:
        return ~jnp.all(
            jnp.logical_or(
                state.cur_index >= max_length - 1,
                jnp.sum(state.sequences == eos_token_id, axis=-1, keepdims=True) >= 2,
            )
        )

    final_state = jax.lax.while_loop(loop_cond_func, loop_body_func, decode_state)
    return final_state.sequences
