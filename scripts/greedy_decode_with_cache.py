import jax
import jax.numpy as jnp
import tensorflow_text as tf_text
from orbax.checkpoint import PyTreeCheckpointer
from scripts.utils import DecodingState
from transformer import Transformer

MAX_LENGTH = 50
EOS_TOKEN_ID = 2


def main():
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
    decoding_start_index = token_ids_ragged_tensor.row_lengths().numpy().max()
    sequences = token_ids_ragged_tensor.to_tensor(
        default_value=0, shape=[batch_size, MAX_LENGTH]
    ).numpy()

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
    cache_mask = jnp.logical_or(
        sequences > 0, batch["position_ids"] >= decoding_start_index
    ).reshape(batch_size, 1, 1, MAX_LENGTH)
    cache = jax.tree_map(
        lambda x: dict(
            x,
            cache_index=jnp.array(decoding_start_index, dtype=jnp.int32),
            cache_mask=cache_mask,
        ),
        cache,
        is_leaf=lambda x: "cached_value" in x,
    )

    sequences[
        jnp.arange(batch_size), token_ids_ragged_tensor.row_lengths().numpy()
    ] = logits.argmax(axis=2)[
        jnp.arange(batch_size), token_ids_ragged_tensor.row_lengths().numpy() - 1
    ]

    decode_state = DecodingState(
        cur_index=token_ids_ragged_tensor.row_lengths().numpy().reshape(batch_size, -1),
        sequences=sequences,
        cache=cache,
    )

    def loop_body_func(state: DecodingState) -> DecodingState:
        batch = {
            "token_ids": state.sequences[
                jnp.arange(batch_size), state.cur_index.ravel()
            ].reshape(batch_size, 1),
            "position_ids": state.cur_index,
            "bidirectional_attention_mask": jnp.zeros((batch_size, 1), dtype="int32"),
        }
        logits, new_vars = transformer.apply(
            {"params": restored_params, "cache": state.cache},
            batch,
            mutable=["cache"],
            eval_mode=True,
        )
        new_cache = new_vars["cache"]

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
            * (jnp.sum(state.sequences == EOS_TOKEN_ID, axis=-1) < 2).reshape(
                batch_size, 1
            )
            * one_hot
        )

        return DecodingState(
            cur_index=state.cur_index + 1,
            sequences=new_sequences,
            cache=new_cache,
        )

    def loop_cond_func(state: DecodingState) -> jax.Array:
        return ~jnp.logical_or(
            jnp.any(state.cur_index >= MAX_LENGTH - 1),
            jnp.all(
                jnp.sum(state.sequences == EOS_TOKEN_ID, axis=-1, keepdims=True) >= 2
            ),
        )

    final_state = jax.lax.while_loop(loop_cond_func, loop_body_func, decode_state)

    for decoded_string in tokenizer.detokenize(final_state.sequences).numpy():
        print(decoded_string.decode())
    print()


if __name__ == "__main__":
    main()
