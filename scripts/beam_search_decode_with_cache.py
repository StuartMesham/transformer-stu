import jax
import jax.numpy as jnp
import tensorflow_text as tf_text
from orbax.checkpoint import PyTreeCheckpointer

from decoding.beam_search import beam_search, flatten_beam_dim
from decoding.utils import DecodingState
from transformer import Transformer
from type_annotations import Array, PyTree

MAX_LENGTH = 25
EOS_TOKEN_ID = 2


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

    def sequences_to_logits(sequences: Array) -> tuple[Array, PyTree]:
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

        return logits, cache

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

    output_sequences, output_scores = beam_search(
        sequences,
        decoding_start_index,
        sequences_to_logits,
        tokens_to_logits,
        EOS_TOKEN_ID,
    )

    print(jnp.array_str(flatten_beam_dim(output_sequences), max_line_width=59999))

    for decoded_string in tokenizer.detokenize(
        flatten_beam_dim(output_sequences)
    ).numpy():
        print(decoded_string.decode())
    print()


if __name__ == "__main__":
    main()
