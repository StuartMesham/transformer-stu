from orbax.checkpoint import PyTreeCheckpointer
from transformer import Transformer
import jax.numpy as jnp
import tensorflow_text as tf_text


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
        decode=False,
    )

    with open("model_saves/70/en-de.model", "rb") as f:
        tokenizer = tf_text.SentencepieceTokenizer(f.read(), add_eos=True)

    token_ids = tokenizer.tokenize(["i would really like a cup of tea."]).numpy()

    batch = {
        "token_ids": token_ids,
        "position_ids": jnp.broadcast_to(
            jnp.arange(0, token_ids.shape[1]), token_ids.shape
        ),
        "bidirectional_attention_mask": jnp.ones_like(token_ids),
    }

    while True:
        logits = transformer.apply({"params": restored_params}, batch, eval_mode=True)
        token_ids = jnp.hstack(
            (batch["token_ids"], logits.argmax(axis=2)[0, -1].reshape(1, 1))
        )
        batch = {
            "token_ids": token_ids,
            "position_ids": jnp.broadcast_to(
                jnp.arange(0, token_ids.shape[1]), token_ids.shape
            ),
            "bidirectional_attention_mask": jnp.hstack(
                (batch["bidirectional_attention_mask"], jnp.zeros((1, 1), "int32"))
            ),
        }
        if batch["token_ids"][0, -1] == 2:  # if model outputs EOS token
            break
        print(tokenizer.detokenize(batch["token_ids"]).numpy()[0].decode())


if __name__ == "__main__":
    main()
