from orbax.checkpoint import PyTreeCheckpointer
from transformer import Transformer
import jax.numpy as jnp
import tensorflow_text as tf_text
import tensorflow as tf

checkpointer = PyTreeCheckpointer()
restored_params = checkpointer.restore("model_saves/70/default")

transformer = Transformer(max_length=522, vocab_size=10149, emb_size=512, mlp_hidden_dim=1024, num_layers=15,
                          num_heads=4, dropout_rate=0.1, decode=False)

with open("model_saves/70/en-de.model", "rb") as f:
    tokenizer = tf_text.SentencepieceTokenizer(f.read(), add_eos=True)

input = tokenizer.tokenize("i would really like a cup of tea.")
input = tf.expand_dims(input, axis=0)

batch = {
    "inputs_ids": input.numpy(),
    "bidirectional_attention_mask": tf.ones_like(input).numpy(),
}

while True:
    logits = transformer.apply({"params": restored_params}, batch, eval_mode=True)
    batch = {
        "inputs_ids": jnp.hstack((batch["inputs_ids"], logits.argmax(axis=2)[0, -1].reshape(1, 1))),
        "bidirectional_attention_mask": jnp.hstack((batch["bidirectional_attention_mask"], jnp.zeros((1, 1), "int32")))
    }
    if batch["inputs_ids"][0, -1] == 2:  # if model outputs EOS token
        break
    print(tokenizer.detokenize(batch["inputs_ids"]).numpy()[0].decode())
