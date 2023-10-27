import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_text as tf_text
from orbax.checkpoint import PyTreeCheckpointer
from transformer import Transformer

checkpointer = PyTreeCheckpointer()
restored_params = checkpointer.restore("model_saves/70/default")

print(restored_params.keys())

transformer = Transformer(max_length=522, vocab_size=10149, emb_size=512, mlp_hidden_dim=1024, num_layers=15,
                          num_heads=4, dropout_rate=0.1, decode=True)

with open("model_saves/70/en-de.model", "rb") as f:
    tokenizer = tf_text.SentencepieceTokenizer(f.read(), add_eos=True)

input = tokenizer.tokenize("i would really like a cup of tea.")
input = tf.expand_dims(input, axis=0)

batch = {
    "inputs_ids": jnp.pad(input.numpy(), pad_width=((0, 0), (0, 50)), mode="constant", constant_values=0),
    "bidirectional_attention_mask": jnp.pad(tf.ones_like(input).numpy(), pad_width=((0, 0), (0, 50)), mode="constant", constant_values=0),
}

logits, initial_variables = transformer.apply(
    {"params": restored_params},
batch,
    eval_mode=True,
    mutable=True
)
cache = initial_variables["cache"]


cache = jax.tree_map(lambda x: dict(x, cache_index=jnp.array(input.shape[1], dtype=jnp.int32)), cache, is_leaf=lambda x: "cache_index" in x)

new_batch = {
    "inputs_ids": logits.argmax(axis=2)[:,11].reshape(1, 1),
    "bidirectional_attention_mask": jnp.array([[0]], dtype="int32"),
}

outputs = [int(new_batch["inputs_ids"])]

while True:
    print(tokenizer.detokenize(outputs).numpy().decode())
    logits, new_vars = transformer.apply({"params": restored_params, "cache": cache}, new_batch, mutable=["cache"], eval_mode=True)
    new_batch["inputs_ids"] = logits.argmax(axis=2)
    cache = new_vars["cache"]
    outputs.append(int(new_batch["inputs_ids"]))
    
    if int(new_batch["inputs_ids"]) == 2:
        break
