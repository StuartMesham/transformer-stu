import jax
import jax.numpy as jnp
import tensorflow_text as tf_text
from orbax.checkpoint import PyTreeCheckpointer
from transformer import Transformer

MAX_LENGTH = 50
EOS_TOKEN_ID = 2

checkpointer = PyTreeCheckpointer()
restored_params = checkpointer.restore("model_saves/70/default")

transformer = Transformer(max_length=522, vocab_size=10149, emb_size=512, mlp_hidden_dim=1024, num_layers=15,
                          num_heads=4, dropout_rate=0.1, decode=True)

with open("model_saves/70/en-de.model", "rb") as f:
    tokenizer = tf_text.SentencepieceTokenizer(f.read(), add_eos=True)

token_ids_ragged_tensor = tokenizer.tokenize(["i would really like a cup of tea.", "please give me one."])
batch_size = token_ids_ragged_tensor.shape[0]
decoding_start_index = token_ids_ragged_tensor.row_lengths().numpy().max()
sequences = token_ids_ragged_tensor.to_tensor(default_value=0, shape=[batch_size, MAX_LENGTH]).numpy()

batch = {
    "token_ids": sequences,
    "position_ids": jnp.broadcast_to(jnp.arange(0, sequences.shape[1]), sequences.shape),
    "bidirectional_attention_mask": sequences != 0,
}

logits, initial_variables = transformer.apply(
    {"params": restored_params},
batch,
    eval_mode=True,
    mutable=True
)

cache = initial_variables["cache"]
cache_mask = jnp.logical_or(sequences > 0, batch["position_ids"] >= decoding_start_index).reshape(batch_size, 1, 1, MAX_LENGTH)
cache = jax.tree_map(lambda x: dict(x, cache_index=jnp.array(decoding_start_index, dtype=jnp.int32), cache_mask=cache_mask), cache, is_leaf=lambda x: "cached_value" in x)

new_batch = {
    "token_ids": logits.argmax(axis=2)[jnp.arange(batch_size),token_ids_ragged_tensor.row_lengths().numpy()-1].reshape(batch_size, -1),
    "position_ids": token_ids_ragged_tensor.row_lengths().numpy().reshape(batch_size, -1),
    "bidirectional_attention_mask": jnp.zeros((batch_size, 1), dtype="int32"),
}

sequences[jnp.arange(batch_size), token_ids_ragged_tensor.row_lengths().numpy()] = new_batch["token_ids"].ravel()

while True:
    for decoded_string in tokenizer.detokenize(sequences).numpy():
        print(decoded_string.decode())
    print()

    logits, new_vars = transformer.apply({"params": restored_params, "cache": cache}, new_batch, mutable=["cache"], eval_mode=True)
    new_batch["token_ids"] = logits.argmax(axis=2)
    new_batch["position_ids"] = new_batch["position_ids"] + 1
    cache = new_vars["cache"]
    sequences[jnp.arange(batch_size), new_batch["position_ids"].ravel()] = new_batch["token_ids"].ravel() * (jnp.sum(sequences == EOS_TOKEN_ID, axis=-1) < 2)
    
    if jnp.any(new_batch["position_ids"] >= MAX_LENGTH - 1) or jnp.all(jnp.sum(sequences == EOS_TOKEN_ID, axis=-1, keepdims=True) >= 2):
        break
