import jax
import optax
import tensorflow as tf
import tensorflow_text as tf_text
from clu import metrics
from flax import struct
from jax import random
import jax.numpy as jnp
from flax.training import train_state
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)

from transformer import Transformer

with open("outputs/m.model", "rb") as f:
    tokenizer = tf_text.SentencepieceTokenizer(f.read(), add_eos=True)


def tokenize_input_target_pair(input, target):
    return tokenizer.tokenize(input), tokenizer.tokenize(target)


def convert_to_prefix_lm_example(input, target):
    return {
        "inputs_ids": tf.concat((input, target[:-1]), axis=0),
        "labels": tf.concat((tf.zeros_like(input)[:-1], target), axis=0),
        "bidirectional_attention_mask": tf.concat(
            (tf.ones_like(input), tf.zeros_like(target[:-1])), axis=0
        ),
    }


data = (
    tf.data.experimental.CsvDataset(
        "data/train.csv",
        record_defaults=["", ""],
        select_cols=[0, 1],
        header=True,
    )
    .map(tokenize_input_target_pair)
    .map(convert_to_prefix_lm_example)
)

max_length = 170
BATCH_SIZE = 32

data = data.bucket_by_sequence_length(
    element_length_func=lambda elem: tf.shape(elem["inputs_ids"])[0],
    bucket_boundaries=[41, 61, 171],
    pad_to_bucket_boundary=True,
    drop_remainder=True,
    bucket_batch_sizes=[BATCH_SIZE, BATCH_SIZE, BATCH_SIZE, BATCH_SIZE],
)

count = 0
transformer = Transformer(max_length=max_length, vocab_size=tokenizer.vocab_size())


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


# Create state
batch = next(data.take(1).as_numpy_iterator())
state = TrainState.create(
    apply_fn=transformer.apply,
    params=transformer.init(
        random.PRNGKey(0),
        {
            k: jnp.zeros((BATCH_SIZE, max_length), dtype=int)
            for k in ["inputs_ids", "bidirectional_attention_mask"]
        },
    )["params"],
    tx=optax.adamw(learning_rate=0.001),
    metrics=Metrics.empty(),
)


@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    def loss_fn(params):
        mask = batch["labels"] != 0
        logits = state.apply_fn({"params": params}, batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["labels"]
        )
        loss *= mask  # zero loss for padded tokens
        return loss.sum(), mask

    value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, mask), grads = value_and_grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, mask.sum()


checkpoint_manager = CheckpointManager(
    "model_saves",
    PyTreeCheckpointer(),
    CheckpointManagerOptions(max_to_keep=2, create=True),
)


step = 0

save_every = 50

lengths = []
for batch in data.as_numpy_iterator():
    state, loss, tokens = train_step(state, batch)
    step += 1

    if step % save_every == 0:
        ckpt = {"model": state}
        checkpoint_manager.save(step, ckpt)

    print("step", step, "avg loss:", loss / tokens)
