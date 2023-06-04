from functools import partial
import jax
import optax
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

from data_loading import get_positive_reframing_dataset
from transformer import Transformer


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


BATCH_SIZE = 32
save_every = 100
num_length_buckets = 5


def main():
    with open("outputs/m.model", "rb") as f:
        tokenizer = tf_text.SentencepieceTokenizer(f.read(), add_eos=True)

    train_dataset, bucket_boundaries = get_positive_reframing_dataset(
        "data/train.csv", tokenizer, BATCH_SIZE, num_length_buckets=num_length_buckets
    )

    val_dataset, _ = get_positive_reframing_dataset(
        "data/dev.csv", tokenizer, BATCH_SIZE, num_length_buckets=num_length_buckets
    )

    max_length = bucket_boundaries[-1] - 1

    transformer = Transformer(max_length=max_length, vocab_size=tokenizer.vocab_size())

    # Create state
    batch = next(train_dataset.take(1).as_numpy_iterator())
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

    def loss_fn(params, state, batch):
        mask = batch["labels"] != 0
        logits = state.apply_fn({"params": params}, batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["labels"]
        )
        loss *= mask  # zero loss for padded tokens
        return loss.sum(), mask

    @partial(jax.jit, static_argnames=["sequence_length"])
    def train_step(state, batch, sequence_length):
        """Train for a single step."""

        value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, mask), grads = value_and_grad_fn(state.params, state, batch)
        state = state.apply_gradients(grads=grads)
        return state, loss, mask.sum()

    @partial(jax.jit, static_argnames=["sequence_length"])
    def eval_step(state, batch, sequence_length):
        """Train for a single step."""

        loss, mask = loss_fn(state.params, state, batch)
        return loss, mask.sum()

    checkpoint_manager = CheckpointManager(
        "model_saves",
        PyTreeCheckpointer(),
        CheckpointManagerOptions(max_to_keep=2, create=True),
    )

    step = 0

    lengths = []
    for epoch in range(5):
        total_loss = jnp.zeros((), dtype="float32")
        total_tokens = jnp.zeros((), dtype="int32")
        for batch in train_dataset.as_numpy_iterator():
            state, loss, tokens = train_step(
                state, batch, sequence_length=batch["inputs_ids"].shape[-1]
            )
            step += 1
            total_loss += loss
            total_tokens += tokens

            if step % save_every == 0:
                ckpt = {"model": state}
                checkpoint_manager.save(step, ckpt)

                print("step", step, "avg loss:", loss / tokens)
        print(f"epoch {epoch + 1} step {step} avg_loss {total_loss / total_tokens}")

        total_loss = jnp.zeros((), dtype="float32")
        total_tokens = jnp.zeros((), dtype="int32")
        for batch in val_dataset.as_numpy_iterator():
            batch_loss, batch_tokens = eval_step(
                state, batch, sequence_length=batch["inputs_ids"].shape[-1]
            )
            total_loss += batch_loss
            total_tokens += batch_tokens
        print(
            f"validation epoch {epoch + 1} step {step} avg_loss {total_loss / total_tokens}"
        )


if __name__ == "__main__":
    main()
