from functools import partial
import click
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


@click.command()
@click.option("--tokenizer_file", required=True)
@click.option("--train_data", required=True)
@click.option("--val_data", required=True)
@click.option("--model_save_dir", default="model_saves")
@click.option("--batch_size", default=32)
@click.option("--num_epochs", default=5)
@click.option("--save_every", default=1)
@click.option("--eval_every", default=1)
@click.option("--num_length_buckets", default=5)
@click.option("--learning_rate", default=0.001)
@click.option("--emb_size", default=64)
@click.option("--mlp_hidden_dim", default=128)
@click.option("--num_layers", default=2)
@click.option("--num_heads", default=4)
def main(
    tokenizer_file,
    train_data,
    val_data,
    model_save_dir,
    batch_size,
    num_epochs,
    save_every,
    eval_every,
    num_length_buckets,
    learning_rate,
    emb_size,
    mlp_hidden_dim,
    num_layers,
    num_heads,
):
    with open(tokenizer_file, "rb") as f:
        tokenizer = tf_text.SentencepieceTokenizer(f.read(), add_eos=True)

    train_dataset, bucket_boundaries = get_positive_reframing_dataset(
        train_data, tokenizer, batch_size, num_length_buckets=num_length_buckets
    )

    val_dataset, _ = get_positive_reframing_dataset(
        val_data, tokenizer, batch_size, num_length_buckets=num_length_buckets
    )

    max_length = bucket_boundaries[-1] - 1

    transformer = Transformer(
        max_length=max_length,
        vocab_size=tokenizer.vocab_size(),
        emb_size=emb_size,
        mlp_hidden_dim=mlp_hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    # Create state
    state = TrainState.create(
        apply_fn=transformer.apply,
        params=transformer.init(
            random.PRNGKey(0),
            {
                k: jnp.zeros((batch_size, max_length), dtype=int)
                for k in ["inputs_ids", "bidirectional_attention_mask"]
            },
        )["params"],
        tx=optax.adamw(learning_rate),
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
        model_save_dir,
        PyTreeCheckpointer(),
        CheckpointManagerOptions(max_to_keep=2, create=True),
    )

    print("starting train loop")

    steps = 0
    for epoch in range(num_epochs):
        total_loss = jnp.zeros((), dtype="float32")
        total_tokens = jnp.zeros((), dtype="int32")
        for batch in train_dataset.as_numpy_iterator():
            state, loss, tokens = train_step(
                state, batch, sequence_length=batch["inputs_ids"].shape[-1]
            )
            steps += 1
            total_loss += loss
            total_tokens += tokens

        print(f"epoch {epoch + 1} steps {steps} avg_loss {total_loss / total_tokens}")
        if (epoch + 1) % save_every == 0:
            ckpt = {"model": state}
            checkpoint_manager.save(epoch + 1, ckpt)

            print("saving checkpoint for epoch", epoch + 1)

        if (epoch + 1) % eval_every == 0:
            total_loss = jnp.zeros((), dtype="float32")
            total_tokens = jnp.zeros((), dtype="int32")
            for batch in val_dataset.as_numpy_iterator():
                batch_loss, batch_tokens = eval_step(
                    state, batch, sequence_length=batch["inputs_ids"].shape[-1]
                )
                total_loss += batch_loss
                total_tokens += batch_tokens
            print(
                f"validation epoch {epoch + 1} steps {steps} avg_loss {total_loss / total_tokens}"
            )


if __name__ == "__main__":
    main()
