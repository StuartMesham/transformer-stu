from functools import partial
import click
import jax
import optax
import tensorflow_text as tf_text
import wandb
from clu import metrics
from flax import struct
from tqdm.auto import tqdm
from jax import random
import jax.numpy as jnp
from flax.training import train_state
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)

from data_loading import get_translation_dataset, bucket
from transformer import Transformer


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


@click.command()
@click.option("--tokenizer_file", required=True)
@click.option("--train_inputs", required=True)
@click.option("--train_targets", required=True)
@click.option("--val_inputs", required=True)
@click.option("--val_targets", required=True)
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
@click.option("--label_smoothing_mass", default=0.0)
def main(**kwargs):
    wandb.init(config=kwargs)
    config = wandb.config

    with open(config.tokenizer_file, "rb") as f:
        tokenizer = tf_text.SentencepieceTokenizer(f.read(), add_eos=True)

    train_dataset = get_translation_dataset(
        config.train_inputs,
        config.train_targets,
        tokenizer,
    )
    train_dataset, bucket_boundaries = bucket(
        train_dataset,
        config.batch_size,
        num_length_buckets=config.num_length_buckets,
    )

    val_dataset = get_translation_dataset(
        config.val_inputs,
        config.val_targets,
        tokenizer,
    )
    val_dataset, _ = bucket(
        val_dataset,
        config.batch_size,
        num_length_buckets=config.num_length_buckets,
    )

    max_length = bucket_boundaries[-1] - 1

    transformer = Transformer(
        max_length=max_length,
        vocab_size=tokenizer.vocab_size(),
        emb_size=config.emb_size,
        mlp_hidden_dim=config.mlp_hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
    )

    # Create state
    state = TrainState.create(
        apply_fn=transformer.apply,
        params=transformer.init(
            random.PRNGKey(0),
            {
                k: jnp.zeros((config.batch_size, max_length), dtype=int)
                for k in ["inputs_ids", "bidirectional_attention_mask"]
            },
        )["params"],
        tx=optax.adamw(config.learning_rate),
        metrics=Metrics.empty(),
    )

    def loss_fn(params, state, batch):
        mask = batch["labels"] != 0
        logits = state.apply_fn({"params": params}, batch)
        if config.label_smoothing_mass:
            labels = optax.smooth_labels(
                jax.nn.one_hot(batch["labels"], logits.shape[-1]),
                config.label_smoothing_mass,
            )
            loss = optax.softmax_cross_entropy(logits=logits, labels=labels)
        else:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=batch["labels"]
            )
        loss *= mask  # zero loss for padded tokens
        return loss.sum(), mask

    @partial(jax.jit, static_argnames=["sequence_length", "batch_size"])
    def train_step(state, batch, sequence_length, batch_size):
        """Train for a single step."""

        value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, mask), grads = value_and_grad_fn(state.params, state, batch)
        state = state.apply_gradients(grads=grads)
        return state, loss, mask.sum()

    @partial(jax.jit, static_argnames=["sequence_length", "batch_size"])
    def eval_step(state, batch, sequence_length, batch_size):
        """Train for a single step."""

        loss, mask = loss_fn(state.params, state, batch)
        return loss, mask.sum()

    checkpoint_manager = CheckpointManager(
        config.model_save_dir,
        PyTreeCheckpointer(),
        CheckpointManagerOptions(max_to_keep=2, create=True),
    )

    print("starting train loop")

    steps = 0
    for epoch in range(config.num_epochs):
        total_loss = jnp.zeros((), dtype="float32")
        total_tokens = jnp.zeros((), dtype="int32")
        for batch in tqdm(train_dataset.as_numpy_iterator()):
            state, loss, tokens = train_step(
                state,
                batch,
                sequence_length=batch["inputs_ids"].shape[-1],
                batch_size=batch["inputs_ids"].shape[0],
            )
            steps += 1
            total_loss += loss
            total_tokens += tokens

        print(f"epoch {epoch + 1} steps {steps} avg_loss {total_loss / total_tokens}")
        wandb.log(
            {
                "train/epoch": epoch + 1,
                "train/mean_per_token_loss": total_loss / total_tokens,
            },
            step=steps,
        )
        if (epoch + 1) % config.save_every == 0:
            ckpt = {"model": state}
            checkpoint_manager.save(epoch + 1, ckpt)

            print("saving checkpoint for epoch", epoch + 1)

        if (epoch + 1) % config.eval_every == 0:
            total_loss = jnp.zeros((), dtype="float32")
            total_tokens = jnp.zeros((), dtype="int32")
            for batch in val_dataset.as_numpy_iterator():
                batch_loss, batch_tokens = eval_step(
                    state,
                    batch,
                    sequence_length=batch["inputs_ids"].shape[-1],
                    batch_size=batch["inputs_ids"].shape[0],
                )
                total_loss += batch_loss
                total_tokens += batch_tokens
            print(
                f"validation epoch {epoch + 1} steps {steps} avg_loss {total_loss / total_tokens}"
            )
            wandb.log(
                {
                    "val/mean_per_token_loss": total_loss / total_tokens,
                },
                step=steps,
            )


if __name__ == "__main__":
    main()
