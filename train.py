import os.path
import shutil
from functools import partial
from glob import glob

import click
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_text as tf_text
import wandb
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from jax import random
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)
from tqdm.auto import tqdm

from data_loading import bucket, get_translation_dataset
from transformer import Transformer
from type_annotations import Array, PRNGKeyLike, PyTree


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
@click.option("--dropout_rate", default=0.1)
@click.option("--label_smoothing_mass", default=0.0)
@click.option("--warmup_steps", default=1000)
@click.option("--early_stopping_patience", default=1)
@click.option("--train_bucket_boundaries", type=str, required=False)
@click.option("--validation_bucket_boundaries", type=str, required=False)
def main(**kwargs: bool | str | int) -> None:
    """Performs a single training run."""
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
        bucket_boundaries=[int(b) for b in config.train_bucket_boundaries.split(",")]
        if config.train_bucket_boundaries
        else None,
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
        bucket_boundaries=[
            int(b) for b in config.validation_bucket_boundaries.split(",")
        ]
        if config.validation_bucket_boundaries
        else None,
    )

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    max_length = bucket_boundaries[-1] - 1

    transformer = Transformer(
        max_length=max_length,
        vocab_size=tokenizer.vocab_size(),
        emb_size=config.emb_size,
        mlp_hidden_dim=config.mlp_hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout_rate=config.dropout_rate,
    )

    lr_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps,
    )

    key, params_key = random.split(random.PRNGKey(0))

    # Create state
    state = TrainState.create(
        apply_fn=transformer.apply,
        params=transformer.init(
            params_key,
            {
                k: jnp.zeros((config.batch_size, max_length), dtype=int)
                for k in ["token_ids", "position_ids", "bidirectional_attention_mask"]
            },
            eval_mode=True,
        )["params"],
        tx=optax.adamw(lr_schedule),
    )

    def loss_fn(
        params: PyTree,
        state: TrainState,
        batch: dict[str, Array],
        dropout_key: PRNGKeyLike = None,
        eval_mode: bool = False,
    ) -> tuple[Array, Array]:
        print(type(dropout_key))
        mask = batch["labels"] != 0
        logits = state.apply_fn(
            {"params": params},
            batch
            | {
                "position_ids": jnp.broadcast_to(
                    jnp.arange(0, batch["token_ids"].shape[1]), batch["token_ids"].shape
                )
            },
            eval_mode,
            rngs={"dropout": dropout_key} if not eval_mode else None,
        )
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
    def train_step(
        state: PyTree,
        batch: dict[str, Array],
        dropout_key: PRNGKeyLike,
        sequence_length: int,
        batch_size: int,
    ) -> tuple[PyTree, Array, Array]:
        """Train for a single step.

        Args:
            state: The state of the model to be trained.
            batch: A dictionary containing the token_ids, labels and bidirectional_attention_mask to be trained on.
            dropout_key: The PRNGKey to be used for dropout layers.
            sequence_length: The sequence length of the batch.
            batch_size: The number of sequences in the batch.

        Returns:
            A tuple containing the updated state of the model, the losses for the training step, and the number of
            non-padding tokens in the batch
        """
        value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, mask), grads = value_and_grad_fn(state.params, state, batch, dropout_key)
        state = state.apply_gradients(grads=grads)
        return state, loss, mask.sum()

    @partial(jax.jit, static_argnames=["sequence_length", "batch_size"])
    def eval_step(
        state: PyTree, batch: dict[str, Array], sequence_length: int, batch_size: int
    ) -> tuple[Array, Array]:
        """Eval on a single batch.

        Args:
            state: The state of the model to be evaluated.
            batch: A dictionary containing the token_ids, labels and bidirectional_attention_mask to be evaluated on.
            sequence_length: The sequence length of the batch.
            batch_size: The number of sequences in the batch.

        Returns:
            A tuple containing the total loss, and number of non-padding tokens in the batch.
        """
        loss, mask = loss_fn(state.params, state, batch, eval_mode=True)
        return loss, mask.sum()

    checkpoint_manager = CheckpointManager(
        config.model_save_dir,
        PyTreeCheckpointer(),
        CheckpointManagerOptions(
            max_to_keep=1,
            create=True,
            best_mode="min",
            best_fn=lambda metrics: metrics["val/mean_per_token_loss"],
        ),
    )

    print("starting train loop")

    steps = 0
    early_stop = EarlyStopping(patience=config.early_stopping_patience)
    for epoch in range(config.num_epochs):
        total_loss = jnp.zeros((), dtype="float32")
        total_tokens = jnp.zeros((), dtype="int32")
        for batch in tqdm(train_dataset.as_numpy_iterator()):
            key, dropout_key = random.split(key)
            state, loss, tokens = train_step(
                state,
                batch,
                dropout_key,
                sequence_length=batch["token_ids"].shape[-1],
                batch_size=batch["token_ids"].shape[0],
            )
            steps += 1
            total_loss += loss
            total_tokens += tokens

        metrics = {
            "train/epoch": epoch + 1,
            "train/mean_per_token_loss": (total_loss / total_tokens).item(),
            "train/learning_rate": lr_schedule(steps).item(),
        }

        if (epoch + 1) % config.eval_every == 0:
            total_loss = jnp.zeros((), dtype="float32")
            total_tokens = jnp.zeros((), dtype="int32")
            for batch in val_dataset.as_numpy_iterator():
                batch_loss, batch_tokens = eval_step(
                    state,
                    batch,
                    sequence_length=batch["token_ids"].shape[-1],
                    batch_size=batch["token_ids"].shape[0],
                )
                total_loss += batch_loss
                total_tokens += batch_tokens
            metrics["val/mean_per_token_loss"] = (total_loss / total_tokens).item()

            _, early_stop = early_stop.update(metrics["val/mean_per_token_loss"])

            if (epoch + 1) % config.save_every == 0:
                ckpt = state.params
                checkpoint_manager.save(epoch + 1, ckpt, metrics=metrics)

                print("saving checkpoint for epoch", epoch + 1)

        print(f"steps: {steps}, metrics: {metrics}")
        wandb.log(metrics, step=steps)

        if early_stop.should_stop:
            print("stopping early")
            break

    for file_name in glob(os.path.join(config.model_save_dir, "*")):
        print("zipping", file_name)
        shutil.make_archive(file_name, "zip", file_name)
        wandb.save(f"{file_name}.zip")

    wandb.save(config.tokenizer_file)


if __name__ == "__main__":
    main()
