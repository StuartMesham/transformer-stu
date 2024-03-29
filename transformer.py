import jax.numpy as jnp
from flax import linen as nn
from jax import random

import caching
from type_annotations import Array


def make_prefix_lm_mask(
    tokens: Array, tokens_has_bidirectional_attention: Array, pad_token_idx: int
) -> Array:
    """Creates a prefix lm mask.

    Args:
        tokens: A batch of input tokens.
        tokens_has_bidirectional_attention: A mask of the same shape as `tokens` with 1 for tokens which should have
            bidirectional attention and 0 for tokens which should have causal attention.
        pad_token_idx: The ID of the padding token.

    Adapted from: https://github.com/google-research/t5x/blob/247d329f4da9506c515a564a52ef385146784fb1/t5x/examples/decoder_only/layers.py#L978.
    """
    causal_mask = nn.make_causal_mask(tokens)

    bidirectional_mask = nn.make_attention_mask(
        tokens_has_bidirectional_attention,
        tokens_has_bidirectional_attention,
        jnp.logical_and,
    )

    prefix_lm_mask = jnp.logical_or(causal_mask, bidirectional_mask)

    padding_mask = nn.make_attention_mask(
        tokens != pad_token_idx, tokens != pad_token_idx
    )

    return jnp.logical_and(padding_mask, prefix_lm_mask)


class MLP(nn.Module):
    """A Multilayer Perceptron with a ReLU activation."""

    hidden_size: int
    output_size: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Performs a forward pass of the MLP on input `x`."""
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_size)(x)
        return x


class EmbedTokens(nn.Module):
    """Transformer token embeddings with learned positional embeddings.

    The final embedding is the element-wise sum of the token and position embeddings.
    """

    vocab_size: int
    max_length: int
    emb_size: int
    decode: bool

    @nn.compact
    def __call__(self, token_ids: Array, position_ids: Array) -> Array:
        """Performs a forward pass of the EmbedTokens layer."""
        tok_emb = nn.Embed(self.vocab_size, self.emb_size)(token_ids)
        pos_emb = nn.Embed(self.max_length, self.emb_size)(position_ids)
        return tok_emb + pos_emb


class Transformer(nn.Module):
    """A simple transformer model."""

    max_length: int = 100
    vocab_size: int = 100
    emb_size: int = 64

    mlp_hidden_dim: int = 128
    num_layers: int = 2

    pad_token_idx: int = 0

    num_heads: int = 4

    dropout_rate: float = 0
    decode: bool = False

    @nn.compact
    def __call__(self, batch: dict[str, Array], eval_mode: bool = False) -> Array:
        """Perform a forwards pass of the Transformer model.

        Args:
            batch: A dictionary containing the keys: "token_ids", "position_ids" and "bidirectional_attention_mask".
            eval_mode: If True, dropout is disabled and the attention cache is enabled for efficient decoding.
                If False, dropout is enabled and the attention cache is disabled. This argument should be set to False
                during training.
        """
        token_ids = batch["token_ids"]
        position_ids = batch["position_ids"]

        attention_mask = make_prefix_lm_mask(
            token_ids, batch["bidirectional_attention_mask"], self.pad_token_idx
        )

        emb = EmbedTokens(self.vocab_size, self.max_length, self.emb_size, self.decode)(
            token_ids, position_ids
        )
        emb = nn.Dropout(self.dropout_rate, deterministic=eval_mode)(emb)

        for _ in range(self.num_layers):
            emb += nn.Dropout(self.dropout_rate, deterministic=eval_mode)(
                caching.SelfAttention(
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    use_bias=False,
                    broadcast_dropout=False,
                    deterministic=eval_mode,
                    decode=self.decode,
                )(emb, attention_mask)
            )
            emb = nn.LayerNorm()(emb)
            emb += nn.Dropout(self.dropout_rate, deterministic=eval_mode)(
                MLP(self.mlp_hidden_dim, self.emb_size)(emb)
            )
            emb = nn.LayerNorm()(emb)
        logits = nn.Dense(self.vocab_size)(emb)

        return logits


if __name__ == "__main__":
    max_length = 150
    transformer = Transformer()
    tokens_has_bidirectional_attention = jnp.array(
        [1, 1, 1, 1] + [0] * (max_length - 4), dtype="uint32"
    )
    weights = transformer.init(
        random.PRNGKey(0),
        jnp.ones(max_length, dtype="uint32"),
        tokens_has_bidirectional_attention,
    )
    print(
        transformer.apply(
            weights,
            jnp.ones(max_length, dtype="uint32"),
            tokens_has_bidirectional_attention,
        ).shape
    )
