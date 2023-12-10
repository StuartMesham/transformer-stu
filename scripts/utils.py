"""Adapted from t5x.decoding module."""
from collections.abc import Mapping

import flax
import jax.numpy as jnp


@flax.struct.dataclass
class DecodingState:
    """Holds decoding state data.

    Used to communicate the current decoding state to tokens_to_logits methods.
    Note that we use a different class than `SamplingLoopState` or `Beamstate` to
    decouple the concerns of what data is useful for the loop vs. what the
    sampling method needs.
    Decodes for a given batch entry are flattened in a column-major way so that
    decodes from the same batch entry are grouped together.

    Attributes:
      cur_index: [batch_size * num_decodes] array position of the sampling loop in
        the length dimension.
      sequences: [batch_size * num_decodes, max_decode_len] array of current
        sampled sequence prefixes.
      cache: any mapping of arrays, e.g. flax attention cache.
    """

    cur_index: jnp.ndarray
    sequences: jnp.ndarray
    cache: Mapping[str, jnp.ndarray]


@flax.struct.dataclass
class BeamDecodingState(DecodingState):
    sequence_log_probs: jnp.ndarray
    sequence_is_terminated: jnp.ndarray
    sequence_lengths: jnp.ndarray


def brevity_penalty(alpha: float, length: int) -> jnp.ndarray:
    """Brevity penalty function for beam search penalizing short sequences.

    Args:
      alpha: float: brevity-penalty scaling parameter.
      length: int: length of considered sequence.

    Returns:
      Brevity penalty score as jax scalar.
    """
    return jnp.power(((5.0 + length) / 6.0), alpha)
