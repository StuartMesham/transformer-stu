import flax
import jax.numpy as jnp
from typing import Mapping


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
    cur_token: [batch_size * num_decodes] single timestep slice containing
      current tokens.
    cache: any mapping of arrays, e.g. flax attention cache.
  """
  cur_index: jnp.ndarray
  sequences: jnp.ndarray
  cache: Mapping[str, jnp.ndarray]
