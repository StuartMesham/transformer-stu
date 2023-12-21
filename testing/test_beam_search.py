import numpy as np
from jax import numpy as jnp

from decoding.beam_search import beam_search
from decoding.utils import DecodingState
from type_annotations import Array, PyTree


def test_beam_search() -> None:
    """Test the beam search function."""
    # Toy problem, we have 4 states, A, B, START, END, (plus PAD).
    # Scores are given by a first-order Markov model.
    # PAD doesn't matter for this test, but part of the contract for beam_search is giving the PAD token id 0.
    # states = ["PAD", "A", "B", "START-", "-END"]

    # Edge potentials (written inside edges for diagonals):
    #            1      -1     1      -1
    #         A ---- A ---- A ---- A ---- A
    #       1   \  -1  \  1   \  -1  \  1   0
    # START      X      X      X      X       END
    #       0.9 /  -1  /  1   /  -1  /  1   0
    #         B ---- B ---- B ---- B ---- B
    #            5      -1     1      -1

    # put the above edge potentials in a 3-tensor
    ab_edge_potentials = np.asarray(
        [[[1, -1], [-1, 5]], [[-1, 1], [1, -1]], [[1, -1], [-1, 1]], [[-1, 1], [1, -1]]]
    )

    # now we have to add on the START, END states
    # and PAD at 0
    edge_potentials = np.ones([6, 5, 5]) * np.array(-1.0e7)
    edge_potentials[1:5, 1:3, 1:3] = ab_edge_potentials
    # START can go to either A or B for free at t0
    edge_potentials[0, 3, 1] = 1
    edge_potentials[0, 3, 2] = 0.9
    # either A or B can go to END for free at t5
    edge_potentials[5, 1, 4] = 0
    edge_potentials[5, 2, 4] = 0
    # PAD can go to anything for free (doesn't matter for this test)
    edge_potentials[:, 0, :] = 0

    edge_potentials = jnp.asarray(edge_potentials)

    def sequences_to_logits(sequences: Array) -> tuple[Array, PyTree]:
        logits = (
            edge_potentials.at[
                jnp.arange(sequences.size) % sequences.shape[1], sequences.ravel()
            ]
            .get()
            .reshape(sequences.shape + (-1,))
        )
        return logits, {}

    def tokens_to_logits(state: DecodingState) -> tuple[Array, PyTree]:
        logits = jnp.expand_dims(
            edge_potentials.at[
                state.cur_index.ravel(),
                state.sequences.at[
                    jnp.arange(state.sequences.shape[0]), state.cur_index.ravel()
                ].get(),
            ].get(),
            axis=1,
        )
        return logits, {}

    outputs = beam_search(
        jnp.array([[3, 1, 0, 0, 0, 0, 0], [3, 2, 0, 0, 0, 0, 0]]),
        decoding_start_index=jnp.array([[2], [1]]),
        sequences_to_logits=sequences_to_logits,
        tokens_to_logits=tokens_to_logits,
        eos_token_id=4,
        beams=2,
    )

    assert jnp.array_equal(
        outputs[0],
        jnp.array(
            [
                [[3, 1, 1, 2, 2, 1, 4], [3, 1, 1, 2, 2, 2, 4]],
                [[3, 2, 2, 1, 1, 2, 4], [3, 2, 2, 1, 1, 1, 4]],
            ]
        ),
    )

    assert jnp.array_equal(
        outputs[1], jnp.array([[-0.5077122, -2.5077124], [-3.152109, -7.127656]])
    )

    # greedy decode
    outputs = beam_search(
        jnp.array([[3, 2, 0, 0, 0, 0, 0], [3, 2, 0, 0, 0, 0, 0]]),
        decoding_start_index=jnp.array([[2], [1]]),
        sequences_to_logits=sequences_to_logits,
        tokens_to_logits=tokens_to_logits,
        eos_token_id=4,
        beams=1,
    )

    assert jnp.array_equal(
        outputs[0],
        jnp.array(
            [
                [[3, 2, 2, 1, 1, 2, 4]],
                [[3, 1, 1, 2, 2, 1, 4]],
            ]
        ),
    )

    assert jnp.array_equal(outputs[1], jnp.array([[-0.38325977], [-1.1521089]]))
