import functools

import numpy as np
import tensorflow as tf

import data_loading


def test_eager_mode_convert_to_prefix_lm_example():
    """Tests the _convert_to_prefix_lm_example method in tensorflow's eager mode."""
    assert data_loading._EOS_TOKEN == 2
    assert data_loading._MASK_TOKEN == 0

    input = tf.constant([3, 5, 7, 7, 2])
    target = tf.constant([8, 3, 5, 2])
    vocab_size = 5000
    tf.random.set_seed(1235)

    output = data_loading._convert_to_prefix_lm_example(input, target, vocab_size)

    np.testing.assert_array_equal(
        output["inputs_ids"].numpy(), [1466, 5, 7, 7, 2, 8, 3, 5]
    )

    np.testing.assert_array_equal(output["labels"].numpy(), [3, 5, 7, 7, 8, 3, 5, 2])

    np.testing.assert_array_equal(
        output["bidirectional_attention_mask"].numpy(), [1, 1, 1, 1, 1, 0, 0, 0]
    )

    output = data_loading._convert_to_prefix_lm_example(input, target, vocab_size)

    np.testing.assert_array_equal(
        output["inputs_ids"].numpy(), [3, 5, 7, 0, 2, 8, 3, 5]
    )

    np.testing.assert_array_equal(output["labels"].numpy(), [3, 5, 7, 7, 8, 3, 5, 2])

    np.testing.assert_array_equal(
        output["bidirectional_attention_mask"].numpy(), [1, 1, 1, 1, 1, 0, 0, 0]
    )


def test_graph_mode_convert_to_prefix_lm_example():
    """Tests the _convert_to_prefix_lm_example method in tensorflow's graph mode."""
    assert data_loading._EOS_TOKEN == 2
    assert data_loading._MASK_TOKEN == 0

    vocab_size = 5000
    tf.random.set_seed(111)

    ds = tf.data.experimental.from_list(
        [([3, 5, 7, 7, 2], [8, 3, 5, 2])]  # (input, target)
    ).map(
        functools.partial(
            data_loading._convert_to_prefix_lm_example, vocab_size=vocab_size
        )
    )

    examples = list(ds.repeat(3))

    np.testing.assert_array_equal(
        examples[0]["inputs_ids"].numpy(), [0, 5, 7, 7, 2, 8, 3, 5]
    )

    np.testing.assert_array_equal(
        examples[0]["labels"].numpy(), [3, 5, 7, 7, 8, 3, 5, 2]
    )

    np.testing.assert_array_equal(
        examples[0]["bidirectional_attention_mask"].numpy(), [1, 1, 1, 1, 1, 0, 0, 0]
    )

    np.testing.assert_array_equal(
        examples[1]["inputs_ids"].numpy(), [3, 5, 7, 7, 2, 8, 3, 5]
    )

    np.testing.assert_array_equal(
        examples[1]["labels"].numpy(), [3, 5, 7, 7, 8, 3, 5, 2]
    )

    np.testing.assert_array_equal(
        examples[1]["bidirectional_attention_mask"].numpy(), [1, 1, 1, 1, 1, 0, 0, 0]
    )

    np.testing.assert_array_equal(
        examples[2]["inputs_ids"].numpy(), [3, 1996, 7, 7, 2, 8, 3, 5]
    )

    np.testing.assert_array_equal(
        examples[2]["labels"].numpy(), [3, 5, 7, 7, 8, 3, 5, 2]
    )

    np.testing.assert_array_equal(
        examples[2]["bidirectional_attention_mask"].numpy(), [1, 1, 1, 1, 1, 0, 0, 0]
    )
