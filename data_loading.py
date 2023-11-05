import functools
import tensorflow as tf
import tensorflow_text as text

_MASK_TOKEN = 3
_EOS_TOKEN = 2


def _get_bucket_boundaries(lengths, n):
    """
    Divides the dataset set into buckets, each containing an approximately equal number of training samples.
    Returns the bucket boundaries (lengths).
    For each boundary, the bucket will contain examples with lengths less than the boundary.
    :param lengths: List containing the lengths of the sequences in the dataset
    :param n: the number of length bins to use
    :return: A list containing the bucket boundaries.
    """
    lengths.sort()
    bin_size = len(lengths) // n
    bin_lengths = [
        lengths[i] + 1 for i in range(bin_size - 1, len(lengths) - bin_size, bin_size)
    ] + [lengths[-1] + 1]
    return bin_lengths


def _convert_to_prefix_lm_example(input, target, vocab_size):

    # https://www.tensorflow.org/text/guide/bert_preprocessing_guide#masked_language_model_task
    masked_input, _, _ = text.mask_language_model(
        tf.RaggedTensor.from_tensor(tf.expand_dims(input, axis=0)),
        item_selector=text.RandomItemSelector(
            max_selections_per_batch=1000,
            selection_rate=0.15,
            unselectable_ids=[_EOS_TOKEN],
        ),
        mask_values_chooser=text.MaskValuesChooser(vocab_size, _MASK_TOKEN),
    )
    masked_input = tf.squeeze(masked_input.to_tensor(), axis=[0])

    return {
        "token_ids": tf.concat((masked_input, target[:-1]), axis=0),
        "labels": tf.concat((input[:-1], target), axis=0),
        "bidirectional_attention_mask": tf.concat(
            (tf.ones_like(input), tf.zeros_like(target[:-1])), axis=0
        ),
    }


def get_positive_reframing_dataset(file_name, tokenizer):
    def tokenize_input_target_pair(input, target):
        return tokenizer.tokenize(input), tokenizer.tokenize(target)

    return (
        tf.data.experimental.CsvDataset(
            file_name,
            record_defaults=["", ""],
            select_cols=[0, 1],
            header=True,
        )
        .map(tokenize_input_target_pair)
        .map(
            functools.partial(
                _convert_to_prefix_lm_example, vocab_size=tokenizer.vocab_size().numpy()
            )
        )
    )


def get_translation_dataset(inputs_file_name, targets_file_name, tokenizer):
    def tokenize_input_target_pair(input, target):
        return tokenizer.tokenize(input), tokenizer.tokenize(target)

    return (
        tf.data.Dataset.zip(
            tf.data.TextLineDataset(
                inputs_file_name,
            ),
            tf.data.TextLineDataset(
                targets_file_name,
            ),
        )
        .map(tokenize_input_target_pair)
        .map(
            functools.partial(
                _convert_to_prefix_lm_example, vocab_size=tokenizer.vocab_size().numpy()
            )
        )
    )


def bucket(data, batch_size, bucket_boundaries=None, num_length_buckets=5):
    if bucket_boundaries is None:
        print("finding bucket boundaries")
        lengths = []
        for ex in data.as_numpy_iterator():
            lengths.append(ex["token_ids"].shape[-1])
        bucket_boundaries = _get_bucket_boundaries(lengths, num_length_buckets)
    else:
        num_length_buckets = len(bucket_boundaries)

    print(f"bucket boundaries: {bucket_boundaries}")

    data = data.bucket_by_sequence_length(
        element_length_func=lambda elem: tf.shape(elem["token_ids"])[0],
        bucket_boundaries=bucket_boundaries,
        pad_to_bucket_boundary=True,
        drop_remainder=False,
        bucket_batch_sizes=[batch_size] * (num_length_buckets + 1),
    )

    return data, bucket_boundaries
