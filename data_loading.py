import tensorflow as tf


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


def get_positive_reframing_dataset(file_name, tokenizer, batch_size, bucket_boundaries=None, num_length_buckets=5):
    def tokenize_input_target_pair(input, target):
        return tokenizer.tokenize(input), tokenizer.tokenize(target)

    def convert_to_prefix_lm_example(input, target):
        return {
            "inputs_ids": tf.concat((input, target[:-1]), axis=0),
            "labels": tf.concat((tf.zeros_like(input)[:-1], target), axis=0),
            "bidirectional_attention_mask": tf.concat(
                (tf.ones_like(input), tf.zeros_like(target[:-1])), axis=0
            ),
        }

    data = (
        tf.data.experimental.CsvDataset(
            file_name,
            record_defaults=["", ""],
            select_cols=[0, 1],
            header=True,
        )
        .map(tokenize_input_target_pair)
        .map(convert_to_prefix_lm_example)
    )

    if bucket_boundaries is None:
        print("finding bucket boundaries")
        lengths = []
        for ex in data.as_numpy_iterator():
            lengths.append(ex["inputs_ids"].shape[-1])
        bucket_boundaries = _get_bucket_boundaries(lengths, num_length_buckets)
    else:
        num_length_buckets = len(bucket_boundaries)

    print(f"bucket boundaries: {bucket_boundaries}")

    data = data.bucket_by_sequence_length(
        element_length_func=lambda elem: tf.shape(elem["inputs_ids"])[0],
        bucket_boundaries=bucket_boundaries,
        pad_to_bucket_boundary=True,
        drop_remainder=False,
        bucket_batch_sizes=[batch_size] * (num_length_buckets + 1),
    )

    return data, bucket_boundaries
