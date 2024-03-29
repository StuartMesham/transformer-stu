import functools

import tensorflow as tf
import tensorflow_text as text
from tensorflow import Tensor
from tensorflow.python.ops.gen_dataset_ops import MapDataset
from tensorflow_text import SentencepieceTokenizer

_MASK_TOKEN = 3
_EOS_TOKEN = 2


def _get_bucket_boundaries(lengths: list[int], n: int) -> list[int]:
    """Divides the dataset set into buckets, each containing an approximately equal number of training samples.

    Returns the bucket boundaries (lengths).
    For each boundary, the bucket will contain examples with lengths less than the boundary.

    Args:
    lengths: List containing the lengths of the sequences in the dataset
    n: the number of length bins to use
    A list containing the bucket boundaries.
    """
    lengths.sort()
    bin_size = len(lengths) // n
    bin_lengths = [
        lengths[i] + 1 for i in range(bin_size - 1, len(lengths) - bin_size, bin_size)
    ] + [lengths[-1] + 1]
    return bin_lengths


def _convert_to_prefix_lm_example(
    input: Tensor, target: Tensor, vocab_size: int
) -> dict[str, Tensor]:
    """Converts `input` and `target` into a prefix LM example with BERT-style masking applied to `input`.

    Operates on a single example (rather than a batch of examples).

    The procedure for BERT-style masking is:
     1. Choose a random 15% of the input tokens.
     2. For each of the chosen tokens, independently choose one of the actions below with the specified probabilities:
        - 80% replace the token with a [MASK] token
        - 10% replace the token with a random token from the vocabulary
        - 10% leave the token unchanged
    See the docs for `tensorflow_text.MaskValuesChooser` for more information. See also:
    https://www.tensorflow.org/text/guide/bert_preprocessing_guide#masked_language_model_task

    The (masked) input and target sequences are concatenated and a `labels` array is generated containing the token ids
        to be predicted at each position during training. The portion of `labels` corresponding to the `input` sequence
        contains the un-masked `input` sequence (excluding the end-of-sequence token). The remaining portion of the
        `labels` array contains the left-shifted autoregressive targets. The model is trained to output the first token
        of `targets` when it sees the end-of-sequence token of the `input` sequence.

    A `bidirectional_attention_mask` is created containing ones where bidirectional attention should be used and zeros
        where causal attention should be used.

    Args:
        input: An input sequence.
        target: A target sequence.
        vocab_size: The size of the vocabulary used for tokenisation.

    Returns:
        A dictionary containing `token_ids`: the combined (masked) input and target sequences, `labels` and
        `bidirectional_attention_mask`.
    """
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


def get_positive_reframing_dataset(
    file: str, tokenizer: SentencepieceTokenizer
) -> MapDataset:
    """Creates and returns a tensorflow dataset for the Positive Reframing task.

    Args:
        file: Path to a CSV file where the first column is the input text and the second column is the output text.
            The CSV file should have a header row.
        tokenizer: The SentencepieceTokenizer to use for tokenization.

    Returns:
        The resulting tensorflow dataset. Each example is a dictionary with the keys: token_ids, labels and
            bidirectional_attention_mask.
    """

    def tokenize_input_target_pair(
        input: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor]:
        return tokenizer.tokenize(input), tokenizer.tokenize(target)

    return (
        tf.data.experimental.CsvDataset(
            file,
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


def get_translation_dataset(
    inputs_file: str, targets_file: str, tokenizer: SentencepieceTokenizer
) -> MapDataset:
    """Creates and returns a tensorflow dataset for the machine translation task.

    Note that `inputs_file` and `targets_file` should be "parallel" files. The sentence on line n of `targets_file`
        should be the translated version of line n of `inputs_file`.

    Args:
        inputs_file: Path to file with one input sentence per line.
        targets_file: Path to file with one target sentence per line.
        tokenizer: The SentencepieceTokenizer to use for tokenization.

    Returns:
        The resulting tensorflow dataset. Each example is a dictionary with the keys: token_ids, labels and
            bidirectional_attention_mask.
    """

    def tokenize_input_target_pair(
        input: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor]:
        return tokenizer.tokenize(input), tokenizer.tokenize(target)

    return (
        tf.data.Dataset.zip(
            tf.data.TextLineDataset(
                inputs_file,
            ),
            tf.data.TextLineDataset(
                targets_file,
            ),
        )
        .map(tokenize_input_target_pair)
        .map(
            functools.partial(
                _convert_to_prefix_lm_example, vocab_size=tokenizer.vocab_size().numpy()
            )
        )
    )


def bucket(
    data: MapDataset,
    batch_size: int,
    bucket_boundaries: list[int] | None = None,
    num_length_buckets: int = 5,
) -> MapDataset:
    """Creates batches of sequences bucketed according to sequence length.

    For optimal compute utilisation, we want to minimise the number of padding tokens passed through our model during
        training.

    Args:
        data: Dataset of sequences to be batched.
        batch_size: The desired number of sequences per batch.
        bucket_boundaries: Sequence length bucket boundaries to use. If None, `num_length_buckets` buckets will be
            created each containing an equal number of sequences. This requires a full iteration over `data`.
        num_length_buckets: The number of buckets to create if `bucket_boundaries` is None. If `bucket_boundaries` is
            not None, the value of `num_length_buckets` will be ignored.
    """
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
