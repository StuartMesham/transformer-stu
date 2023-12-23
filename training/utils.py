from sacrebleu import BLEU
from tqdm.auto import tqdm


def calc_bleu(state, padded_val_dataset, tokenizer, autoregressive_inference_step_fn):
    predicted_texts = []
    expected_texts = []
    total_completed_sequences = 0
    for batch in tqdm(
        padded_val_dataset.as_numpy_iterator(), desc="running auto-regressive inference"
    ):
        # calculate autoregressive validation metrics
        (
            predicted_tokens,
            expected_tokens,
            num_completed_sequences,
        ) = autoregressive_inference_step_fn(
            state,
            batch,
            sequence_length=batch["token_ids"].shape[-1],
            batch_size=batch["token_ids"].shape[0],
        )

        total_completed_sequences += num_completed_sequences.item()
        predicted_texts.extend(
            [s.decode() for s in tokenizer.detokenize(predicted_tokens).numpy()]
        )
        expected_texts.extend(
            [[s.decode()] for s in tokenizer.detokenize(expected_tokens).numpy()]
        )

    return BLEU().corpus_score(
        predicted_texts, expected_texts
    ).score, total_completed_sequences
