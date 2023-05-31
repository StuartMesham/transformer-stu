import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="outputs/flattened_train.txt",
    model_prefix="outputs/m",
    vocab_size=8000,
    model_type="bpe",
    pad_id=0,
    unk_id=1,
    bos_id=-1,
    eos_id=2,
)
