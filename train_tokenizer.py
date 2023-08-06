import sentencepiece as spm
import click


@click.command()
@click.option("--vocab_size", default=8000)
@click.option("--model_prefix", required=True)
@click.argument("input", nargs=-1)
def main(**kwargs):
    kwargs["input"] = ",".join(kwargs["input"])

    spm.SentencePieceTrainer.train(
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=2,
        user_defined_symbols="<mask>",
        **kwargs,
    )


if __name__ == "__main__":
    main()
