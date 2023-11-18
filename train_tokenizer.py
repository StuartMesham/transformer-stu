import click
import sentencepiece as spm


@click.command()
@click.option("--vocab_size", default=8000)
@click.option("--model_prefix", required=True)
@click.argument("input", nargs=-1)
def main(**kwargs: bool | int | str | list[str]) -> None:
    """Trains a byte pair encoding tokenizer on the provided input files.

    Note that the `input` argument should be a comma separated list of input files.
    """
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
