# Transformer-Stu

A repository for learning JAX.

## Description

This repository contains an implementation of a transformer model and training harness using [JAX](https://jax.readthedocs.io/en/latest/) and [Flax](https://flax.readthedocs.io/en/latest/).
I use [SentencePiece](https://github.com/google/sentencepiece) for tokenization and [tf.data](https://www.tensorflow.org/guide/data) for dataset loading and preprocessing.
I train an English-to-German machine-translation model on a parallel corpus.

### Model
I am implementing Gao et al.'s paper [Is Encoder-Decoder Redundant for Neural Machine Translation?](https://aclanthology.org/2022.aacl-main.43.pdf).
It is a decoder-only prefix-LM trained with a BERT-style masked-language-modelling objective on the source language (prefix) and an autoregressive LM objective on the target language. The losses for these two objectives are applied simultaneously on each backwards pass of the model.


## Setup
### Install
On Apple Silicon you must use Python 3.11 and run:
```bash
pip install -r requirements_apple_silicon.txt
```
On any other platform, run:
```bash
pip install -r requirements.txt
```

Developers can install the pre-commit hooks with:
```bash
pre-commit install
```

### Download Data

Download and extract the IWSLT 2014 de-en dataset:
```
wget https://huggingface.co/datasets/bbaaaa/iwslt14-de-en/resolve/main/data/de-en.zip
unzip de-en.zip
```

## Run

### Train BPE Tokenizer

```bash
mkdir outputs
python3 train_tokenizer.py --vocab_size 10149 --model_prefix "outputs/en-de" data/de-en/train.*
```

### Train an English to German model

```bash
train.py --tokenizer_file outputs/en-de.model --train_inputs=de-en/train.en --train_targets=de-en/train.de --val_inputs=de-en/valid.en --val_targets=de-en/valid.de --emb_size=512 --mlp_hidden_dim=1024 --num_layers=15 --label_smoothing_mass=0.1 --batch_size=64 --dropout_rate=0.1 --eval_every=10 --label_smoothing_mass=0.1 --learning_rate=0.0001 --num_epochs=200 --num_heads=4 --save_every=10 --warmup_steps=4000 --num_length_buckets 10 --train_bucket_boundaries=19,24,29,35,41,49,58,72,94,523 --validation_bucket_boundaries=19,24,29,35,41,48,57,71,93,324
```

### Perform Hyper-Parameter sweep with WandB

Set WandB project name:
```bash
export WANDB_PROJECT=transformer-stu
```

Create a sweep:
```bash
python create_wandb_sweep.py
```

Launch an agent to perform training runs for the sweep:
```bash
wandb agent <username>/<project>/<sweep_id>
```
