# Transformer-Stu

A repository for learning JAX.

## Motivation

There are two ways to achieve happiness:
1. Appreciate the beautiful world around you.
2. Write code.

As usual, I am only going to do 2, but hopefully the resulting code will help myself and others do 1.
I aim to implement a transformer model and training harness using [JAX](https://jax.readthedocs.io/en/latest/) and [Flax](https://flax.readthedocs.io/en/latest/).
I use [SentencePiece](https://github.com/google/sentencepiece) for tokenization and [tf.data](https://www.tensorflow.org/guide/data) for dataset loading and preprocessing. 
If all goes smoothly I will be able to train the model on a [positive reframing](https://github.com/SALT-NLP/positive-frames) dataset.
The resulting model should be able to translate from negative to positive language.

### Side-quest

Some people just want to watch the world burn.
We could train a model which takes positive language and turns it negative.
Maybe one day we will have a browser plugin that exposes the dark side of those shiny happy LinkedIn posts. 

## Setup
### Install Requirements
On Apple Silicon you must use Python 3.9 and run
```bash
pip install -r requirements_apple_silicon.txt
```
On any other platform, run
```bash
pip install -r requirements.txt
```

### Data

Download and extract the [positive reframing data](https://www.dropbox.com/sh/pnoczmv0uyn51e6/AAAGek6yX12Yc4PA2RwtZeZKa?dl=0) to the following directory structure:
```
data
├── dev.csv
├── test.csv
└── train.csv
```

## Run

### Train BPE Tokenizer

```bash
mkdir outputs
python flatten_data.py --input_file data/train.csv --output_file outputs/flattened_train.txt
python train_tokenizer.py
```

### Train Prefix-LM

```bash
python train.py
```
