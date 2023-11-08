import wandb


def main():
    """Creates a Weights & Biases sweep."""
    sweep_configuration = {
        "name": "machine-translation-sweep-1",
        "program": "train.py",
        "command": ["${env}", "python3", "${program}", "${args}"],
        "metric": {"name": "val/mean_per_token_loss", "goal": "minimize"},
        "method": "grid",
        "parameters": {
            "tokenizer_file": {"values": ["outputs/en-de.model"]},
            "train_inputs": {"values": ["data/de-en/train.en"]},
            "train_targets": {"values": ["data/de-en/train.de"]},
            "val_inputs": {"values": ["data/de-en/valid.en"]},
            "val_targets": {"values": ["data/de-en/valid.de"]},
            "model_save_dir": {"values": ["model_saves"]},
            "batch_size": {"values": [64]},
            "num_epochs": {"values": [200]},
            "save_every": {"values": [10]},
            "eval_every": {"values": [10]},
            "num_length_buckets": {"values": [10]},
            "learning_rate": {"values": [0.0001, 0.00001, 0.000001]},
            "emb_size": {"values": [512]},
            "mlp_hidden_dim": {"values": [1024]},
            "num_layers": {"values": [15]},
            "num_heads": {"values": [4, 8, 16]},
            "label_smoothing_mass": {"values": [0.1]},
            "warmup_steps": {"values": [4000]},
            "dropout_rate": {"values": [0.1]},
        },
    }

    wandb.sweep(sweep=sweep_configuration)


if __name__ == "__main__":
    main()
