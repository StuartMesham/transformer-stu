import wandb


def main():
    sweep_configuration = {
        "name": "positive-reframing-sweep-1",
        "program": "train.py",
        "metric": {"name": "val/mean_per_token_loss", "goal": "minimize"},
        "method": "grid",
        "parameters": {
            "tokenizer_file": {"values": ["outputs/m.model"]},
            "train_data": {"values": ["data/train.csv"]},
            "val_data": {"values": ["data/dev.csv"]},
            "model_save_dir": {"values": ["model_saves"]},
            "batch_size": {"values": [32]},
            "num_epochs": {"values": [5]},
            "save_every": {"values": [1]},
            "eval_every": {"values": [1]},
            "num_length_buckets": {"values": [5]},
            "learning_rate": {"values": [0.01, 0.001, 0.0001]},
            "emb_size": {"values": [64]},
            "mlp_hidden_dim": {"values": [128]},
            "num_layers": {"values": [2]},
            "num_heads": {"values": [4]},
        },
    }

    wandb.sweep(sweep=sweep_configuration)


if __name__ == "__main__":
    main()
