import wandb


def main():
    sweep_configuration = {
        "name": "positive-reframing-sweep-1",
        "program": "train.py",
        "command": ["${env}", "python3", "${program}", "${args}"],
        "metric": {"name": "val/mean_per_token_loss", "goal": "minimize"},
        "method": "grid",
        "parameters": {
            "tokenizer_file": {"values": ["outputs/m.model"]},
            "train_data": {"values": ["data/train.csv"]},
            "val_data": {"values": ["data/dev.csv"]},
            "model_save_dir": {"values": ["model_saves"]},
            "batch_size": {"values": [128]},
            "num_epochs": {"values": [500]},
            "save_every": {"values": [500]},
            "eval_every": {"values": [50]},
            "num_length_buckets": {"values": [5]},
            "learning_rate": {"values": [0.00001]},
            "emb_size": {"values": [128, 256]},
            "mlp_hidden_dim": {"values": [512, 1024]},
            "num_layers": {"values": [2, 4]},
            "num_heads": {"values": [2, 4]},
        },
    }

    wandb.sweep(sweep=sweep_configuration)


if __name__ == "__main__":
    main()
