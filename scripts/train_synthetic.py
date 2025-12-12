import argparse

import torch
import pytorch_lightning as pl

from data.dataset import create_synthetic_dataloaders
from models.transformer_baseline import AETransformerBaseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--input_dim", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_loader, val_loader = create_synthetic_dataloaders(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        input_dim=args.input_dim,
    )

    model = AETransformerBaseline(
        input_dim=args.input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        num_classes=2,
        lr=args.lr,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=args.max_epochs,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
