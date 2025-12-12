import argparse

import torch
import pytorch_lightning as pl

from data.dataset import create_ae_dataloaders
from models.transformer_baseline import AETransformerBaseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Directory containing .mat files and index.csv")
    parser.add_argument("--index_file", type=str, default="index.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_loader, val_loader = create_ae_dataloaders(
        root_dir=args.root_dir,
        index_file=args.index_file,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    # Infer input_dim from dataset
    example_batch = next(iter(train_loader))
    x_example, _ = example_batch
    input_dim = x_example.shape[-1]

    model = AETransformerBaseline(
        input_dim=input_dim,
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
