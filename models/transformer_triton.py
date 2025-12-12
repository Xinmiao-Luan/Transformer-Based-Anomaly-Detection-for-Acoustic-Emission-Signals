import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .transformer_baseline import PositionalEncoding
from kernels.softmax_triton import softmax_triton_attention


class TritonSelfAttention(nn.Module):
    """
    Single-head self-attention using Triton softmax.

    Input:  x [B, L, D]
    Output: y [B, L, D]
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q = self.q_proj(x)  # [B, L, D]
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_k**0.5)  # [B, L, L]
        attn = softmax_triton_attention(scores)
        attn = self.dropout(attn)

        y = torch.matmul(attn, v)  # [B, L, D]
        y = self.out_proj(y)
        return y


class TritonTransformerBlock(nn.Module):
    """
    Transformer encoder block using Triton-based attention.
    """

    def __init__(self, d_model: int, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.self_attn = TritonSelfAttention(d_model, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + residual
        attn_out = self.self_attn(x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # FFN + residual
        ffn = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(ffn)
        x = self.norm2(x)
        return x


class AETransformerTriton(pl.LightningModule):
    """
    Transformer encoder with Triton-accelerated self-attention.
    """

    def __init__(
        self,
        input_dim: int = 16,
        d_model: int = 64,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        num_classes: int = 2,
        lr: float = 1e-3,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        self.layers = nn.ModuleList(
            [
                TritonTransformerBlock(
                    d_model=d_model,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        pooled = x.mean(dim=1)
        logits = self.cls_head(pooled)
        return logits

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
