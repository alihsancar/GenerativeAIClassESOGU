"""
ViT-L/16 + Bidirectional GRU hibrit modeli.
Notebook Cell 19 ile birebir aynıdır — değiştirilmedi.
"""

import torch
import torch.nn as nn
from transformers import ViTModel

from config.settings import (
    NUM_CLASSES, GRU_HIDDEN, GRU_LAYERS, GRU_DROPOUT, MODEL_NAMES
)


class ViT_GRU(nn.Module):
    """
    ViT-L/16 @ 384 backbone → patch tokens → Bidirectional GRU → classifier

    Patch tokens: (batch, 576+1, 1024)  [CLS + 24×24 patches]
    GRU input:    patch tokens (CLS hariç, 576 token, 1024-dim)
    GRU son hidden state → linear → num_classes

    Notebook Cell 19 referansı: Aly vd. (2024) — ViT+GRU+XAI
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        gru_hidden: int  = GRU_HIDDEN,
        gru_layers: int  = GRU_LAYERS,
        dropout: float   = GRU_DROPOUT,
        backbone_name: str = MODEL_NAMES["lgru"],
    ):
        super().__init__()
        self.vit = ViTModel.from_pretrained(backbone_name, add_pooling_layer=False)
        self.hidden_size = self.vit.config.hidden_size  # 1024 (L mimarisi)

        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.norm       = nn.LayerNorm(gru_hidden * 2)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_hidden * 2, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (B, 3, 384, 384)
        döner:        (B, num_classes)  ham logit
        """
        out          = self.vit(pixel_values=pixel_values)
        patch_tokens = out.last_hidden_state[:, 1:, :]   # CLS hariç → (B, 576, 1024)

        gru_out, _  = self.gru(patch_tokens)             # (B, 576, gru_hidden*2)
        last_step   = gru_out[:, -1, :]                  # son adım
        mean_pool   = gru_out.mean(dim=1)                # ortalama pooling
        feat        = (last_step + mean_pool) / 2        # ikisinin ortalaması

        feat = self.norm(feat)
        feat = self.dropout(feat)
        return self.classifier(feat)
