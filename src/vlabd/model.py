from __future__ import annotations

import torch
from torch import nn


class MiniVLA(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        text_dim: int = 64,
        vision_dim: int = 96,
        fusion_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(),
            nn.LayerNorm(text_dim),
        )
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, vision_dim),
            nn.ReLU(),
            nn.LayerNorm(vision_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + vision_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_dim),
        )
        self.action_head = nn.Linear(fusion_dim, 2)

    def forward(self, image: torch.Tensor, tokens: torch.Tensor, return_features: bool = False):
        text = self.text_embedding(tokens).mean(dim=1)
        text = self.text_encoder(text)
        vision = self.vision_encoder(image)
        fusion = self.fusion(torch.cat([vision, text], dim=-1))
        logits = self.action_head(fusion)

        if return_features:
            return logits, {
                "vision": vision,
                "text": text,
                "fusion": fusion,
                "logits": logits,
            }
        return logits
