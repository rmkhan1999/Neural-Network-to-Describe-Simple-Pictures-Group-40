from __future__ import annotations
import torch.nn as nn


class EmbeddingCNN(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 256),  # for 64x64 input
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)
