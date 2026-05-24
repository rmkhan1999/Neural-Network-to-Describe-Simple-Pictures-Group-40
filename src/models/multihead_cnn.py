"""
Multi-head CNN:
Image -> symbolic attributes/state -> reconstructed caption.
Used to test structured prediction.
"""
from __future__ import annotations
import torch.nn as nn


class StructuredCNN(nn.Module):
    def __init__(self, head_dims: dict[str, int]):
        super().__init__()

        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # IMPORTANT:
            # Keep spatial layout using 4x4 pooling instead of 1x1 global pooling.
            # This helps relation prediction for shapes: above/below/left/right.
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),

            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

        self.heads = nn.ModuleDict({
            name: nn.Linear(512, num_classes)
            for name, num_classes in head_dims.items()
        })

    def forward(self, x):
        features = self.backbone(x)
        return {
            name: head(features)
            for name, head in self.heads.items()
        }
