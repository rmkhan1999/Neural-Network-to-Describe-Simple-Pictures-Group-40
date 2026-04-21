from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CaptionEmbeddingDataset(Dataset):
    def __init__(
        self,
        metadata_file: str | Path,
        caption_to_embedding: dict[str, torch.Tensor],
        split: Optional[str] = None,
        image_size: int = 64,
    ) -> None:
        self.metadata_file = Path(metadata_file).resolve()
        self.dataset_root = self.metadata_file.parent.parent

        with open(self.metadata_file, "r", encoding="utf-8") as f:
            all_rows = [json.loads(line) for line in f if line.strip()]

        if split is not None:
            self.rows = [row for row in all_rows if row.get("split") == split]
        else:
            self.rows = all_rows

        self.caption_to_embedding = caption_to_embedding

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_path = (self.dataset_root / row["image_path"]).resolve()

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        caption = row["caption"]
        embedding = self.caption_to_embedding[caption]

        return image, embedding, caption, row["id"]
