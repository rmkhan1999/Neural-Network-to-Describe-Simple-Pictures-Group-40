"""
CaptionClassificationDataset - Approach 1

This dataset converts image-caption pairs into a classification problem:
- Input: Image
- Output: Integer label corresponding to a caption

Metadata file format (JSONL):
{"image_path": "path/to/image.jpg", "caption": "a cat"}
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CaptionClassificationDataset(Dataset):
    def __init__(
        self,
        metadata_file: str | Path,
        label_map: Optional[Dict[str, int]] = None,
        image_size: int = 64,
    ) -> None:
        self.metadata_file = Path(metadata_file).resolve()
        self.project_root = Path(__file__).resolve().parents[2]

        with open(self.metadata_file, "r", encoding="utf-8") as f:
            self.rows: List[Dict] = [json.loads(line) for line in f if line.strip()]

        captions = sorted({row["caption"] for row in self.rows})
        if label_map is None:
            self.label_map = {caption: idx for idx, caption in enumerate(captions)}
        else:
            self.label_map = label_map

        self.id_to_caption = {idx: caption for caption, idx in self.label_map.items()}

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_path = (self.project_root / row["image_path"]).resolve()

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        caption = row["caption"]
        label = self.label_map[caption]

        return image, torch.tensor(label, dtype=torch.long)

    @staticmethod
    def save_label_map(path: str | Path, label_map: Dict[str, int]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_label_map(path: str | Path) -> Dict[str, int]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
