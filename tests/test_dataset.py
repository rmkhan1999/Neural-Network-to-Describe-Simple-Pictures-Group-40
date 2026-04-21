from src.data.simple_pipeline_dataset import ImageCaptionDataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "mock" / "metadata" / "data.jsonl"

dataset = ImageCaptionDataset(str(DATA_PATH))

print("Dataset size:", len(dataset))

img, label = dataset[0]
print("Image shape:", img.shape)
print("Label:", label)
