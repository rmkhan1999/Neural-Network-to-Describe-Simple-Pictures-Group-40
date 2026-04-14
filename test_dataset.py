# script to check dataset.py file in src/training
from src.training.dataset import ImageCaptionDataset

dataset = ImageCaptionDataset("data/mock/data.jsonl")

print("Dataset size:", len(dataset))

img, label = dataset[0]
print("Image shape:", img.shape)
print("Label:", label)
