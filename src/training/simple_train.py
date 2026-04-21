# Training script for image-to-caption classification model.
# This script:
# 1. Loads the dataset (images + captions)
# 2. Trains a CNN using cross-entropy loss
# 3. Updates model weights using Adam optimizer
# 4. Saves the trained model and label mapping for inference
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import json
import sys


from src.data.simple_pipeline_dataset import ImageCaptionDataset
from src.models.simple_cnn import SimpleCNN

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "mock" / "metadata" / "data.jsonl"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
def train():
    # loading data
    dataset = ImageCaptionDataset(str(DATA_PATH))
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print("Dataset size:", len(dataset))
    print("Num classes:", len(dataset.label_map))
    model = SimpleCNN(num_classes=len(dataset.label_map))
   # Define optimizer and loss function (CrossEntropy for classification)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(50):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in loader:
            preds = model(imgs)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(preds, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}")

    # Save model
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ARTIFACTS_DIR / "model.pth")
    with open(ARTIFACTS_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(dataset.label_map, f, ensure_ascii=False, indent=2)
    # Save label map
    with open("artifacts/label_map.json", "w") as f:
        json.dump(dataset.label_map, f)

    print("Model + label map saved!")
    print("\n--- Sample Predictions ---")

    model.eval()

    with torch.no_grad():
        for i in range(len(dataset)):
            img, label = dataset[i]
            img = img.unsqueeze(0)  # add batch dimension

            pred = model(img)
            _, predicted = torch.max(pred, 1)
            pred_caption = dataset.id_to_caption[predicted.item()]
            true_caption = dataset.id_to_caption[label.item()]
            print(f"\nSample {i}")
            print("True:", true_caption)
            print("Pred:", pred_caption)

if __name__ == "__main__":
    train()



