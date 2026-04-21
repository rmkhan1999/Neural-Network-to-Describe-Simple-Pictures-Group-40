# In this script caption data is loaded from jsonl file where image is converted into tensors and caption is converted to numeric value
# to make it neural network readable. Thus  preparing data for model
import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

 # Created a  class  for loading image-caption data.
        # Parameters:
        #     json_file (str): Path to JSONL file where each line contains:
        #         - image_path: path to image file
        #         - caption: corresponding text description eg. a large blue circle is above a small red square
class ImageCaptionDataset(Dataset):
    def __init__(self, json_file):

        self.json_file = Path(json_file).resolve()
        self.project_root = Path(__file__).resolve().parents[2]
        # Loading JSONL file
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]

        # Create label map (caption → index)
        captions = sorted({d["caption"] for d in self.data})
        self.label_map = {c: i for i, c in enumerate(captions)}
        self.inverse_map = {i: c for c, i in self.label_map.items()}

        self.id_to_caption = {i: caption for caption, i in self.label_map.items()}
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

# defined a function to calculate length of the dataset(total samples in data)
    def __len__(self):
        return len(self.data)

# defined a  function to read a single entry in the dataset
    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        img_path = (self.project_root / item["image_path"]).resolve()
        img = Image.open(img_path).convert("RGB")

        # Convert caption to label
        label = self.label_map[item["caption"]]
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


