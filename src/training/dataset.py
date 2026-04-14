# In this script caption data is loaded from jsonl file where image is converted into tensors and caption is converted to numeric value
# to make it neural network readable. Thus  preparing data for model
import json
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
        # Loading JSONL file
        self.data = [json.loads(line) for line in open(json_file)]

        # Create label map (caption → index)
        captions = list(set(d["caption"] for d in self.data))
        self.label_map = {c: i for i, c in enumerate(captions)}
        self.inverse_map = {i: c for c, i in self.label_map.items()}

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
        img = Image.open(item["image_path"]).convert("RGB")
        img = self.transform(img)

        # Convert caption to label
        label = self.label_map[item["caption"]]

        return img, label
