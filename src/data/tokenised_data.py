from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.idx = 4

    def add_sentence(self, sentence):
        for word in sentence.split():
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def encode(self, sentence):
        tokens = [self.word2idx["<start>"]]
        for word in sentence.split():
            tokens.append(self.word2idx.get(word, self.word2idx["<unk>"]))
        tokens.append(self.word2idx["<end>"])
        return tokens


class SeqDataset(Dataset):
    def __init__(self, metadata_file, split=None, image_size=64, vocab=None):
        self.metadata_file = Path(metadata_file).resolve()
        self.root = self.metadata_file.parent.parent

        with open(self.metadata_file, "r", encoding="utf-8") as f:
            data = [json.loads(l) for l in f if l.strip()]

        if split is not None:
            data = [d for d in data if d["split"] == split]

        self.data = data

        if vocab is None:
            self.vocab = Vocabulary()
            for d in self.data:
                self.vocab.add_sentence(d["caption"])
        else:
            self.vocab = vocab

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_path = (self.root / item["image_path"]).resolve()
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        caption_tokens = self.vocab.encode(item["caption"])

        return img, torch.tensor(caption_tokens, dtype=torch.long)
