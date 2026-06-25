from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as transforms

from src.models.cnn_lstm import CNN_LSTM
from src.data.tokenised_data import Vocabulary

_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


def build_vocab(task: str, project_root: Path) -> Vocabulary:
    metadata_dir = project_root / "data" / "processed" / task / "metadata"
    single = metadata_dir / f"{task}_metadata.jsonl"

    if single.exists():
        files = [single]
    else:
        files = [metadata_dir / "train.jsonl", metadata_dir / "val.jsonl", metadata_dir / "test.jsonl"]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Metadata not found: {missing}")

    vocab = Vocabulary()
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    vocab.add_sentence(json.loads(line)["caption"])
    return vocab


def load_model(task: str, project_root: Path, vocab_size: int, device: torch.device) -> CNN_LSTM:
    model = CNN_LSTM(vocab_size=vocab_size)
    weights = project_root / "artifacts" / task / "cnn_lstm" / "best_model.pth"
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()
    return model


def predict(
    model: CNN_LSTM,
    vocab: Vocabulary,
    image: Image.Image,
    device: torch.device,
    max_len: int = 30,
) -> str:
    """Greedy autoregressive decode."""
    tensor = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encoder(tensor)

    tokens = [vocab.word2idx["<start>"]]
    end_idx = vocab.word2idx.get("<end>", -1)

    with torch.no_grad():
        for _ in range(max_len):
            inp = torch.tensor([tokens], dtype=torch.long).to(device)
            # decoder returns (batch, 1+len(tokens), vocab_size)
            outputs = model.decoder(features, inp)
            next_token = int(outputs[0, -1, :].argmax().item())
            if next_token == end_idx:
                break
            tokens.append(next_token)

    pad_idx = vocab.word2idx.get("<pad>", 0)
    start_idx = vocab.word2idx["<start>"]
    words = [
        vocab.idx2word[t]
        for t in tokens
        if t not in (pad_idx, start_idx, end_idx)
    ]
    return " ".join(words)
