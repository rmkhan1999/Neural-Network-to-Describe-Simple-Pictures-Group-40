from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.tokenised_data import SeqDataset, Vocabulary
from src.models.cnn_lstm import CNN_LSTM


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    captions = [b[1] for b in batch]

    max_len = max(len(c) for c in captions)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)

    for i, c in enumerate(captions):
        padded[i, :len(c)] = c

    return images, padded


def build_vocab_from_files(files):
    vocab = Vocabulary()
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    vocab.add_sentence(row["caption"])
    return vocab


def decode_tokens(tokens, vocab):
    words = []
    for token in tokens:
        word = vocab.idx2word.get(int(token), "<unk>")
        if word == "<end>":
            break
        if word in {"<pad>", "<start>"}:
            continue
        words.append(word)
    return " ".join(words)


def evaluate(model, loader, vocab_size, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        for images, padded in loader:
            images = images.to(device)
            padded = padded.to(device)

            outputs = model(images, padded[:, :-1])
            targets = padded[:, 1:]

            outputs = outputs[:, 1:, :]

            loss = loss_fn(outputs.reshape(-1, vocab_size), targets.reshape(-1))

            total_loss += loss.item()
            total_batches += 1

            preds = outputs.argmax(dim=2)

            mask = targets != 0  # ignore padding

            correct_tokens += ((preds == targets) & mask).sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    approx_prob = math.exp(-avg_loss) if total_batches > 0 else 0.0
    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    return {
        "avg_loss": avg_loss,
        "approx_token_prob": approx_prob,
        "token_accuracy": token_acc,
    }


def save_test_predictions(model, loader, vocab, device, output_file: Path):
    model.eval()
    predictions = []

    with torch.no_grad():
        sample_counter = 1

        for images, padded in loader:
            images = images.to(device)
            padded = padded.to(device)

            outputs = model(images, padded[:, :-1])
            outputs = outputs[:, 1:, :]
            preds = outputs.argmax(dim=2)

            for i in range(preds.size(0)):
                pred_caption = decode_tokens(preds[i].tolist(), vocab)

                predictions.append({
                    "id": f"sample_{sample_counter:06d}",
                    "pred_caption": pred_caption,
                })
                sample_counter += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="shapes", choices=["shapes", "numbers", "tictactoe"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    metadata_dir = project_root / "data" / "processed" / args.dataset / "metadata"

    single_file = metadata_dir / f"{args.dataset}_metadata.jsonl"
    train_file = metadata_dir / "train.jsonl"
    val_file = metadata_dir / "val.jsonl"
    test_file = metadata_dir / "test.jsonl"

    if single_file.exists():
        full_dataset = SeqDataset(single_file, split=None)
        vocab = full_dataset.vocab

        train_dataset = SeqDataset(single_file, split="train", vocab=vocab)
        val_dataset = SeqDataset(single_file, split="val", vocab=vocab)
        test_dataset = SeqDataset(single_file, split="test", vocab=vocab)

    elif train_file.exists():
        vocab = build_vocab_from_files([train_file, val_file, test_file])

        train_dataset = SeqDataset(train_file, split=None, vocab=vocab)
        val_dataset = SeqDataset(val_file, split=None, vocab=vocab)
        test_dataset = SeqDataset(test_file, split=None, vocab=vocab)

    else:
        raise FileNotFoundError("Dataset not found")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    vocab_size = len(vocab.word2idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM(vocab_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    artifacts_dir = project_root / "artifacts" / args.dataset / "cnn_lstm"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for images, padded in train_loader:
            images = images.to(device)
            padded = padded.to(device)

            outputs = model(images, padded[:, :-1])
            targets = padded[:, 1:]

            outputs = outputs[:, 1:, :]

            loss = loss_fn(outputs.reshape(-1, vocab_size), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, vocab_size, loss_fn, device)

        print(
            f"Epoch {epoch+1}, "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Token Acc: {val_metrics['token_accuracy']:.2f}"
        )

        if val_metrics["token_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["token_accuracy"]
            torch.save(model.state_dict(), artifacts_dir / "best_model.pth")

    # Final test
    model.load_state_dict(torch.load(artifacts_dir / "best_model.pth", map_location=device))
    test_metrics = evaluate(model, test_loader, vocab_size, loss_fn, device)

    save_test_predictions(model, test_loader, vocab, device, artifacts_dir / "test_predictions.json")

    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\nFinal Test Token Accuracy:", test_metrics["token_accuracy"])


if __name__ == "__main__":
    train()
