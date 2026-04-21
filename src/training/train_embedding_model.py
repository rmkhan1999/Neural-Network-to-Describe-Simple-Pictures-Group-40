from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from src.data.embedding_dataset import CaptionEmbeddingDataset
from src.models.embedding_cnn import EmbeddingCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def collect_captions_from_files(files: list[Path]) -> list[str]:
    captions = set()
    for file in files:
        for row in load_rows(file):
            captions.add(row["caption"])
    return sorted(captions)


def build_metadata_paths(project_root: Path, dataset_name: str):
    metadata_dir = project_root / "data" / "processed" / dataset_name / "metadata"

    single_file = metadata_dir / f"{dataset_name}_metadata.jsonl"
    train_file = metadata_dir / "train.jsonl"
    val_file = metadata_dir / "val.jsonl"
    test_file = metadata_dir / "test.jsonl"

    if single_file.exists():
        return {
            "format": "single",
            "single_file": single_file,
            "metadata_dir": metadata_dir,
        }

    if train_file.exists() and val_file.exists() and test_file.exists():
        return {
            "format": "split",
            "train_file": train_file,
            "val_file": val_file,
            "test_file": test_file,
            "metadata_dir": metadata_dir,
        }

    raise FileNotFoundError(
        f"No valid metadata found in {metadata_dir}. "
        f"Expected either {single_file.name} or train/val/test.jsonl files."
    )


def build_caption_embeddings(captions: list[str], model_name: str):
    text_model = SentenceTransformer(model_name)
    vectors = text_model.encode(
        captions,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return {caption: vectors[i].detach().cpu() for i, caption in enumerate(captions)}, vectors.detach().cpu(), captions


def make_datasets(paths_info, caption_to_embedding, image_size: int):
    if paths_info["format"] == "single":
        metadata_file = paths_info["single_file"]

        train_dataset = CaptionEmbeddingDataset(
            metadata_file=metadata_file,
            caption_to_embedding=caption_to_embedding,
            split="train",
            image_size=image_size,
        )
        val_dataset = CaptionEmbeddingDataset(
            metadata_file=metadata_file,
            caption_to_embedding=caption_to_embedding,
            split="val",
            image_size=image_size,
        )
        test_dataset = CaptionEmbeddingDataset(
            metadata_file=metadata_file,
            caption_to_embedding=caption_to_embedding,
            split="test",
            image_size=image_size,
        )
        metadata_source = str(metadata_file)

    else:
        train_dataset = CaptionEmbeddingDataset(
            metadata_file=paths_info["train_file"],
            caption_to_embedding=caption_to_embedding,
            split=None,
            image_size=image_size,
        )
        val_dataset = CaptionEmbeddingDataset(
            metadata_file=paths_info["val_file"],
            caption_to_embedding=caption_to_embedding,
            split=None,
            image_size=image_size,
        )
        test_dataset = CaptionEmbeddingDataset(
            metadata_file=paths_info["test_file"],
            caption_to_embedding=caption_to_embedding,
            split=None,
            image_size=image_size,
        )
        metadata_source = ", ".join(
            str(p) for p in [paths_info["train_file"], paths_info["val_file"], paths_info["test_file"]]
        )

    return train_dataset, val_dataset, test_dataset, metadata_source


def retrieve_caption(pred_embeddings: torch.Tensor, caption_embedding_matrix: torch.Tensor, caption_list: list[str]) -> list[str]:
    pred_embeddings = F.normalize(pred_embeddings, dim=1)
    sims = pred_embeddings @ caption_embedding_matrix.T
    best_idx = sims.argmax(dim=1).tolist()
    return [caption_list[i] for i in best_idx]


def evaluate(model, loader, loss_fn, device, caption_embedding_matrix, caption_list):
    model.eval()
    total_loss = 0.0
    total_count = 0
    correct = 0

    with torch.no_grad():
        for images, target_embeddings, true_captions, _ids in loader:
            images = images.to(device)
            target_embeddings = target_embeddings.to(device)

            pred_embeddings = model(images)
            pred_embeddings = F.normalize(pred_embeddings, dim=1)

            loss = loss_fn(pred_embeddings, target_embeddings)

            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)

            pred_captions = retrieve_caption(
                pred_embeddings.cpu(),
                caption_embedding_matrix,
                caption_list,
            )

            for pred_caption, true_caption in zip(pred_captions, true_captions):
                if pred_caption == true_caption:
                    correct += 1

    return {
        "loss": total_loss / total_count,
        "sentence_accuracy": correct / total_count,
    }


def save_predictions(model, loader, device, caption_embedding_matrix, caption_list, out_file: Path):
    model.eval()
    rows = []

    with torch.no_grad():
        for images, _target_embeddings, true_captions, ids in loader:
            images = images.to(device)
            pred_embeddings = model(images)
            pred_embeddings = F.normalize(pred_embeddings, dim=1)

            pred_captions = retrieve_caption(
                pred_embeddings.cpu(),
                caption_embedding_matrix,
                caption_list,
            )

            for sample_id, true_caption, pred_caption in zip(ids, true_captions, pred_captions):
                rows.append({
                    "id": sample_id,
                    "true_caption": true_caption,
                    "pred_caption": pred_caption,
                    "exact_match": pred_caption == true_caption,
                })

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def train(args):
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parents[2]
    paths_info = build_metadata_paths(project_root, args.dataset_name)

    if paths_info["format"] == "single":
        captions = collect_captions_from_files([paths_info["single_file"]])
    else:
        captions = collect_captions_from_files(
            [paths_info["train_file"], paths_info["val_file"], paths_info["test_file"]]
        )

    caption_to_embedding, caption_embedding_matrix, caption_list = build_caption_embeddings(
        captions,
        args.text_model,
    )

    train_dataset, val_dataset, test_dataset, metadata_source = make_datasets(
        paths_info,
        caption_to_embedding,
        args.image_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_dim = caption_embedding_matrix.shape[1]
    model = EmbeddingCNN(embedding_dim=embedding_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    caption_embedding_matrix = caption_embedding_matrix.to(torch.float32)

    run_dir = project_root / "artifacts" / args.dataset_name / "embedding_cnn"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    history = []

    print(f"Dataset: {args.dataset_name}")
    print(f"Metadata source: {metadata_source}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Unique captions: {len(caption_list)}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Text model: {args.text_model}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        train_correct = 0

        for images, target_embeddings, true_captions, _ids in train_loader:
            images = images.to(device)
            target_embeddings = target_embeddings.to(device)

            pred_embeddings = model(images)
            pred_embeddings = F.normalize(pred_embeddings, dim=1)

            loss = loss_fn(pred_embeddings, target_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            train_count += images.size(0)

            pred_captions = retrieve_caption(
                pred_embeddings.detach().cpu(),
                caption_embedding_matrix,
                caption_list,
            )
            for pred_caption, true_caption in zip(pred_captions, true_captions):
                if pred_caption == true_caption:
                    train_correct += 1

        train_metrics = {
            "loss": train_loss_sum / train_count,
            "sentence_accuracy": train_correct / train_count,
        }

        val_metrics = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            caption_embedding_matrix,
            caption_list,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_sentence_accuracy": train_metrics["sentence_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_sentence_accuracy": val_metrics["sentence_accuracy"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_sent_acc={train_metrics['sentence_accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_sent_acc={val_metrics['sentence_accuracy']:.4f}"
        )

        if val_metrics["sentence_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["sentence_accuracy"]
            torch.save(model.state_dict(), run_dir / "best_model.pth")

    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device))

    test_metrics = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
        caption_embedding_matrix,
        caption_list,
    )

    save_predictions(
        model,
        test_loader,
        device,
        caption_embedding_matrix,
        caption_list,
        run_dir / "test_predictions.json",
    )

    summary = {
        "dataset": args.dataset_name,
        "model": "embedding_cnn",
        "text_model": args.text_model,
        "metadata_source": metadata_source,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "embedding_dim": embedding_dim,
        "num_unique_captions": len(caption_list),
        "best_val_sentence_accuracy": best_val_acc,
        "test_loss": test_metrics["loss"],
        "test_sentence_accuracy": test_metrics["sentence_accuracy"],
        "history": history,
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(run_dir / "best_model.pth")
    print(run_dir / "metrics.json")
    print(run_dir / "test_predictions.json")
    print(f"\nFinal test sentence accuracy: {test_metrics['sentence_accuracy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="shapes", choices=["shapes", "numbers", "tictactoe"])
    parser.add_argument("--text-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
