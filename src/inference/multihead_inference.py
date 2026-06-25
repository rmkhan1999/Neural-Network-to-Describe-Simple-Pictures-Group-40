from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from src.models.multihead_cnn import StructuredCNN
from src.data.multihead_dataset import reconstruct_caption

POSITION_ORDER = [
    "top left", "top middle", "top right",
    "middle left", "center", "middle right",
    "bottom left", "bottom middle", "bottom right",
]

# Label maps are deterministic from the generation configs:
#   shapes/numbers: sorted(set(values)) — alphabetical
#   tictactoe:      hardcoded {empty:0, X:1, O:2} in multihead_dataset.py
#   numbers digits: ["<pad>"] + ["0".."9"] — insertion order
LABEL_MAPS: dict[str, dict[str, dict[str, int]]] = {
    "shapes": {
        "object_1_size":  {"large": 0, "small": 1},
        "object_1_color": {"blue": 0, "green": 1, "red": 2, "yellow": 3},
        "object_1_shape": {"circle": 0, "square": 1, "triangle": 2},
        "relation":       {"above": 0, "below": 1, "left of": 2, "right of": 3},
        "object_2_size":  {"large": 0, "small": 1},
        "object_2_color": {"blue": 0, "green": 1, "red": 2, "yellow": 3},
        "object_2_shape": {"circle": 0, "square": 1, "triangle": 2},
    },
    "numbers": {
        "size":    {"large": 0, "small": 1},
        "color":   {"blue": 0, "green": 1, "red": 2, "yellow": 3},
        "length":  {"1": 0, "2": 1, "3": 2, "4": 3},
        "digit_0": {"<pad>": 0, **{str(i): i + 1 for i in range(10)}},
        "digit_1": {"<pad>": 0, **{str(i): i + 1 for i in range(10)}},
        "digit_2": {"<pad>": 0, **{str(i): i + 1 for i in range(10)}},
        "digit_3": {"<pad>": 0, **{str(i): i + 1 for i in range(10)}},
    },
    "tictactoe": {
        pos: {"empty": 0, "X": 1, "O": 2} for pos in POSITION_ORDER
    },
}

_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


def load_model(task: str, project_root: Path, device: torch.device) -> tuple[StructuredCNN, dict]:
    label_maps = LABEL_MAPS[task]
    head_dims = {k: len(v) for k, v in label_maps.items()}
    model = StructuredCNN(head_dims)
    weights = project_root / "artifacts" / task / "multihead_cnn" / "best_model.pth"
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()
    return model, label_maps


def predict(
    model: StructuredCNN,
    label_maps: dict,
    image: Image.Image,
    task: str,
    device: torch.device,
) -> tuple[str, dict[str, dict]]:
    """
    Returns (caption, head_results) where head_results maps
    head_name → {label, confidence, probs: {label: float}}.
    """
    tensor = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    inverse = {k: {v: lbl for lbl, v in m.items()} for k, m in label_maps.items()}
    pred_values: dict[str, str] = {}
    head_results: dict[str, dict] = {}

    for head, logits in outputs.items():
        probs = F.softmax(logits[0], dim=0).cpu()
        idx = int(probs.argmax().item())
        label = inverse[head][idx]
        pred_values[head] = label
        head_results[head] = {
            "label": label,
            "confidence": float(probs[idx].item()),
            "probs": {inverse[head][i]: float(probs[i].item()) for i in range(len(probs))},
        }

    caption = reconstruct_caption(task, pred_values)
    return caption, head_results
