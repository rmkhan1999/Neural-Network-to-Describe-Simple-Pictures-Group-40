from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


POSITION_ORDER = [
    "top left",
    "top middle",
    "top right",
    "middle left",
    "center",
    "middle right",
    "bottom left",
    "bottom middle",
    "bottom right",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/processed/tictactoe"))
    return parser.parse_args()


def parse_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_caption(x_positions: List[str], o_positions: List[str]) -> str:
    def format_positions(values: List[str]) -> str:
        if not values:
            return "none"
        ordered = sorted(values, key=lambda p: POSITION_ORDER.index(p))
        if len(ordered) == 1:
            return ordered[0]
        return " and ".join(ordered)

    return f"X is in {format_positions(x_positions)}; O is in {format_positions(o_positions)}"


def expected_label(x_positions: Set[str], o_positions: Set[str]) -> Dict[str, str]:
    label: Dict[str, str] = {}
    for p in POSITION_ORDER:
        if p in x_positions:
            label[p] = "X"
        elif p in o_positions:
            label[p] = "O"
        else:
            label[p] = "empty"
    return label


def validate_row(row: Dict[str, object], ids_seen: Set[str]) -> List[str]:
    errors: List[str] = []
    required_fields = ["id", "task", "image_path", "symbolic_state", "caption", "canonical_label", "split"]
    for field in required_fields:
        if field not in row:
            errors.append(f"missing field: {field}")

    sample_id = row.get("id")
    if isinstance(sample_id, str):
        if sample_id in ids_seen:
            errors.append(f"duplicate id: {sample_id}")
        ids_seen.add(sample_id)
    else:
        errors.append("id is not string")

    if row.get("task") != "tictactoe":
        errors.append("task is not tictactoe")

    symbolic = row.get("symbolic_state")
    if not isinstance(symbolic, dict):
        errors.append("symbolic_state is not object")
        return errors

    x_positions = symbolic.get("X")
    o_positions = symbolic.get("O")
    if not isinstance(x_positions, list) or not isinstance(o_positions, list):
        errors.append("symbolic_state must contain X and O arrays")
        return errors

    occupied = len(x_positions) + len(o_positions)
    if occupied < 1 or occupied > 6:
        errors.append("occupied cells out of range [1, 6]")

    used = set()
    for p in x_positions + o_positions:
        if p not in POSITION_ORDER:
            errors.append(f"invalid position: {p}")
        if p in used:
            errors.append(f"duplicate occupied position: {p}")
        used.add(p)

    caption = row.get("caption")
    if isinstance(caption, str):
        expected = build_caption(x_positions, o_positions)
        if caption != expected:
            errors.append("caption mismatch")
    else:
        errors.append("caption is not string")

    canonical = row.get("canonical_label")
    if isinstance(canonical, dict):
        expected = expected_label(set(x_positions), set(o_positions))
        if canonical != expected:
            errors.append("canonical_label mismatch")
    else:
        errors.append("canonical_label is not object")

    image_path = row.get("image_path")
    if not isinstance(image_path, str):
        errors.append("image_path is not string")
    return errors


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    split_rows = {
        "train": parse_jsonl(dataset_root / "train.jsonl"),
        "val": parse_jsonl(dataset_root / "val.jsonl"),
        "test": parse_jsonl(dataset_root / "test.jsonl"),
    }
    ids_seen: Set[str] = set()
    all_errors: List[str] = []

    for split, rows in split_rows.items():
        for i, row in enumerate(rows):
            row_errors = validate_row(row, ids_seen)
            for err in row_errors:
                all_errors.append(f"{split}:{i + 1}: {err}")
            image_path = row.get("image_path")
            if isinstance(image_path, str):
                full_image_path = Path.cwd() / image_path
                if not full_image_path.exists():
                    all_errors.append(f"{split}:{i + 1}: missing image file {image_path}")
            row_split = row.get("split")
            if row_split != split:
                all_errors.append(f"{split}:{i + 1}: split mismatch")

    total_rows = sum(len(rows) for rows in split_rows.values())
    print(json.dumps({"rows": total_rows, "errors": len(all_errors)}, indent=2))
    if all_errors:
        for err in all_errors[:200]:
            print(err)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
