from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw


POSITIONS = {
    "TL": (0, 0),
    "TM": (0, 1),
    "TR": (0, 2),
    "ML": (1, 0),
    "C": (1, 1),
    "MR": (1, 2),
    "BL": (2, 0),
    "BM": (2, 1),
    "BR": (2, 2),
}
COORD_TO_NAME = {v: k for k, v in POSITIONS.items()}
ALIASES = {"TC": "TM", "BC": "BM", "CTR": "C", "M": "C", "MM": "C"}
SHORT_TO_LONG = {
    "TL": "top left",
    "TM": "top middle",
    "TR": "top right",
    "ML": "middle left",
    "C": "center",
    "MR": "middle right",
    "BL": "bottom left",
    "BM": "bottom middle",
    "BR": "bottom right",
}
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
LONG_TO_SHORT = {v: k for k, v in SHORT_TO_LONG.items()}


@dataclass(frozen=True)
class GeneratorConfig:
    num_samples: int
    seed: int
    output_root: Path
    previews_root: Path
    image_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    min_occupied: int
    max_occupied: int
    allow_single_player: bool


@dataclass
class Board:
    x: List[str] = field(default_factory=list)
    o: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.x = [normalize_position_name(p) for p in self.x]
        self.o = [normalize_position_name(p) for p in self.o]
        self._validate()

    def _validate(self) -> None:
        x_coords = [resolve(p) for p in self.x]
        o_coords = [resolve(p) for p in self.o]
        all_coords = x_coords + o_coords
        if len(all_coords) != len(set(all_coords)):
            raise ValueError("duplicate positions detected")
        if not (len(self.x) == len(self.o) or len(self.x) == len(self.o) + 1):
            raise ValueError("invalid move counts for legal tic-tac-toe state")

    def to_grid(self) -> List[List[str]]:
        grid = [[" "] * 3 for _ in range(3)]
        for pos in self.x:
            r, c = resolve(pos)
            grid[r][c] = "X"
        for pos in self.o:
            r, c = resolve(pos)
            grid[r][c] = "O"
        return grid

    def winner(self) -> str | None:
        grid = self.to_grid()
        lines = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)],
        ]
        for line in lines:
            values = {grid[r][c] for r, c in line}
            if values == {"X"}:
                return "X"
            if values == {"O"}:
                return "O"
        return None


def normalize_position_name(pos: str) -> str:
    value = pos.strip().upper()
    return ALIASES.get(value, value)


def resolve(pos: str) -> Tuple[int, int]:
    key = normalize_position_name(pos)
    if key not in POSITIONS:
        raise ValueError(f"unknown position: {pos}")
    return POSITIONS[key]


def parse(notation: str) -> Board:
    x_positions: List[str] = []
    o_positions: List[str] = []
    for token in notation.upper().split():
        if ":" not in token:
            raise ValueError(f"invalid notation token: {token}")
        player, _, pos_str = token.partition(":")
        positions = [p.strip() for p in pos_str.split(",") if p.strip()]
        if player == "X":
            x_positions = positions
        elif player == "O":
            o_positions = positions
        else:
            raise ValueError(f"unknown player: {player}")
    return Board(x=x_positions, o=o_positions)


def to_notation(board: Board) -> str:
    parts: List[str] = []
    if board.x:
        parts.append("X:" + ",".join(board.x))
    if board.o:
        parts.append("O:" + ",".join(board.o))
    return " ".join(parts)


def parse_args() -> GeneratorConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/tictactoe"))
    parser.add_argument("--previews-root", type=Path, default=Path("data/previews/tictactoe"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--min-occupied", type=int, default=1)
    parser.add_argument("--max-occupied", type=int, default=6)
    parser.add_argument("--allow-single-player", action="store_true")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if args.min_occupied < 1:
        raise ValueError("min_occupied must be at least 1")
    if args.max_occupied > 6 or args.max_occupied < args.min_occupied:
        raise ValueError("max_occupied must be in [min_occupied, 6]")
    if not args.allow_single_player and args.min_occupied < 2:
        args.min_occupied = 2

    return GeneratorConfig(
        num_samples=args.num_samples,
        seed=args.seed,
        output_root=args.output_root,
        previews_root=args.previews_root,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_occupied=args.min_occupied,
        max_occupied=args.max_occupied,
        allow_single_player=args.allow_single_player,
    )


def ordered_positions(positions: Sequence[str]) -> List[str]:
    order = {name: idx for idx, name in enumerate(POSITION_ORDER)}
    return sorted(positions, key=lambda x: order[x])


def positions_to_text(positions: Sequence[str]) -> str:
    if not positions:
        return "none"
    ordered = ordered_positions(positions)
    if len(ordered) == 1:
        return ordered[0]
    return " and ".join(ordered)


def board_to_symbolic_state(board: Board) -> Dict[str, List[str]]:
    x_positions = ordered_positions([SHORT_TO_LONG[p] for p in board.x])
    o_positions = ordered_positions([SHORT_TO_LONG[p] for p in board.o])
    return {"X": x_positions, "O": o_positions}


def symbolic_state_to_board(symbolic_state: Dict[str, List[str]]) -> Board:
    x_positions = [LONG_TO_SHORT[p] for p in symbolic_state["X"]]
    o_positions = [LONG_TO_SHORT[p] for p in symbolic_state["O"]]
    return Board(x=x_positions, o=o_positions)


def random_board(rng: random.Random, min_occupied: int, max_occupied: int, allow_single_player: bool) -> Board:
    while True:
        n = rng.randint(min_occupied, max_occupied)
        board = random_board_with_n_moves(rng, n)
        if not allow_single_player and (len(board.x) == 0 or len(board.o) == 0):
            continue
        return board


def random_board_with_n_moves(rng: random.Random, n: int) -> Board:
    if n < 1 or n > 9:
        raise ValueError("n must be between 1 and 9")
    all_positions = list(POSITIONS.keys())
    while True:
        chosen = rng.sample(all_positions, n)
        x_count = (n + 1) // 2
        x_pos = chosen[:x_count]
        o_pos = chosen[x_count:]
        board = Board(x=x_pos, o=o_pos)
        if is_reachable(board):
            return board


def is_reachable(board: Board) -> bool:
    winner = board.winner()
    if winner == "X" and len(board.x) != len(board.o) + 1:
        return False
    if winner == "O" and len(board.x) != len(board.o):
        return False
    last_player = "x" if len(board.x) > len(board.o) else "o"
    prev_x = board.x[:-1] if last_player == "x" else board.x
    prev_o = board.o[:-1] if last_player == "o" else board.o
    if prev_x or prev_o or len(board.x) + len(board.o) > 1:
        prev_board = Board(x=prev_x, o=prev_o)
        if prev_board.winner() is not None:
            return False
    return True


def build_caption(symbolic_state: Dict[str, List[str]]) -> str:
    x_text = positions_to_text(symbolic_state["X"])
    o_text = positions_to_text(symbolic_state["O"])
    return f"X is in {x_text}; O is in {o_text}"


def build_canonical_label(symbolic_state: Dict[str, List[str]]) -> Dict[str, str]:
    x_set = set(symbolic_state["X"])
    o_set = set(symbolic_state["O"])
    label: Dict[str, str] = {}
    for pos in POSITION_ORDER:
        if pos in x_set:
            label[pos] = "X"
        elif pos in o_set:
            label[pos] = "O"
        else:
            label[pos] = "empty"
    return label


def draw_rotated_x(draw: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float, stroke: int) -> None:
    draw.line((x0, y0, x1, y1), fill=(30, 30, 30), width=stroke)
    draw.line((x0, y1, x1, y0), fill=(30, 30, 30), width=stroke)


def render_board(board: Board, image_path: Path, image_size: int, rng: random.Random) -> None:
    image = Image.new("RGB", (image_size, image_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    margin = int(image_size * 0.08)
    board_size = image_size - 2 * margin
    cell_size = board_size / 3.0
    line_width = max(3, image_size // 64)
    symbol_width = max(4, image_size // 42)
    jitter = max(1, image_size // 120)

    for i in [1, 2]:
        x = margin + i * cell_size + rng.randint(-jitter, jitter)
        draw.line((x, margin, x, margin + board_size), fill=(0, 0, 0), width=line_width)
        y = margin + i * cell_size + rng.randint(-jitter, jitter)
        draw.line((margin, y, margin + board_size, y), fill=(0, 0, 0), width=line_width)

    grid = board.to_grid()
    for r in range(3):
        for c in range(3):
            sym = grid[r][c]
            if sym == " ":
                continue
            cx0 = margin + c * cell_size
            cy0 = margin + r * cell_size
            cx1 = cx0 + cell_size
            cy1 = cy0 + cell_size
            p = int(cell_size * 0.24)
            ox = rng.randint(-jitter, jitter)
            oy = rng.randint(-jitter, jitter)
            if sym == "X":
                draw_rotated_x(draw, cx0 + p + ox, cy0 + p + oy, cx1 - p + ox, cy1 - p + oy, symbol_width)
            else:
                draw.ellipse(
                    (cx0 + p + ox, cy0 + p + oy, cx1 - p + ox, cy1 - p + oy),
                    outline=(20, 20, 20),
                    width=symbol_width,
                )

    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path, format="PNG")


def split_indices(num_samples: int, seed: int, train_ratio: float, val_ratio: float) -> Dict[str, List[int]]:
    indices = list(range(num_samples))
    random.Random(seed + 1).shuffle(indices)
    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def build_summary(samples: List[Dict[str, object]]) -> Dict[str, object]:
    occupied_counts = {str(i): 0 for i in range(1, 7)}
    position_counts = {p: 0 for p in POSITION_ORDER}
    symbol_counts = {"X": 0, "O": 0}

    for sample in samples:
        symbolic_state = sample["symbolic_state"]
        if not isinstance(symbolic_state, dict):
            continue
        x_positions = symbolic_state["X"]
        o_positions = symbolic_state["O"]
        occupied_counts[str(len(x_positions) + len(o_positions))] += 1
        for p in x_positions:
            position_counts[p] += 1
            symbol_counts["X"] += 1
        for p in o_positions:
            position_counts[p] += 1
            symbol_counts["O"] += 1

    return {
        "samples": len(samples),
        "occupied_cell_count_distribution": occupied_counts,
        "board_position_frequency": position_counts,
        "symbol_frequency": symbol_counts,
    }


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_dataset(config: GeneratorConfig) -> None:
    rng = random.Random(config.seed)
    split_map = split_indices(config.num_samples, config.seed, config.train_ratio, config.val_ratio)
    index_to_split: Dict[int, str] = {}
    for split, ids in split_map.items():
        for idx in ids:
            index_to_split[idx] = split

    counters = {"train": 0, "val": 0, "test": 0}
    rows_by_split: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    all_rows: List[Dict[str, object]] = []

    for sample_index in range(config.num_samples):
        split = index_to_split[sample_index]
        counters[split] += 1
        sample_id = f"tictactoe_{split}_{counters[split]:06d}"

        board = random_board(rng, config.min_occupied, config.max_occupied, config.allow_single_player)
        symbolic_state = board_to_symbolic_state(board)
        caption = build_caption(symbolic_state)
        canonical_label = build_canonical_label(symbolic_state)

        relative_image_path = Path("data/processed/tictactoe") / split / f"{sample_id}.png"
        absolute_image_path = Path.cwd() / relative_image_path
        render_board(board, absolute_image_path, config.image_size, rng)

        row = {
            "id": sample_id,
            "task": "tictactoe",
            "image_path": str(relative_image_path).replace("\\", "/"),
            "symbolic_state": symbolic_state,
            "caption": caption,
            "canonical_label": canonical_label,
            "split": split,
            "notation": to_notation(symbolic_state_to_board(symbolic_state)),
        }
        rows_by_split[split].append(row)
        all_rows.append(row)

    for split in ["train", "val", "test"]:
        write_jsonl(config.output_root / f"{split}.jsonl", rows_by_split[split])

    summary = build_summary(all_rows)
    with (config.output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    generation_config = {
        "task": "tictactoe",
        "seed": config.seed,
        "num_samples": config.num_samples,
        "split_ratio": {"train": config.train_ratio, "val": config.val_ratio, "test": config.test_ratio},
        "min_occupied": config.min_occupied,
        "max_occupied": config.max_occupied,
        "allow_single_player": config.allow_single_player,
        "position_order": POSITION_ORDER,
        "caption_format": "X is in [positions]; O is in [positions]",
    }
    with (config.output_root / "generation_config.json").open("w", encoding="utf-8") as f:
        json.dump(generation_config, f, ensure_ascii=False, indent=2)

    preview_rows = all_rows[: min(20, len(all_rows))]
    preview_dir = config.previews_root
    preview_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(preview_dir / "preview_samples.jsonl", preview_rows)


if __name__ == "__main__":
    generate_dataset(parse_args())
