# generate_numbers.py
# Generates the Numbers dataset: images + metadata.
# Section 3B of the data generation format spec.
#
# Usage (Colab):
#   !python src/generate_numbers.py --num_samples 1000 --seed 42
#   !python src/generate_numbers.py --pilot            # 80 samples for manual review

import os
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    COLORS, SIZES, DIGIT_LENGTHS, COLOR_RGB,
    IMAGE_W, IMAGE_H, BACKGROUND_COLOR, FONT_SIZE, FONT_CANDIDATES,
    SPLIT_RATIOS, METADATA_DIR, METADATA_FILE,
    DEFAULT_SEED, DEFAULT_NUM_SAMPLES, PILOT_NUM_SAMPLES, VERSION,
)


# ── Rendering ─────────────────────────────────────────────────────────────────

def load_font(size: int, font_path: str | None = None) -> ImageFont.FreeTypeFont:
    """Load a monospace font at the given size. Falls back to default."""
    candidates = ([font_path] if font_path else []) + FONT_CANDIDATES
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_image(size: str, color: str, digits: str,
                 font_path: str | None = None) -> Image.Image:
    """
    Render a plain-background image with a centered colored number.
    Sections 3B.7 and 3B.8.
    """
    img  = Image.new("RGB", (IMAGE_W, IMAGE_H), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    font = load_font(FONT_SIZE[size], font_path)
    rgb  = COLOR_RGB[color]

    # Center text precisely
    bbox   = draw.textbbox((0, 0), digits, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (IMAGE_W - text_w) // 2 - bbox[0]
    y = (IMAGE_H - text_h) // 2 - bbox[1]

    draw.text((x, y), digits, fill=rgb, font=font)
    return img


# ── Sampling helpers ──────────────────────────────────────────────────────────

def sample_number(length: int, rng: random.Random) -> str:
    """Random positive integer string of given length, no leading zeros. §3B.3"""
    low  = 10 ** (length - 1) if length > 1 else 1
    high = 10 ** length - 1
    return str(rng.randint(low, high))


def build_symbolic_state(size: str, color: str, digits: str) -> dict:
    """Section 3B.5"""
    return {"size": size, "color": color, "digits": digits}


def build_canonical_label(size: str, color: str, digits: str) -> dict:
    """Section 3B.6"""
    return {"size": size, "color": color, "digits": digits, "length": len(digits)}


def build_caption(size: str, color: str, digits: str) -> str:
    """Section 3B.4: a [size] [color] [number]"""
    return f"a {size} {color} {digits}"


def assign_splits(n: int, rng: random.Random) -> list:
    """Deterministic split assignment — Section 2.6."""
    indices = list(range(n))
    rng.shuffle(indices)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val   = int(n * SPLIT_RATIOS["val"])
    splits  = [""] * n
    for i in indices[:n_train]:
        splits[i] = "train"
    for i in indices[n_train: n_train + n_val]:
        splits[i] = "val"
    for i in indices[n_train + n_val:]:
        splits[i] = "test"
    return splits


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(num_samples: int, seed: int, output_dir: str,
             font_path: str | None = None) -> None:

    rng        = random.Random(seed)
    output_dir = Path(output_dir)
    meta_dir   = output_dir / METADATA_DIR
    img_dir    = output_dir / "images"

    meta_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # ── Balanced attribute sampling — Section 3B.9 ───────────────────────────
    # Cycle through every (color, size, digit_length) combo to avoid skew,
    # then shuffle with the seeded RNG.
    combo = [(c, s, l) for c in COLORS for s in SIZES for l in DIGIT_LENGTHS]
    attributes = []
    for i in range(num_samples):
        color, size, length = combo[i % len(combo)]
        digits = sample_number(length, rng)
        attributes.append((color, size, digits))
    rng.shuffle(attributes)

    split_labels   = assign_splits(num_samples, rng)
    split_counters = defaultdict(int)
    records        = []

    print(f"Generating {num_samples} samples (seed={seed}) ...")

    for (color, size, digits), split in zip(attributes, split_labels):
        split_counters[split] += 1
        n = split_counters[split]

        sample_id  = f"numbers_{split}_{n:06d}"
        image_name = f"{sample_id}.png"
        image_path = f"images/{image_name}"   # relative path stored in metadata

        # Build record fields
        sym   = build_symbolic_state(size, color, digits)
        label = build_canonical_label(size, color, digits)
        cap   = build_caption(size, color, digits)

        record = {
            "id":              sample_id,
            "task":            "numbers",
            "image_path":      image_path,
            "symbolic_state":  sym,
            "caption":         cap,
            "canonical_label": label,
            "split":           split,
        }
        records.append(record)

        # Render and save image
        img = render_image(size, color, digits, font_path)
        img.save(img_dir / image_name)

    # ── Write single metadata JSONL — matches friend's shapes structure ───────
    meta_path = meta_dir / METADATA_FILE
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Save generation config — Section 2.5 ─────────────────────────────────
    config_out = {
        "task": "numbers", "seed": seed, "num_samples": num_samples,
        "split_ratios": SPLIT_RATIOS,
        "vocabulary": {"colors": COLORS, "sizes": SIZES, "digit_lengths": DIGIT_LENGTHS},
        "image_size": [IMAGE_W, IMAGE_H], "font_sizes": FONT_SIZE,
        "caption_format": "a [size] [color] [number]", "version": VERSION,
    }
    with open(output_dir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2)

    # ── Print split summary ───────────────────────────────────────────────────
    print(f"\nSplit counts: { {k: v for k, v in split_counters.items()} }")
    print(f"Metadata  -> {meta_path}")
    print(f"Images    -> {img_dir}/")
    print("Done!")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the Numbers dataset.")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed",        type=int, default=DEFAULT_SEED)
    parser.add_argument("--output_dir",  type=str, default=".")
    parser.add_argument("--font_path",   type=str, default=None)
    parser.add_argument("--pilot", action="store_true",
                        help=f"Generate only {PILOT_NUM_SAMPLES} samples for manual review (Section 6.1).")
    args = parser.parse_args()

    n = PILOT_NUM_SAMPLES if args.pilot else args.num_samples
    generate(n, args.seed, args.output_dir, args.font_path)
