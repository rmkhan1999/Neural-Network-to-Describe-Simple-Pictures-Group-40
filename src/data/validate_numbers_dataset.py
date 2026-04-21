# validate_numbers.py
# Validates the Numbers dataset against the data format spec (Section 5.1).
# Run this after generation before submitting or scaling up.
#
# Usage (Colab):
#   !python src/validate_numbers.py --data_dir .

import json
import argparse
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import COLORS, SIZES, DIGIT_LENGTHS, METADATA_DIR, METADATA_FILE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "numbers"
REQUIRED_FIELDS = {
    "id", "task", "image_path", "symbolic_state",
    "caption", "canonical_label", "split",
}
VALID_SPLITS = {"train", "val", "test"}


def check(data_dir: str, image_check_limit: int = 20) -> None:
    data_dir = Path(data_dir).resolve()
    meta_path = data_dir / METADATA_DIR / METADATA_FILE

    if not meta_path.exists():
        print(f"ERROR: metadata file not found at {meta_path}")
        return

    errors   = []
    warnings = []
    seen_ids = set()
    total    = 0
    split_counts = {}

    with open(meta_path, encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Checking {len(lines)} records in {meta_path} ...\n")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        total += 1
        # ── JSON parse ────────────────────────────────────────────────────────
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {i+1}: malformed JSON — {e}")
            continue

        rid = rec.get("id", f"<line {i+1}>")

        # ── Required fields ───────────────────────────────────────────────────
        missing = REQUIRED_FIELDS - rec.keys()
        if missing:
            errors.append(f"[{rid}] missing fields: {missing}")

        # ── task field ────────────────────────────────────────────────────────
        if rec.get("task") != "numbers":
            errors.append(f"[{rid}] task should be 'numbers', got '{rec.get('task')}'")

        # ── ID format: numbers_<split>_000001 ─────────────────────────────────
        parts = rid.split("_")
        if len(parts) != 3 or parts[0] != "numbers":
            errors.append(f"[{rid}] bad ID format (expected numbers_<split>_<6digits>)")
        elif not parts[2].isdigit() or len(parts[2]) != 6:
            errors.append(f"[{rid}] ID counter must be zero-padded to 6 digits")

        # ── ID uniqueness ─────────────────────────────────────────────────────
        if rid in seen_ids:
            errors.append(f"Duplicate ID: {rid}")
        seen_ids.add(rid)

        # ── Split field ───────────────────────────────────────────────────────
        split = rec.get("split", "")
        if split not in VALID_SPLITS:
            errors.append(f"[{rid}] invalid split '{split}'")
        else:
            split_counts[split] = split_counts.get(split, 0) + 1
        # ── Symbolic state ────────────────────────────────────────────────────
        ss = rec.get("symbolic_state", {})
        color  = ss.get("color")
        size   = ss.get("size")
        digits = str(ss.get("digits", ""))

        if color not in COLORS:
            errors.append(f"[{rid}] invalid color: '{color}'")
        if size not in SIZES:
            errors.append(f"[{rid}] invalid size: '{size}'")
        if not digits.isdigit() or digits != digits.lstrip("0") or len(digits) == 0:
            errors.append(f"[{rid}] invalid digits: '{digits}' (must be positive int, no leading zeros)")
        if len(digits) not in DIGIT_LENGTHS:
            errors.append(f"[{rid}] digit length {len(digits)} not in allowed set {DIGIT_LENGTHS}")

        # ── Caption format ────────────────────────────────────────────────────
        cap      = rec.get("caption", "")
        expected = f"a {size} {color} {digits}"
        if cap != expected:
            errors.append(f"[{rid}] caption mismatch\n   got:      '{cap}'\n   expected: '{expected}'")

        # ── Canonical label ───────────────────────────────────────────────────
        cl = rec.get("canonical_label", {})
        for key in ("size", "color", "digits"):
            if cl.get(key) != ss.get(key):
                errors.append(f"[{rid}] canonical_label.{key} ('{cl.get(key)}') != symbolic_state.{key} ('{ss.get(key)}')")
        if cl.get("length") != len(digits):
            errors.append(f"[{rid}] canonical_label.length ({cl.get('length')}) != len(digits) ({len(digits)})")

        # ── Image file (first image_check_limit samples only) ─────────────────
        if i < image_check_limit:
            img_path = data_dir / rec.get("image_path", "")
            if not img_path.exists():
                errors.append(f"[{rid}] image not found: {img_path}")
            else:
                try:
                    with Image.open(img_path) as img:
                        if img.format != "PNG":
                            errors.append(f"[{rid}] image is not PNG (got {img.format})")
                        if img.size[0] == 0 or img.size[1] == 0:
                            errors.append(f"[{rid}] image has zero dimension")
                except Exception as e:
                    errors.append(f"[{rid}] corrupted image — {e}")

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"Total records : {total}")
    print(f"Split counts  : {split_counts}")
    print()

    if errors:
        print(f"FAILED — {len(errors)} error(s) found:\n")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("PASSED — all checks OK!")

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  ! {w}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the Numbers dataset (Section 5.1).")
    parser.add_argument("--data_dir",          type=str, default=str(DEFAULT_DATA_DIR),
                    help="Root of the numbers dataset folder.")
    parser.add_argument("--image_check_limit", type=int, default=20,
                        help="How many images to physically open and check (default: 20).")
    args = parser.parse_args()
    check(args.data_dir, args.image_check_limit)
