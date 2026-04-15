# summarise_numbers.py
# Produces a frequency summary of the Numbers dataset — Section 5.3.
# Shows counts by color, size, digit length, and split.
#
# Usage (Colab):
#   !python src/summarise_numbers.py --data_dir .

import json
import argparse
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import METADATA_DIR, METADATA_FILE


def summarise(data_dir: str) -> None:
    data_dir  = Path(data_dir)
    meta_path = data_dir / METADATA_DIR / METADATA_FILE

    if not meta_path.exists():
        print(f"ERROR: metadata file not found at {meta_path}")
        return

    counts = {
        "color":        defaultdict(int),
        "size":         defaultdict(int),
        "digit_length": defaultdict(int),
        "split":        defaultdict(int),
    }
    total = 0

    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec    = json.loads(line)
            ss     = rec.get("symbolic_state", {})
            digits = str(ss.get("digits", ""))
            total += 1

            counts["color"][ss.get("color", "?")] += 1
            counts["size"][ss.get("size", "?")]   += 1
            counts["digit_length"][str(len(digits))] += 1
            counts["split"][rec.get("split", "?")]   += 1

    # ── Pretty print ──────────────────────────────────────────────────────────
    print(f"Numbers dataset summary — {total} total samples")
    print("=" * 45)

    for category, freq in counts.items():
        print(f"\n{category.upper()}:")
        for key in sorted(freq.keys()):
            bar   = "#" * (freq[key] * 30 // total)
            pct   = freq[key] / total * 100
            print(f"  {key:<15} {freq[key]:>5}  ({pct:5.1f}%)  {bar}")

    # ── Save summary JSON ─────────────────────────────────────────────────────
    out_path = data_dir / "frequency_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({k: dict(v) for k, v in counts.items()}, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frequency summary for the Numbers dataset.")
    parser.add_argument("--data_dir", type=str, default=".",
                        help="Root of the numbers dataset folder.")
    args = parser.parse_args()
    summarise(args.data_dir)
