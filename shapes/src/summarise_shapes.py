import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_ROOT / "metadata" / "shapes_metadata.jsonl"


def load_samples(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def main():
    samples = load_samples(METADATA_FILE)

    object_1_shapes = Counter()
    object_2_shapes = Counter()
    object_1_colors = Counter()
    object_2_colors = Counter()
    object_1_sizes = Counter()
    object_2_sizes = Counter()
    relations = Counter()
    splits = Counter()

    for sample in samples:
        state = sample["symbolic_state"]
        o1 = state["object_1"]
        o2 = state["object_2"]

        object_1_shapes[o1["shape"]] += 1
        object_2_shapes[o2["shape"]] += 1
        object_1_colors[o1["color"]] += 1
        object_2_colors[o2["color"]] += 1
        object_1_sizes[o1["size"]] += 1
        object_2_sizes[o2["size"]] += 1
        relations[state["relation"]] += 1
        splits[sample["split"]] += 1

    print("Split counts:")
    print(splits)
    print()

    print("Object 1 shapes:")
    print(object_1_shapes)
    print()

    print("Object 2 shapes:")
    print(object_2_shapes)
    print()

    print("Object 1 colors:")
    print(object_1_colors)
    print()

    print("Object 2 colors:")
    print(object_2_colors)
    print()

    print("Object 1 sizes:")
    print(object_1_sizes)
    print()

    print("Object 2 sizes:")
    print(object_2_sizes)
    print()

    print("Relations:")
    print(relations)


if __name__ == "__main__":
    main()