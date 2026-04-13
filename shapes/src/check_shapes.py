import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_ROOT / "metadata" / "shapes_metadata.jsonl"

ALLOWED_TASK = "shapes"
ALLOWED_SPLITS = {"train", "val", "test"}
ALLOWED_COLORS = {"red", "blue", "green", "yellow"}
ALLOWED_SHAPES = {"circle", "square", "triangle"}
ALLOWED_SIZES = {"small", "large"}
ALLOWED_RELATIONS = {"above", "below", "left of", "right of"}


def load_samples(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def check_required_fields(sample):
    required = {
        "id", "task", "image_path", "symbolic_state",
        "caption", "canonical_label", "split"
    }
    missing = required - set(sample.keys())
    return list(missing)


def check_vocab(sample):
    errors = []

    state = sample["symbolic_state"]
    o1 = state["object_1"]
    o2 = state["object_2"]
    relation = state["relation"]

    if o1["color"] not in ALLOWED_COLORS:
        errors.append(f"invalid object_1 color: {o1['color']}")
    if o2["color"] not in ALLOWED_COLORS:
        errors.append(f"invalid object_2 color: {o2['color']}")

    if o1["shape"] not in ALLOWED_SHAPES:
        errors.append(f"invalid object_1 shape: {o1['shape']}")
    if o2["shape"] not in ALLOWED_SHAPES:
        errors.append(f"invalid object_2 shape: {o2['shape']}")

    if o1["size"] not in ALLOWED_SIZES:
        errors.append(f"invalid object_1 size: {o1['size']}")
    if o2["size"] not in ALLOWED_SIZES:
        errors.append(f"invalid object_2 size: {o2['size']}")

    if relation not in ALLOWED_RELATIONS:
        errors.append(f"invalid relation: {relation}")

    return errors


def check_caption(sample):
    state = sample["symbolic_state"]
    o1 = state["object_1"]
    o2 = state["object_2"]
    relation = state["relation"]

    expected = f"a {o1['size']} {o1['color']} {o1['shape']} is {relation} a {o2['size']} {o2['color']} {o2['shape']}"
    return expected == sample["caption"], expected


def check_image_exists(sample):
    image_path = PROJECT_ROOT / sample["image_path"]
    return image_path.exists()


def main():
    samples = load_samples(METADATA_FILE)
    print(f"Loaded {len(samples)} samples")

    seen_ids = set()
    total_errors = 0

    for sample in samples:
        sample_id = sample.get("id", "UNKNOWN")

        missing = check_required_fields(sample)
        if missing:
            total_errors += 1
            print(f"[{sample_id}] Missing fields: {missing}")

        if sample.get("task") != ALLOWED_TASK:
            total_errors += 1
            print(f"[{sample_id}] Invalid task: {sample.get('task')}")

        if sample.get("split") not in ALLOWED_SPLITS:
            total_errors += 1
            print(f"[{sample_id}] Invalid split: {sample.get('split')}")

        if sample_id in seen_ids:
            total_errors += 1
            print(f"[{sample_id}] Duplicate ID")
        seen_ids.add(sample_id)

        vocab_errors = check_vocab(sample)
        for err in vocab_errors:
            total_errors += 1
            print(f"[{sample_id}] {err}")

        caption_ok, expected_caption = check_caption(sample)
        if not caption_ok:
            total_errors += 1
            print(f"[{sample_id}] Caption mismatch")
            print(f"  expected: {expected_caption}")
            print(f"  actual:   {sample['caption']}")

        if not check_image_exists(sample):
            total_errors += 1
            print(f"[{sample_id}] Image file missing: {sample['image_path']}")

    if total_errors == 0:
        print("All checks passed")
    else:
        print(f"Finished with {total_errors} issue(s)")


if __name__ == "__main__":
    main()