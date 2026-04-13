import json
import random
from pathlib import Path
from PIL import Image, ImageDraw

from config import (
    COLORS, SHAPES, SIZES, RELATIONS,
    IMAGE_WIDTH, IMAGE_HEIGHT,
    SMALL_SIZE, LARGE_SIZE, SEED
)

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "shapes"
METADATA_ROOT = PROJECT_ROOT / "metadata"

METADATA_ROOT.mkdir(exist_ok=True, parents=True)
(DATA_ROOT / "train").mkdir(exist_ok=True, parents=True)
(DATA_ROOT / "val").mkdir(exist_ok=True, parents=True)
(DATA_ROOT / "test").mkdir(exist_ok=True, parents=True)

random.seed(SEED)


def generate_symbolic_state():
    return {
        "object_1": {
            "size": random.choice(SIZES),
            "color": random.choice(COLORS),
            "shape": random.choice(SHAPES)
        },
        "relation": random.choice(RELATIONS),
        "object_2": {
            "size": random.choice(SIZES),
            "color": random.choice(COLORS),
            "shape": random.choice(SHAPES)
        }
    }


def generate_caption(state):
    o1 = state["object_1"]
    o2 = state["object_2"]
    r = state["relation"]

    return f"a {o1['size']} {o1['color']} {o1['shape']} is {r} a {o2['size']} {o2['color']} {o2['shape']}"


def generate_canonical_label(state):
    return {
        "object_1_size": state["object_1"]["size"],
        "object_1_color": state["object_1"]["color"],
        "object_1_shape": state["object_1"]["shape"],
        "relation": state["relation"],
        "object_2_size": state["object_2"]["size"],
        "object_2_color": state["object_2"]["color"],
        "object_2_shape": state["object_2"]["shape"]
    }


def size_to_pixels(size_name):
    if size_name == "small":
        return SMALL_SIZE
    if size_name == "large":
        return LARGE_SIZE
    raise ValueError(f"Unknown size: {size_name}")


def draw_shape(draw, shape, color, center_x, center_y, size_px):
    half = size_px // 2
    left = center_x - half
    top = center_y - half
    right = center_x + half
    bottom = center_y + half

    if shape == "circle":
        draw.ellipse([left, top, right, bottom], fill=color)
    elif shape == "square":
        draw.rectangle([left, top, right, bottom], fill=color)
    elif shape == "triangle":
        points = [
            (center_x, top),
            (left, bottom),
            (right, bottom)
        ]
        draw.polygon(points, fill=color)
    else:
        raise ValueError(f"Unknown shape: {shape}")


def get_positions(relation):
    # fixed positions keep the relation obvious
    if relation == "above":
        return (128, 70), (128, 186)
    if relation == "below":
        return (128, 186), (128, 70)
    if relation == "left of":
        return (70, 128), (186, 128)
    if relation == "right of":
        return (186, 128), (70, 128)
    raise ValueError(f"Unknown relation: {relation}")


def render_image(state, save_path):
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    relation = state["relation"]
    (x1, y1), (x2, y2) = get_positions(relation)

    o1 = state["object_1"]
    o2 = state["object_2"]

    size1 = size_to_pixels(o1["size"])
    size2 = size_to_pixels(o2["size"])

    draw_shape(draw, o1["shape"], o1["color"], x1, y1, size1)
    draw_shape(draw, o2["shape"], o2["color"], x2, y2, size2)

    image.save(save_path)


def assign_split(index, total_count):
    train_cutoff = int(total_count * 0.8)
    val_cutoff = int(total_count * 0.9)

    if index < train_cutoff:
        return "train"
    if index < val_cutoff:
        return "val"
    return "test"


def generate_sample(index, split):
    sample_id = f"shapes_{split}_{index:06d}"

    state = generate_symbolic_state()
    caption = generate_caption(state)
    label = generate_canonical_label(state)

    image_rel_path = f"data/processed/shapes/{split}/{sample_id}.png"
    image_abs_path = PROJECT_ROOT / image_rel_path

    render_image(state, image_abs_path)

    return {
        "id": sample_id,
        "task": "shapes",
        "image_path": image_rel_path.replace("\\", "/"),
        "symbolic_state": state,
        "caption": caption,
        "canonical_label": label,
        "split": split
    }


def generate_dataset(total_count=1000):
    samples = []

    for i in range(1, total_count + 1):
        split = assign_split(i - 1, total_count)
        samples.append(generate_sample(i, split))

    return samples


def save_jsonl(samples, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    samples = generate_dataset(total_count=1000)

    output_file = METADATA_ROOT / "shapes_metadata.jsonl"
    save_jsonl(samples, output_file)

    print(f"Done. Generated {len(samples)} samples.")
    print(f"Metadata saved to: {output_file}")