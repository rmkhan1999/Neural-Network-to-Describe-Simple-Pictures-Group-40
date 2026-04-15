# config.py
# Central config for the Numbers dataset generator.
# ALL vocabulary is defined here — do not hardcode values elsewhere.
# Section 3B.2 of the data generation format spec.

# ── Vocabulary (frozen for v1) ────────────────────────────────────────────────
COLORS = ["red", "blue", "green", "yellow"]
SIZES  = ["small", "large"]
DIGIT_LENGTHS = [1, 2, 3, 4]   # number of digits in generated number

# RGB tuples for each allowed color
COLOR_RGB = {
    "red":    (220, 50,  50),
    "blue":   (50,  100, 220),
    "green":  (30,  160, 60),
    "yellow": (200, 170, 0),
}

# ── Image rendering ───────────────────────────────────────────────────────────
IMAGE_W = 128
IMAGE_H = 128
BACKGROUND_COLOR = (255, 255, 255)   # plain white — Section 2.10

# Font sizes — must be visually distinct between small and large (Section 3B.7)
FONT_SIZE = {
    "small": 24,
    "large": 52,
}

# Candidate system fonts tried in order (Colab has DejaVu by default)
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
]

# ── Dataset split ratios — Section 2.6 ───────────────────────────────────────
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}

# ── Paths (relative to project root) ─────────────────────────────────────────
METADATA_DIR    = "metadata"
SAMPLE_IMG_DIR  = "sample_images"
METADATA_FILE   = "numbers_metadata.jsonl"

# ── Generation defaults ───────────────────────────────────────────────────────
DEFAULT_SEED        = 42
DEFAULT_NUM_SAMPLES = 1000
PILOT_NUM_SAMPLES   = 80    # Section 6.1 — pilot before full generation

# ── Dataset version ───────────────────────────────────────────────────────────
VERSION = "1.0"
