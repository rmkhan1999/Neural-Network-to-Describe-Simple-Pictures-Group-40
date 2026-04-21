# ──Shapes Dataset Config ───────────────────────────────────────────
COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["circle", "square", "triangle"]
SIZES = ["small", "large"]
RELATIONS = ["above", "below", "left of", "right of"]


IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

SMALL_SIZE = 40
LARGE_SIZE = 70

SEED = 42
# ── Numbers Dataset Config ───────────────────────────────────────────

DIGIT_LENGTHS = [1, 2, 3, 4]

COLOR_RGB = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 128, 0),
    "yellow": (255, 255, 0),
}

# Numbers uses separate naming from shapes
IMAGE_W = 128
IMAGE_H = 128

BACKGROUND_COLOR = (255, 255, 255)

FONT_SIZE = {
    "small": 24,
    "large": 52,
}

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/Library/Fonts/Arial.ttf",
]

SPLIT_RATIOS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}

METADATA_DIR = "metadata"
METADATA_FILE = "numbers_metadata.jsonl"

DEFAULT_SEED = 42
DEFAULT_NUM_SAMPLES = 1000
PILOT_NUM_SAMPLES = 80

VERSION = "1.0"
