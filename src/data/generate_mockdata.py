from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMG_DIR = PROJECT_ROOT / "data" / "processed" / "mock" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

for i in range(1, 5):
    img = Image.new("RGB", (64, 64), "white")
    img.save(IMG_DIR / f"img{i}.png")

print(f"Mock images created at {IMG_DIR}")
