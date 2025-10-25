from pathlib import Path
from PIL import Image

def list_images(root):
    exts = {".jpg",".jpeg",".png",".webp",".bmp"}
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in exts]

def load_image(path):
    return Image.open(path).convert("RGB")
