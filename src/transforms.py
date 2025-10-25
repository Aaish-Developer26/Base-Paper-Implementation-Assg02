from PIL import Image, ImageFilter
import io, random
import numpy as np

def jpeg_compress_pil(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def resize_pil(img: Image.Image, scale: float) -> Image.Image:
    if scale == 1.0: return img
    w, h = img.size
    return img.resize((max(8, int(w*scale)), max(8, int(h*scale))), Image.BICUBIC)

def blur_pil(img: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0: return img
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))

def apply_pipeline(img: Image.Image, jpeg_q=None, scale=None, sigma=None) -> Image.Image:
    out = img
    if scale is not None: out = resize_pil(out, scale)
    if sigma is not None: out = blur_pil(out, sigma)
    if jpeg_q is not None: out = jpeg_compress_pil(out, jpeg_q)
    return out

def grid_augment(img: Image.Image, jpeg_qualities, resize_scales, blur_sigmas):
    for q in jpeg_qualities:
        for s in resize_scales:
            for b in blur_sigmas:
                yield apply_pipeline(img, jpeg_q=q, scale=s, sigma=b)
