"""
Original CLIP-kNN baseline support bank:
- No compression/resize/blur augmentations
- Just a small, balanced subset from data/real and data/fake
Saves: cache/baseline_bank.npz
"""

from pathlib import Path
import numpy as np
import yaml

from .dataset import list_images, load_image
from .features import CLIPFeaturizer


def main():
    # Read config (Windows-safe)
    with open("src/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["paths"]["data_root"])
    cache_root = Path(cfg["paths"]["cache_root"])
    cache_root.mkdir(parents=True, exist_ok=True)

    max_per = int(cfg["build_bank"]["max_per_class"])
    real_paths = list_images(data_root / "real")[:max_per]
    fake_paths = list_images(data_root / "fake")[:max_per]

    if len(real_paths) == 0 or len(fake_paths) == 0:
        raise RuntimeError("No images found for one of the classes. Ensure data/real and data/fake contain images.")

    fzr = CLIPFeaturizer(**cfg["model"])

    feats, labels = [], []
    for p in real_paths:
        feats.append(fzr.encode(load_image(p))[0])
        labels.append(0)
    for p in fake_paths:
        feats.append(fzr.encode(load_image(p))[0])
        labels.append(1)

    X = np.stack(feats).astype("float32")
    y = np.array(labels, dtype=np.int64)

    # Centroids (cosine baseline uses these; no covariance/Mahalanobis)
    mu_r = X[y == 0].mean(axis=0)
    mu_f = X[y == 1].mean(axis=0)

    np.savez(cache_root / "baseline_bank.npz", X=X, y=y, mu_r=mu_r, mu_f=mu_f)
    print(f"[baseline] Bank built: X={X.shape}, real={int((y==0).sum())}, fake={int((y==1).sum())}")


if __name__ == "__main__":
    main()
