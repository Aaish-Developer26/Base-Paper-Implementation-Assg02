from pathlib import Path

import yaml
import numpy as np

from .dataset import list_images, load_image
from .transforms import grid_augment
from .features import CLIPFeaturizer


def compute_centroids(X, y):
    mu_real = X[y == 0].mean(axis=0)
    mu_fake = X[y == 1].mean(axis=0)
    return mu_real, mu_fake


def shrinkage_cov(X, eps=1e-3):
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, len(X) - 1)
    cov += eps * np.eye(cov.shape[0])
    return cov


def main():
    # read config with utf-8 (Windows-safe)
    with open("src/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["paths"]["data_root"])
    cache_root = Path(cfg["paths"]["cache_root"])
    cache_root.mkdir(parents=True, exist_ok=True)

    # list and CAP (fast)
    max_per = int(cfg["build_bank"]["max_per_class"])
    real_paths = list_images(data_root / "real")[:max_per]
    fake_paths = list_images(data_root / "fake")[:max_per]

    if len(real_paths) == 0 or len(fake_paths) == 0:
        raise RuntimeError("No images found for one of the classes. "
                           "Ensure data/real and data/fake contain images.")

    # Minimal augmentation grid (ideally single values for speed)
    aug = cfg["build_bank"]["augment"]
    jpeg_qualities = aug.get("jpeg_qualities", [85])
    resize_scales  = aug.get("resize_scales",  [1.0])
    blur_sigmas    = aug.get("blur_sigmas",    [0.0])

    fzr = CLIPFeaturizer(**cfg["model"])

    feats, labels = [], []
    for cls, paths in [(0, real_paths), (1, fake_paths)]:
        for p in paths:
            im = load_image(p)
            for im_t in grid_augment(im, jpeg_qualities, resize_scales, blur_sigmas):
                feats.append(fzr.encode(im_t)[0])
                labels.append(cls)

    X = np.stack(feats).astype("float32")
    y = np.array(labels, dtype=np.int64)

    mu_r, mu_f = compute_centroids(X, y)
    Sigma = shrinkage_cov(X)

    np.savez(cache_root / "bank.npz", X=X, y=y, mu_r=mu_r, mu_f=mu_f, Sigma=Sigma)
    print(f"Bank built: X={X.shape}, real={int((y==0).sum())}, fake={int((y==1).sum())}")


if __name__ == "__main__":
    main()
