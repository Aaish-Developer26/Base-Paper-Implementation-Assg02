"""
Original CLIP-kNN baseline detector:
- Cosine centroid margin + cosine k-NN density ratio
- No TTA, no Mahalanobis
Usage:
    python -m src.baseline_detect <image_path>
"""
import sys
from pathlib import Path
import numpy as np
import yaml
from sklearn.neighbors import NearestNeighbors

from .dataset import load_image
from .features import CLIPFeaturizer


def cosine(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def score_sample_baseline(x, mu_r, mu_f, X, y, k=5, alpha=0.5):
    # Centroid margin (cosine)
    s_cent = cosine(x, mu_f) - cosine(x, mu_r)

    # kNN density ratio (cosine distance)
    nbrs = NearestNeighbors(n_neighbors=min(k, len(X)), metric="cosine").fit(X)
    dists, idxs = nbrs.kneighbors(x[None, :], return_distance=True)
    labs = y[idxs[0]]
    df = dists[0][labs == 1].mean() if (labs == 1).any() else 1.0
    dr = dists[0][labs == 0].mean() if (labs == 0).any() else 1.0
    s_knn = (-df) - (-dr)

    return alpha * s_cent + (1 - alpha) * s_knn


def main():
    # config
    with open("src/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cache = np.load(Path(cfg["paths"]["cache_root"]) / "baseline_bank.npz", allow_pickle=True)
    X, y = cache["X"], cache["y"]
    mu_r, mu_f = cache["mu_r"], cache["mu_f"]

    if len(sys.argv) < 2:
        print("Usage: python -m src.baseline_detect <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    im = load_image(img_path)
    fzr = CLIPFeaturizer(**cfg["model"])
    x = fzr.encode(im)[0]

    k = cfg.get("detect", {}).get("k", 5)
    alpha = cfg.get("detect", {}).get("alpha", 0.5)
    s = score_sample_baseline(x, mu_r, mu_f, X, y, k=k, alpha=alpha)
    print({"path": img_path, "score": float(s), "pred": int(s > 0.0)})


if __name__ == "__main__":
    main()
