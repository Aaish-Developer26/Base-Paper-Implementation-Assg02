import sys
from pathlib import Path

import yaml
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .dataset import load_image
from .features import CLIPFeaturizer
from .transforms import apply_pipeline


def cosine(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def mahalanobis(x, mu, invCov):
    d = x - mu
    return float(d @ invCov @ d)


def score_sample(x, mu_r, mu_f, X, y, invCov, k=5, alpha=0.5, use_maha=True):
    # centroid margin
    if use_maha and invCov is not None:
        s_cent = -(mahalanobis(x, mu_f, invCov)) + (mahalanobis(x, mu_r, invCov))
    else:
        s_cent = cosine(x, mu_f) - cosine(x, mu_r)

    # kNN density ratio
    nbrs = NearestNeighbors(n_neighbors=min(k, len(X)), metric="cosine").fit(X)
    dists, idxs = nbrs.kneighbors(x[None, :], return_distance=True)
    labs = y[idxs[0]]
    df = dists[0][labs == 1].mean() if (labs == 1).any() else 1.0
    dr = dists[0][labs == 0].mean() if (labs == 0).any() else 1.0
    s_knn = (-df) - (-dr)

    return alpha * s_cent + (1 - alpha) * s_knn


def main():
    # read config with utf-8 (Windows-safe)
    with open("src/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cache = np.load(Path(cfg["paths"]["cache_root"]) / "bank.npz", allow_pickle=True)
    X, y = cache["X"], cache["y"]
    mu_r, mu_f, Sigma = cache["mu_r"], cache["mu_f"], cache["Sigma"]
    use_maha = cfg.get("detect", {}).get("use_mahalanobis", False)
    invCov = np.linalg.inv(Sigma) if use_maha else None

    fzr = CLIPFeaturizer(**cfg["model"])

    # example usage: run on a single image path
    if len(sys.argv) < 2:
        print("Usage: python -m src.detect <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    im = load_image(img_path)

    # TTA
    tta_cfg = cfg["detect"].get("tta", {"enable": False})
    feats = []
    if tta_cfg.get("enable", False):
        for q in tta_cfg.get("jpeg_qualities", [85]):
            for s in tta_cfg.get("resize_scales", [1.0]):
                for b in tta_cfg.get("blur_sigmas", [0.0]):
                    im_t = apply_pipeline(im, jpeg_q=q, scale=s, sigma=b)
                    feats.append(fzr.encode(im_t)[0])
    else:
        feats.append(fzr.encode(im)[0])

    k = cfg.get("detect", {}).get("k", 5)
    alpha = cfg.get("detect", {}).get("alpha", 0.5)
    scores = [
        score_sample(f, mu_r, mu_f, X, y, invCov, k=k, alpha=alpha, use_maha=use_maha)
        for f in feats
    ]
    s = float(np.mean(scores))
    print({"path": img_path, "score": s, "pred": int(s > 0.0)})


if __name__ == "__main__":
    main()
