"""
Original CLIP-kNN baseline evaluation:
- Uses the same stress grid from config for fair comparison
- Evaluates on data/test/... if present, else falls back to data/... (capped by eval.max_per_class)
- No TTA, no Mahalanobis, pure cosine baseline
Saves a bar figure to reports/figures/baseline_summary_bar.png
"""

import os
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .dataset import list_images, load_image
from .features import CLIPFeaturizer


def cosine(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def score_sample_baseline(x, mu_r, mu_f, X, y, k=5, alpha=0.5):
    s_cent = cosine(x, mu_f) - cosine(x, mu_r)
    nbrs = NearestNeighbors(n_neighbors=min(k, len(X)), metric="cosine").fit(X)
    dists, idxs = nbrs.kneighbors(x[None, :], return_distance=True)
    labs = y[idxs[0]]
    df = dists[0][labs == 1].mean() if (labs == 1).any() else 1.0
    dr = dists[0][labs == 0].mean() if (labs == 0).any() else 1.0
    s_knn = (-df) - (-dr)
    return alpha * s_cent + (1 - alpha) * s_knn


def stress_eval(cfg, Xb, yb, mu_r, mu_f, paths_real, paths_fake, stress):
    fzr = CLIPFeaturizer(**cfg["model"])
    det = cfg.get("detect", {})
    k = det.get("k", 5)
    alpha = det.get("alpha", 0.5)

    ys, scores = [], []

    total = (len(paths_real) + len(paths_fake)) \
            * max(1, len(stress["jpeg_qualities"])) \
            * max(1, len(stress["resize_scales"])) \
            * max(1, len(stress["blur_sigmas"]))

    pbar = tqdm(total=total, desc="[baseline] eval")

    # We reuse your transforms.apply_pipeline via a tiny local import to avoid cycles
    from .transforms import apply_pipeline

    for label, paths in [(0, paths_real), (1, paths_fake)]:
        for p in paths:
            im = load_image(p)
            for q in stress["jpeg_qualities"]:
                for s in stress["resize_scales"]:
                    for b in stress["blur_sigmas"]:
                        im_t = apply_pipeline(im, jpeg_q=q, scale=s, sigma=b)
                        x = fzr.encode(im_t)[0]
                        sc = score_sample_baseline(x, mu_r, mu_f, Xb, yb, k=k, alpha=alpha)
                        ys.append(label)
                        scores.append(sc)
                        pbar.update(1)

    pbar.close()
    ys = np.array(ys)
    scores = np.array(scores)
    if len(np.unique(ys)) < 2:
        raise RuntimeError("Evaluation needs both classes present.")

    auroc = roc_auc_score(ys, scores)
    preds = (scores > 0).astype(int)
    acc = accuracy_score(ys, preds)
    f1 = f1_score(ys, preds)

    return {"auroc": float(auroc), "accuracy": float(acc), "f1": float(f1)}


def main():
    with open("src/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load baseline bank (built by baseline_support_bank.py)
    cache = np.load(Path(cfg["paths"]["cache_root"]) / "baseline_bank.npz", allow_pickle=True)
    Xb, yb = cache["X"], cache["y"]
    mu_r, mu_f = cache["mu_r"], cache["mu_f"]

    # Choose eval sets (cap by eval.max_per_class)
    data_root = Path(cfg["paths"]["data_root"])
    real_all = list_images(data_root / "test/real") or list_images(data_root / "real")
    fake_all = list_images(data_root / "test/fake") or list_images(data_root / "fake")
    max_per = int(cfg.get("eval", {}).get("max_per_class", 10))
    real_test = real_all[:max_per]
    fake_test = fake_all[:max_per]

    if len(real_test) == 0 or len(fake_test) == 0:
        raise RuntimeError("No images for evaluation. Ensure data/real and data/fake (or data/test/...) contain images.")

    res = stress_eval(cfg, Xb, yb, mu_r, mu_f, real_test, fake_test, cfg["eval"]["stress"])
    print(res)

    # Figure
    os.makedirs(cfg["paths"]["figs_root"], exist_ok=True)
    fig_path = Path(cfg["paths"]["figs_root"]) / "baseline_summary_bar.png"
    k, v = list(res.keys()), list(res.values())
    plt.figure()
    plt.bar(k, v)
    plt.title("CLIP-kNN Baseline summary metrics")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
