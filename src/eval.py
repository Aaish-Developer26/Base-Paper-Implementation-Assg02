import os
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .dataset import list_images, load_image
from .features import CLIPFeaturizer
from .transforms import apply_pipeline
from .detect import cosine, mahalanobis, score_sample  # reuse scorer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def batch_encode(fzr, paths):  # simple cached pass
    feats = []
    for p in tqdm(paths, desc="encode"):
        feats.append(fzr.encode(load_image(p))[0])
    return np.stack(feats).astype("float32")


def stress_eval(cfg, X_bank, y_bank, mu_r, mu_f, invCov, paths_real, paths_fake, stress):
    fzr = CLIPFeaturizer(**cfg["model"])
    ys, scores = [], []

    # safe getters
    k = cfg.get("detect", {}).get("k", 5)
    alpha = cfg.get("detect", {}).get("alpha", 0.5)
    use_maha = cfg.get("detect", {}).get("use_mahalanobis", False)

    for label, paths in [(0, paths_real), (1, paths_fake)]:
        for p in paths:
            im = load_image(p)
            for q in stress["jpeg_qualities"]:
                for s in stress["resize_scales"]:
                    for b in stress["blur_sigmas"]:
                        im_t = apply_pipeline(im, jpeg_q=q, scale=s, sigma=b)
                        x = fzr.encode(im_t)[0]
                        sc = score_sample(
                            x, mu_r, mu_f, X_bank, y_bank, invCov,
                            k=k, alpha=alpha, use_maha=use_maha
                        )
                        ys.append(label)
                        scores.append(sc)

    ys = np.array(ys); scores = np.array(scores)
    if len(np.unique(ys)) < 2:
        raise RuntimeError("Evaluation needs both classes present. "
                           "Ensure there are images in data/test/real and data/test/fake "
                           "or in data/real and data/fake.")

    auroc = roc_auc_score(ys, scores)
    preds = (scores > 0).astype(int)  # default threshold 0
    acc = accuracy_score(ys, preds)
    f1  = f1_score(ys, preds)

    return {"auroc": float(auroc), "accuracy": float(acc), "f1": float(f1)}


def main():
    # read config with utf-8 (Windows-safe)
    with open("src/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cache = np.load(Path(cfg["paths"]["cache_root"]) / "bank.npz", allow_pickle=True)
    X_bank, y_bank = cache["X"], cache["y"]
    mu_r, mu_f, Sigma = cache["mu_r"], cache["mu_f"], cache["Sigma"]

    use_maha = cfg.get("detect", {}).get("use_mahalanobis", False)
    invCov = np.linalg.inv(Sigma) if use_maha else None

    data_root = Path(cfg["paths"]["data_root"])
    real_test = list_images(data_root / "test/real") or list_images(data_root / "real")
    fake_test = list_images(data_root / "test/fake") or list_images(data_root / "fake")

    # minimal stress grid expected to be small/fast per your config
    res = stress_eval(cfg, X_bank, y_bank, mu_r, mu_f, invCov, real_test, fake_test, cfg["eval"]["stress"])
    print(res)

    # quick bar figure
    os.makedirs(cfg["paths"]["figs_root"], exist_ok=True)
    fig_path = Path(cfg["paths"]["figs_root"]) / "summary_bar.png"
    k, v = list(res.keys()), list(res.values())
    plt.figure()
    plt.bar(k, v)
    plt.title("CLIP-NN summary metrics")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
