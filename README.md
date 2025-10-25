# Course work - Advance Artificial Intelligence - MS(AI)

# Compression-Aligned CLIP-kNN (Base Paper Implementation – Assg02)

A minimal, CPU-friendly prototype for detecting AI-generated images using CLIP embeddings with simple non-parametric scoring. The prototype emphasizes **robustness under social-platform transforms** (JPEG, down-scaling, mild blur) without training a neural network.

---

## 1) Context & Base Paper (Overview)

**Base idea (CLIP-NN / “nearest-neighbor in CLIP space”)**
Use a pretrained CLIP image encoder to embed images into a semantic feature space and classify **real vs. synthetic** by comparing a test image to a **small labeled support bank** (real/fake) using simple distances (e.g., cosine) or centroids. This approach is attractive because it:

* requires **no gradient training** (fast to stand up, CPU-compatible),
* generalizes across generators better than standard CNNs in many settings,
* is easy to extend with small, transparent tweaks.

> We implement this idea as a practical prototype and **extend** it with a tiny robustness trick described below.

---

## 2) Problem Statement

**Brittleness under “social pipeline” transforms.**
Plain CLIP-kNN often **degrades** when images are **compressed, resized, or lightly blurred** (typical of social platforms). These transforms shift the CLIP features, so nearest-neighbor/centroid decisions become **unstable**, especially with small support sets.

**Goal**
Keep a training-free detector **robust** to common social-platform transforms on a **CPU-only** setup, with clear, reproducible evaluation.

---

## 3) Our Solution: Compression-Aligned CLIP-kNN (CACNN)

**Support-bank alignment (no heavy training):**

1. Build a small **support bank** of labeled real and fake images.
2. For each support image, add **a few lossy variants** (e.g., JPEG {95, 85, 70}, resize {1.0, 0.75}, light blur {0.0, 1.0}) and store their CLIP features as well.
3. Score a test image by combining:

   * **Centroid margin** (fake vs. real centroid) and
   * **kNN density ratio** (k-nearest neighbors among support features),
     optionally with **Mahalanobis** refinement using a shared covariance from the bank.
4. (Optional) **TTA** at test time with 1–2 light transforms and average scores.

**Why it helps**
By “teaching” the support bank the same lossy conditions the test image will face, we **reduce feature drift** caused by compression/resize/blur. The detector remains **training-free**, **explainable**, and **fast** on CPU.

---

## 4) Repository Structure

```
Base Paper Implementation Assg02/
  .venv/                    # your virtualenv (optional)
  data/
    real/                   # real photos (any source)
    fake/                   # AI-generated images (e.g., CIFAKE, DiffusionDB)
    test/real/              # small held-out eval subset (recommended)
    test/fake/
  cache/                    # feature bank (bank.npz)
  reports/figures/          # plots (summary_bar.png)
  src/
    __init__.py
    config.yaml
    dataset.py
    transforms.py
    features.py
    support_bank.py
    detect.py
    eval.py
    viz.py
    utils.py
  run_build_support.sh      # optional (Linux/macOS)
  run_eval.sh               # optional (Linux/macOS)
  README.md
```

---

## 5) Requirements

* Python 3.9+
* CPU-only is fine. (PyTorch CPU wheel will be installed.)
* Disk space for CLIP weights (~600 MB).

### Install (inside your venv)

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install open-clip-torch scikit-learn numpy scipy pillow opencv-python matplotlib umap-learn seaborn tqdm rich pyyaml
```

---

## 6) Configuration (key pieces)

Open `src/config.yaml` and ensure the paths match your layout:

```yaml
model:
  name: "ViT-B-32"
  pretrained: "laion2b_s34b_b79k"
  image_size: 224

paths:
  data_root: "data"
  cache_root: "cache"
  figs_root: "reports/figures"

build_bank:                 # small, fast bank for CPU
  augment:
    jpeg_qualities: [85]    # later: [95, 85, 70] for alignment
    resize_scales: [1.0]    # later: [1.0, 0.75]
    blur_sigmas: [0.0]      # later: [0.0, 1.0]
  max_per_class: 10         # small for speed; can raise to 50–100

detect:
  k: 5
  alpha: 0.5
  use_mahalanobis: false    # turn on (true) after alignment if desired
  tta:
    enable: false

eval:
  # set these for quick demos; you can increase later
  max_per_class: 10
  metrics: ["auroc", "accuracy", "f1"]
  plot_roc: true
  stress:
    jpeg_qualities: [85]    # later: [95, 85, 70]
    resize_scales: [1.0]    # later: [1.0, 0.75]
    blur_sigmas: [0.0]      # later: [0.0, 1.0]
```

> Tip: Put ~10–20 images per class in `data/test/...` so evaluation stays instant. If `data/test` is absent, `eval` falls back to `data/real` and `data/fake`.

---

## 7) Quick Start (Smoke Test)

From the project root (`Base Paper Implementation Assg02`):

```bash
# Build a tiny support bank (downloads CLIP weights on first run)
python -m src.support_bank

# Evaluate on small test sets (prints metrics and saves a bar plot)
python -m src.eval
# -> reports/figures/summary_bar.png
```

**Single-image detection (ad-hoc check)**

```bash
python -m src.detect path\to\your_image.jpg
# prints: {"path": "...", "score": 0.23, "pred": 1}   # 1=fake, 0=real
```

---

## 8) “Robustness” Demonstration (Recommended Flow)

1. **Stress test** (simulate social platforms)
   In `config.yaml`, under `eval.stress`, use:

   ```yaml
   jpeg_qualities: [95, 85, 70]
   resize_scales: [1.0, 0.75]
   blur_sigmas: [0.0, 1.0]
   ```

   Keep `build_bank.augment` simple for this run (e.g., single values).
   Run:

   ```bash
   python -m src.eval
   ```

   You should see a **drop** in AUROC/Accuracy vs. the smoke test.

2. **Cross-generator** (optional but convincing)
   Replace `data/test/fake` with a few **DiffusionDB** images while keeping reals the same. Re-run `python -m src.eval`. Expect further drop (unseen generator + distortions).

3. **Our fix: Compression-Aligned support bank**
   Now align the bank:

   ```yaml
   build_bank:
     augment:
       jpeg_qualities: [95, 85, 70]
       resize_scales:  [1.0, 0.75]
       blur_sigmas:    [0.0]
     max_per_class: 10
   detect:
     use_mahalanobis: true   # optional, often gives a small boost
   ```

   Rebuild the bank and evaluate:

   ```bash
   python -m src.support_bank
   python -m src.eval
   ```

   You should see **metrics recover** vs. the stressed baseline, showing our contribution.

---

## 9) Interpreting Results

* **Smoke test near 1.0** is expected on tiny, same-distribution splits. It proves the pipeline works.
* **Stress & cross-gen** runs reflect real-world conditions (upload pipelines, unseen generators) and should reduce scores.
* **Aligned bank** (and optional **Mahalanobis**) should **recover robustness** without any neural training, validating the proposed idea.

---

## 10) Troubleshooting

* **Long `eval` time**: create `data/test/real` and `data/test/fake` (10–20 images each) or set `eval.max_per_class` in the config.
* **Import errors**: ensure `src/__init__.py` exists and run commands from the project root using `python -m src.*`.
* **Windows encoding error for YAML**: this repo reads config with `encoding="utf-8"`. If you edited the file elsewhere, re-save as UTF-8 in VS Code.
* **First run is slow**: CLIP weights (~600 MB) download once; subsequent runs are fast.

---

## 11) What’s Included vs. Future Work

* **Included**: CLIP feature extraction (OpenCLIP), support-bank building, centroid/kNN/Mahalanobis scoring, stress evaluation, quick plots.
* **Future (nice-to-have)**: richer visualizations (UMAP), per-generator breakdowns, automatic dataset samplers, ROC curves per stress level.

---

## 12) License & Acknowledgements

* CLIP model weights and code via **OpenCLIP** (respect their license).
* Datasets referenced (e.g., CIFAKE, DiffusionDB, COCO/Unsplash) remain under their respective licenses; use only for coursework/demonstration.
* This prototype is for academic coursework (Assg02) and is not a production system.

---


