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


def build_bank(cfg, use_augmentation=True, cache_suffix=""):
    """
    Build support bank with optional augmentation.
    
    Args:
        cfg: Configuration dictionary
        use_augmentation: If False, only use original images (vanilla baseline)
        cache_suffix: Suffix for cache file (e.g., "_vanilla" or "_cacnn")
    
    Returns:
        dict: Bank data with features, labels, centroids, covariance
    """
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

    # Get augmentation parameters
    aug = cfg["build_bank"]["augment"]
    
    if use_augmentation:
        # CACNN: Use augmentation grid
        jpeg_qualities = aug.get("jpeg_qualities", [85])
        resize_scales  = aug.get("resize_scales",  [1.0])
        blur_sigmas    = aug.get("blur_sigmas",    [0.0])
        method_name = "CACNN (with augmentation)"
    else:
        # Vanilla: Only original images (no augmentation)
        jpeg_qualities = [100]  # Original quality
        resize_scales  = [1.0]  # Original size
        blur_sigmas    = [0.0]  # No blur
        method_name = "Vanilla CLIP-kNN (no augmentation)"

    print(f"\nBuilding {method_name}...")
    print(f"  Real images: {len(real_paths)}")
    print(f"  Fake images: {len(fake_paths)}")
    
    if use_augmentation:
        print(f"  Augmentations: JPEG {jpeg_qualities}, Resize {resize_scales}, Blur {blur_sigmas}")
    else:
        print(f"  No augmentation (baseline)")

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

    # Save with appropriate suffix
    cache_file = cache_root / f"bank{cache_suffix}.npz"
    np.savez(cache_file, X=X, y=y, mu_r=mu_r, mu_f=mu_f, Sigma=Sigma)
    
    print(f"✓ Bank built: {X.shape[0]} features ({int((y==0).sum())} real, {int((y==1).sum())} fake)")
    print(f"  Saved to: {cache_file}\n")
    
    return {
        'X': X, 
        'y': y, 
        'mu_r': mu_r, 
        'mu_f': mu_f, 
        'Sigma': Sigma,
        'cache_file': cache_file
    }


def main():
    """
    Build both vanilla and CACNN support banks for comparison.
    """
    # read config with utf-8 (Windows-safe)
    with open("src/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Check if comparison mode is enabled
    run_comparison = cfg.get("experiment", {}).get("run_comparison", False)
    
    if run_comparison:
        print("=" * 70)
        print("BUILDING BOTH VANILLA AND CACNN BANKS FOR COMPARISON")
        print("=" * 70)
        
        # Build vanilla baseline (no augmentation)
        vanilla_bank = build_bank(cfg, use_augmentation=False, cache_suffix="_vanilla")
        
        # Build CACNN (with augmentation)
        cacnn_bank = build_bank(cfg, use_augmentation=True, cache_suffix="_cacnn")
        
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Vanilla bank: {vanilla_bank['X'].shape[0]} features")
        print(f"CACNN bank:   {cacnn_bank['X'].shape[0]} features")
        print(f"Augmentation factor: {cacnn_bank['X'].shape[0] / vanilla_bank['X'].shape[0]:.1f}x")
        print("=" * 70)
        
    else:
        # Normal mode: only build CACNN
        print("=" * 70)
        print("BUILDING CACNN SUPPORT BANK (SINGLE MODE)")
        print("=" * 70)
        print("Note: To build both vanilla and CACNN for comparison,")
        print("      set 'experiment.run_comparison: true' in config.yaml")
        print("=" * 70)
        
        build_bank(cfg, use_augmentation=True, cache_suffix="")


if __name__ == "__main__":
    main()