import sys
from pathlib import Path

import yaml
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .dataset import load_image
from .features import CLIPFeaturizer
from .transforms import apply_pipeline


def cosine(a, b):
    """Cosine similarity between two vectors"""
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def mahalanobis(x, mu, invCov):
    """Mahalanobis distance from x to mu"""
    d = x - mu
    return float(d @ invCov @ d)


def score_sample(x, mu_r, mu_f, X, y, invCov, k=5, alpha=0.5, use_maha=True):
    """
    Compute detection score for a single feature vector.
    
    Args:
        x: Feature vector to score
        mu_r: Real class centroid
        mu_f: Fake class centroid
        X: Support bank features
        y: Support bank labels
        invCov: Inverse covariance matrix (for Mahalanobis)
        k: Number of nearest neighbors
        alpha: Weight between centroid and kNN scores
        use_maha: Use Mahalanobis distance instead of cosine
    
    Returns:
        float: Detection score (positive = fake, negative = real)
    """
    # Centroid margin
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


def combined_score(feat, X, y, mu_r, mu_f, Sigma, k=5, alpha=0.5, use_mahal=False):
    """
    Wrapper function for score_sample (used by eval.py).
    
    Args:
        feat: Feature vector
        X: Support bank features
        y: Support bank labels
        mu_r: Real centroid
        mu_f: Fake centroid
        Sigma: Covariance matrix
        k: Number of neighbors
        alpha: Centroid weight
        use_mahal: Use Mahalanobis distance
    
    Returns:
        float: Detection score
    """
    invCov = np.linalg.inv(Sigma) if use_mahal else None
    return score_sample(feat, mu_r, mu_f, X, y, invCov, k=k, alpha=alpha, use_maha=use_mahal)


def load_bank(cache_root, bank_type="auto"):
    """
    Load support bank from cache.
    
    Args:
        cache_root: Path to cache directory
        bank_type: Which bank to load:
            - "auto": Load bank.npz, or bank_cacnn.npz if exists
            - "vanilla": Load bank_vanilla.npz
            - "cacnn": Load bank_cacnn.npz
    
    Returns:
        dict: Bank data (X, y, mu_r, mu_f, Sigma)
    """
    cache_root = Path(cache_root)
    
    if bank_type == "vanilla":
        cache_file = cache_root / "bank_vanilla.npz"
    elif bank_type == "cacnn":
        cache_file = cache_root / "bank_cacnn.npz"
    else:  # auto
        # Try default first
        cache_file = cache_root / "bank.npz"
        if not cache_file.exists():
            # Try CACNN
            cache_file = cache_root / "bank_cacnn.npz"
            if not cache_file.exists():
                raise FileNotFoundError(
                    f"No bank found in {cache_root}.\n"
                    f"Run 'python -m src.support_bank' first."
                )
    
    if not cache_file.exists():
        raise FileNotFoundError(f"Bank not found: {cache_file}")
    
    cache = np.load(cache_file, allow_pickle=True)
    return {
        'X': cache['X'],
        'y': cache['y'],
        'mu_r': cache['mu_r'],
        'mu_f': cache['mu_f'],
        'Sigma': cache['Sigma'],
        'file': cache_file
    }


def detect_image(img_path, cfg, bank_type="auto", verbose=True):
    """
    Detect if an image is real or fake.
    
    Args:
        img_path: Path to image file
        cfg: Configuration dictionary
        bank_type: Which bank to use ("auto", "vanilla", "cacnn")
        verbose: Print detection details
    
    Returns:
        dict: Detection results
    """
    # Load bank
    bank = load_bank(cfg["paths"]["cache_root"], bank_type)
    
    X, y = bank['X'], bank['y']
    mu_r, mu_f, Sigma = bank['mu_r'], bank['mu_f'], bank['Sigma']
    
    use_maha = cfg.get("detect", {}).get("use_mahalanobis", False)
    invCov = np.linalg.inv(Sigma) if use_maha else None
    
    # Initialize feature extractor
    fzr = CLIPFeaturizer(**cfg["model"])
    
    # Load image
    im = load_image(img_path)
    
    # TTA (Test-Time Augmentation)
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
    
    # Detection parameters
    k = cfg.get("detect", {}).get("k", 5)
    alpha = cfg.get("detect", {}).get("alpha", 0.5)
    
    # Compute scores
    scores = [
        score_sample(f, mu_r, mu_f, X, y, invCov, k=k, alpha=alpha, use_maha=use_maha)
        for f in feats
    ]
    s = float(np.mean(scores))
    pred = int(s > 0.0)
    
    result = {
        'path': str(img_path),
        'score': round(s, 4),
        'prediction': 'FAKE' if pred == 1 else 'REAL',
        'pred': pred,
        'bank_used': str(bank['file'].name),
        'confidence': round(abs(s), 4)
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Image: {img_path}")
        print(f"Bank:  {bank['file'].name}")
        print(f"{'='*60}")
        print(f"Score:      {result['score']:>8.4f}  {'(positive = fake)':>20}")
        print(f"Prediction: {result['prediction']:>8}  {'':>20}")
        print(f"Confidence: {result['confidence']:>8.4f}  {'(higher = more certain)':>20}")
        print(f"{'='*60}\n")
    
    return result


def main():
    """Main function for command-line usage"""
    
    # Read config
    with open("src/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python -m src.detect <image_path> [--bank vanilla|cacnn]")
        print("\nExamples:")
        print("  python -m src.detect sample.jpg")
        print("  python -m src.detect sample.jpg --bank vanilla")
        print("  python -m src.detect sample.jpg --bank cacnn")
        sys.exit(1)
    
    img_path = sys.argv[1]
    
    # Check for bank selection
    bank_type = "auto"
    if len(sys.argv) >= 4 and sys.argv[2] == "--bank":
        bank_type = sys.argv[3]
        if bank_type not in ["vanilla", "cacnn", "auto"]:
            print(f"Error: Invalid bank type '{bank_type}'")
            print("Valid options: vanilla, cacnn, auto")
            sys.exit(1)
    
    # Check if comparison mode to show both
    run_comparison = cfg.get("experiment", {}).get("run_comparison", False)
    
    if run_comparison and bank_type == "auto":
        # Show results from both banks for comparison
        print("\n" + "="*70)
        print("COMPARISON MODE: Testing with both Vanilla and CACNN banks")
        print("="*70)
        
        try:
            # Test with vanilla
            print("\n[1/2] Testing with VANILLA CLIP-kNN bank...")
            result_vanilla = detect_image(img_path, cfg, bank_type="vanilla", verbose=True)
            
            # Test with CACNN
            print("\n[2/2] Testing with CACNN bank...")
            result_cacnn = detect_image(img_path, cfg, bank_type="cacnn", verbose=True)
            
            # Show comparison
            print("\n" + "="*70)
            print("COMPARISON SUMMARY")
            print("="*70)
            print(f"{'Method':<20} {'Score':<12} {'Prediction':<12} {'Confidence':<12}")
            print("-"*70)
            print(f"{'Vanilla CLIP-kNN':<20} {result_vanilla['score']:<12.4f} "
                  f"{result_vanilla['prediction']:<12} {result_vanilla['confidence']:<12.4f}")
            print(f"{'CACNN (Ours)':<20} {result_cacnn['score']:<12.4f} "
                  f"{result_cacnn['prediction']:<12} {result_cacnn['confidence']:<12.4f}")
            print("="*70)
            
            # Agreement check
            if result_vanilla['pred'] == result_cacnn['pred']:
                print("✓ Both methods AGREE on the prediction")
            else:
                print("⚠ Methods DISAGREE - vanilla may be affected by compression")
            print()
            
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("\nTo enable comparison mode:")
            print("1. Set 'run_comparison: true' in config.yaml")
            print("2. Run: python -m src.support_bank")
            print("3. Then try again")
            sys.exit(1)
    else:
        # Normal single bank detection
        try:
            result = detect_image(img_path, cfg, bank_type=bank_type, verbose=True)
            
            # Also print JSON for scripting
            print("JSON output:")
            import json
            print(json.dumps({
                'path': result['path'],
                'score': result['score'],
                'pred': result['pred']
            }))
            
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("\nRun 'python -m src.support_bank' first to build the support bank.")
            sys.exit(1)


if __name__ == "__main__":
    main()