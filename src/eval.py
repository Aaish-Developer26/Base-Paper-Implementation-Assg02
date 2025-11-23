"""
Fixed evaluation module with debugging
"""

from pathlib import Path
import yaml
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from .dataset import list_images, load_image
from .transforms import grid_augment
from .features import CLIPFeaturizer
from .detect import score_sample


def load_bank(cache_root, suffix=""):
    """Load support bank from cache"""
    cache_file = Path(cache_root) / f"bank{suffix}.npz"
    if not cache_file.exists():
        raise FileNotFoundError(f"Bank not found: {cache_file}\n"
                              f"Run 'python -m src.support_bank' first.")
    
    cache = np.load(cache_file, allow_pickle=True)
    return {
        'X': cache['X'],
        'y': cache['y'],
        'mu_r': cache['mu_r'],
        'mu_f': cache['mu_f'],
        'Sigma': cache['Sigma']
    }


def evaluate_on_dataset(bank, test_images, fzr, cfg, apply_stress=False):
    """
    Evaluate detector on a set of test images.
    """
    if apply_stress:
        stress = cfg["eval"]["stress"]
        jpeg_q = stress.get("jpeg_qualities", [85])
        resize_s = stress.get("resize_scales", [1.0])
        blur_s = stress.get("blur_sigmas", [0.0])
    else:
        # No stress: original images
        jpeg_q = [100]
        resize_s = [1.0]
        blur_s = [0.0]
    
    X, y = bank['X'], bank['y']
    mu_r, mu_f, Sigma = bank['mu_r'], bank['mu_f'], bank['Sigma']
    
    use_maha = cfg.get("detect", {}).get("use_mahalanobis", False)
    invCov = np.linalg.inv(Sigma) if use_maha else None
    
    k = cfg.get("detect", {}).get("k", 5)
    alpha = cfg.get("detect", {}).get("alpha", 0.5)
    
    scores = []
    true_labels = []
    
    print(f"  Testing {len(test_images)} images...")
    print(f"  Bank size: {len(X)} features ({(y==0).sum()} real, {(y==1).sum()} fake)")
    
    for img_path, true_label in test_images:
        img = load_image(img_path)
        
        # Apply transforms (if stress enabled)
        variants = list(grid_augment(img, jpeg_q, resize_s, blur_s))
        
        # Extract features and compute scores
        variant_scores = []
        for variant in variants:
            feat = fzr.encode(variant)[0]
            score = score_sample(feat, mu_r, mu_f, X, y, invCov, k=k, alpha=alpha, use_maha=use_maha)
            variant_scores.append(score)
        
        # Average scores across variants (TTA)
        avg_score = np.mean(variant_scores)
        scores.append(avg_score)
        true_labels.append(true_label)
    
    scores = np.array(scores)
    true_labels = np.array(true_labels)
    
    # DEBUG: Print score distribution
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  Real images avg score: {scores[true_labels==0].mean():.3f}")
    print(f"  Fake images avg score: {scores[true_labels==1].mean():.3f}")
    
    # Find optimal threshold using ROC
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    
    # Optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    
    # Compute metrics with optimal threshold
    auroc = roc_auc_score(true_labels, scores)
    preds = (scores > optimal_threshold).astype(int)
    acc = accuracy_score(true_labels, preds)
    
    # Fix F1 score calculation (handle edge case)
    if preds.sum() == 0 or preds.sum() == len(preds):
        f1 = 0.0
        print("  WARNING: All predictions are same class!")
    else:
        f1 = f1_score(true_labels, preds, zero_division=0)
    
    return {
        'auroc': auroc,
        'accuracy': acc,
        'f1': f1,
        'scores': scores,
        'labels': true_labels,
        'threshold': optimal_threshold
    }


def load_test_data(data_root, max_per_class=None):
    """Load test images - prefer stressed version if available"""
    
    # Priority: test_stress > test_extreme > test > fallback
    for test_dir_name in ['test_stress', 'test_extreme', 'test']:
        test_root = Path(data_root) / test_dir_name
        if test_root.exists():
            print(f"Using test set: {test_dir_name}/")
            real_paths = list(list_images(test_root / "real"))
            fake_paths = list(list_images(test_root / "fake"))
            break
    else:
        # Fallback to data/real and data/fake
        print("Warning: No test folder found, using data/real and data/fake")
        real_paths = list(list_images(Path(data_root) / "real"))
        fake_paths = list(list_images(Path(data_root) / "fake"))
    
    if max_per_class:
        real_paths = real_paths[:max_per_class]
        fake_paths = fake_paths[:max_per_class]
    
    test_images = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    
    return test_images


def print_results(results, method_name=""):
    """Pretty print evaluation results"""
    if method_name:
        print(f"\n{method_name}:")
    print(f"  AUROC:    {results['auroc']:.3f}")
    print(f"  Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    print(f"  F1 Score: {results['f1']:.3f}")


def print_comparison_table(results_dict):
    """Print comparison table between methods"""
    print("\n" + "=" * 90)
    print("PERFORMANCE COMPARISON")
    print("=" * 90)
    print(f"{'Scenario':<20} {'Metric':<12} {'Vanilla':<15} {'CACNN':<15} {'Improvement':<15}")
    print("-" * 90)
    
    scenarios = [k for k in results_dict.keys() if k.startswith('vanilla_')]
    
    for scenario in scenarios:
        scenario_name = scenario.replace('vanilla_', '').replace('_', ' ').title()
        
        vanilla_key = scenario
        cacnn_key = scenario.replace('vanilla_', 'cacnn_')
        
        if cacnn_key in results_dict:
            v_res = results_dict[vanilla_key]
            c_res = results_dict[cacnn_key]
            
            # Print AUROC
            v_auroc = v_res['auroc']
            c_auroc = c_res['auroc']
            auroc_imp = ((c_auroc - v_auroc) / v_auroc) * 100 if v_auroc > 0 else 0
            print(f"{scenario_name:<20} {'AUROC':<12} {v_auroc:.3f}         {c_auroc:.3f}         "
                  f"{'+' if auroc_imp > 0 else ''}{auroc_imp:.1f}%")
            
            # Print Accuracy
            v_acc = v_res['accuracy']
            c_acc = c_res['accuracy']
            acc_imp = ((c_acc - v_acc) / v_acc) * 100 if v_acc > 0 else 0
            print(f"{'':<20} {'Accuracy':<12} {v_acc:.3f}         {c_acc:.3f}         "
                  f"{'+' if acc_imp > 0 else ''}{acc_imp:.1f}%")
            
            # Print F1
            v_f1 = v_res['f1']
            c_f1 = c_res['f1']
            f1_imp = ((c_f1 - v_f1) / v_f1) * 100 if v_f1 > 0 else float('inf')
            f1_imp_str = f"+∞" if f1_imp == float('inf') else f"{'+' if f1_imp > 0 else ''}{f1_imp:.1f}%"
            print(f"{'':<20} {'F1 Score':<12} {v_f1:.3f}         {c_f1:.3f}         {f1_imp_str}")
            print()
    
    print("=" * 90)



def main():
    """Main evaluation function with comparison support"""
    
    # Load config
    with open("src/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Load test data
    max_per = cfg["eval"].get("max_per_class", None)
    test_images = load_test_data(cfg["paths"]["data_root"], max_per)
    
    print(f"Loaded {len(test_images)} test images "
          f"({sum(1 for _, l in test_images if l==0)} real, "
          f"{sum(1 for _, l in test_images if l==1)} fake)")
    
    # Initialize feature extractor
    fzr = CLIPFeaturizer(**cfg["model"])
    
    # Check if comparison mode
    run_comparison = cfg.get("experiment", {}).get("run_comparison", False)
    
    if run_comparison:
        print("\n" + "=" * 80)
        print("RUNNING COMPARISON: VANILLA vs CACNN")
        print("=" * 80)
        
        # Load both banks
        cache_root = cfg["paths"]["cache_root"]
        try:
            vanilla_bank = load_bank(cache_root, suffix="_vanilla")
            cacnn_bank = load_bank(cache_root, suffix="_cacnn")
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("\nPlease run: python -m src.support_bank")
            print("(Make sure run_comparison: true in config.yaml)")
            return
        
        results = {}
        
        # 1. Clean images
        print("\n[1/4] Evaluating on CLEAN images...")
        print("\n  Vanilla CLIP-kNN:")
        results['vanilla_clean'] = evaluate_on_dataset(
            vanilla_bank, test_images, fzr, cfg, apply_stress=False
        )
        print_results(results['vanilla_clean'], "Vanilla")
        
        print("\n  CACNN:")
        results['cacnn_clean'] = evaluate_on_dataset(
            cacnn_bank, test_images, fzr, cfg, apply_stress=False
        )
        print_results(results['cacnn_clean'], "CACNN")
        
        # 2. Stress test
        print("\n[2/4] Evaluating on STRESSED images (compressed/resized)...")
        print("\n  Vanilla CLIP-kNN:")
        results['vanilla_stress'] = evaluate_on_dataset(
            vanilla_bank, test_images, fzr, cfg, apply_stress=True
        )
        print_results(results['vanilla_stress'], "Vanilla")
        
        print("\n  CACNN:")
        results['cacnn_stress'] = evaluate_on_dataset(
            cacnn_bank, test_images, fzr, cfg, apply_stress=True
        )
        print_results(results['cacnn_stress'], "CACNN")
        
        # Print comparison table
        print_comparison_table(results)
        
        # Save results
        results_file = Path(cfg["paths"]["cache_root"]) / "comparison_results.npz"
        np.savez(results_file, **{k: v for k, v in results.items()})
        print(f"\n✓ Results saved to: {results_file}")
        
    else:
        # Normal single-method evaluation
        print("\n" + "=" * 80)
        print("EVALUATING CACNN (SINGLE MODE)")
        print("=" * 80)
        print("Note: To compare vanilla vs CACNN, set 'experiment.run_comparison: true'")
        print("=" * 80)
        
        # Load bank
        bank = load_bank(cfg["paths"]["cache_root"], suffix="")
        
        # Evaluate on clean
        print("\n[1/2] Clean images:")
        results_clean = evaluate_on_dataset(bank, test_images, fzr, cfg, apply_stress=False)
        print_results(results_clean)
        
        # Evaluate on stress
        print("\n[2/2] Stress test:")
        results_stress = evaluate_on_dataset(bank, test_images, fzr, cfg, apply_stress=True)
        print_results(results_stress)


if __name__ == "__main__":
    main()