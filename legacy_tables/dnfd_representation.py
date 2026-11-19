"""
DNFD Linear Probing + HSIC Information-Theoretic Analysis
==========================================================

Research question (Reviewer 3 of JBHI revision)
------------------------------------------------
Does DNFD separate tumor-discriminative information from nuisance
information into different feature spaces?

Method
------
We take the already-trained Full TopoFormer (frozen), extract
``f_disc`` (256-d) and ``f_indisc`` (256-d) for all DeepHisto
train+val+test samples, and run two complementary analyses.

  (A) Linear Probe  — trains a multinomial logistic regression
      on each feature space to predict the actual 5-class tumor
      label.  Higher accuracy on f_disc means f_disc is more
      aligned with tumor information.

  (B) HSIC  (Hilbert-Schmidt Independence Criterion)  —
      a non-parametric, model-free measure of statistical
      dependence between two random variables.

        HSIC(f, y)  =  || C_FF^1/2 K_yy C_FF^1/2 ||^2_F  / (n-1)^2

      where K is a Gaussian RBF kernel on f, L is a delta-kernel
      on the one-hot encoded y, and C is the centering matrix.

      HSIC = 0  →  f and y are independent
      HSIC > 0  →  some dependence; the larger, the stronger.

      We compute HSIC(f_disc, y_tumor) and HSIC(f_indisc, y_tumor)
      on the test set.  If HSIC(f_disc) ≫ HSIC(f_indisc), the
      claim that "DNFD concentrates tumor information into f_disc"
      is supported by an information-theoretic argument rather
      than just a downstream classifier.

Output
------
  - Per-seed CSV: ``linerprobe_seed{SEED}.csv``
  - Summary CSV:  ``linerprobe_summary.csv``  (mean ± SD, 3 seeds)
  - This script:  ``train_briancancer_rgb_200_seed42_MI.py``

No model weights are modified.  All features are loaded from existing
NPZ caches produced by ``train_briancancer_rgb_200_seed42.py``.
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import sys
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


# ── Config ───────────────────────────────────────────────────────────────────
# Per-seed cache roots (Full TopoFormer runs only).
CACHE_DIRS = {
    42:  "runs/features_cache_briancancer_dinov3FD_fusion_RGB_200_seed42",
    123: "runs/features_cache_briancancer_dinov3FD_fusion_RGB_200_seed123",
    666: "runs/features_cache_briancancer_dinov3FD_fusion_RGB_200_seed666",
}

# DeepHisto data splits (absolute path, same as split_builders.py).
BRIANCANCER_TRAIN_N = 34689   # total images in train_dir
BRIANCANCER_TEST_N  = 4174    # total images in test_dir

# Tumor class names (sorted alphabetically as in split_builders.py).
TUMOR_CLASSES = ["astro", "gbm", "necro", "normal", "oligo"]   # id 0-4

# Global random state used for the train/val split (must match split_builders.py).
_SPLIT_RS = 42
_SPLIT_TEST_SIZE = 0.1   # 10 % of original train_dir → validation

# Probe hyperparameter grid (selected on val, evaluated on test).
_PROBE_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]

# Output directory (relative to script dir, or absolute).
OUT_DIR = "runs/linerprobe"


# ── HSIC: Hilbert-Schmidt Independence Criterion ────────────────────────────

def _rbf_kernel_matrix(X: np.ndarray, sigma: float = None) -> np.ndarray:
    """
    Gaussian RBF kernel matrix for the row matrix X (n × d).
    Uses scipy.spatial.distance.cdist (compiled C) for the heavy
    O(n^2 d) distance computation, then a vectorised numpy exp.

    Bandwidth sigma: if None, use the median heuristic on the
    upper-triangular squared distances.
    """
    from scipy.spatial.distance import cdist
    X = X.astype(np.float32, copy=False)
    # Squared Euclidean distance via cdist (much faster than the
    # hand-rolled X X^T - diag - diag path we had before).
    D2 = cdist(X, X, metric="sqeuclidean").astype(np.float32)
    if sigma is None:
        d_flat = D2[np.triu_indices_from(D2, k=1)]
        if d_flat.size == 0:
            sigma = 1.0
        else:
            sigma = float(np.sqrt(np.median(d_flat)))
            if sigma <= 1e-12:
                sigma = 1.0
    K = np.exp(-D2 / (2.0 * sigma * sigma)).astype(np.float32)
    return K


def _delta_kernel_matrix(y_int: np.ndarray) -> np.ndarray:
    """
    Class-indicator kernel for an integer label vector y_int.
    L[i, j] = 1 if y[i] == y[j] else 0.
    """
    y_int = y_int.astype(np.int64)
    return (y_int[:, None] == y_int[None, :]).astype(np.float64)


def _center(K: np.ndarray) -> np.ndarray:
    """Double-center a kernel matrix: K_c = H K H, where H = I - 1/n 11^T."""
    n = K.shape[0]
    one = np.ones((n, n), dtype=np.float64) / n
    return K - one @ K - K @ one + one @ K @ one


def hsic_statistic_from_K(K_centered: np.ndarray, y_int: np.ndarray) -> float:
    """
    HSIC statistic given an already-centered kernel matrix K_c on
    the feature side and a label vector.  Computes
        HSIC_n = (1 / n(n-3)) * tr( K_c L_c )
    where L_c is the centered delta kernel on y_int.

    This is the fast path used inside the permutation test,
    because the (expensive) feature-kernel matrix K is computed
    only once.
    """
    n = K_centered.shape[0]
    if n < 4:
        return 0.0
    L  = _delta_kernel_matrix(y_int)
    Lc = _center(L)
    val = float(np.trace(K_centered @ Lc) / (n * (n - 3)))
    return val


def hsic_statistic(X_feat: np.ndarray, y_int: np.ndarray,
                   sigma: float = None) -> float:
    """Empirical HSIC estimator with RBF kernel on X_feat and
    delta kernel on y_int (Gretton et al., ALT 2005)."""
    K = _rbf_kernel_matrix(X_feat, sigma=sigma)
    Kc = _center(K)
    return hsic_statistic_from_K(Kc, y_int)


def hsic_permutation_test(X_feat: np.ndarray, y_int: np.ndarray,
                          n_perm: int = 200, seed: int = 0) -> tuple:
    """
    Compute HSIC and a permutation p-value (null = independence).
    Returns (hsic, p_value, hsic_null_mean, hsic_null_std).

    Optimisation: the centred RBF kernel on X_feat is computed
    ONCE; the permutation test only re-permutes the cheap
    delta-kernel side, so the loop is dominated by O(n^2) work.
    """
    rng = np.random.default_rng(seed)
    n = X_feat.shape[0]

    # Heavy work: RBF kernel + centering (done once)
    K = _rbf_kernel_matrix(X_feat, sigma=None)
    Kc = _center(K)

    # Observe HSIC
    obs = hsic_statistic_from_K(Kc, y_int)

    # Permutation null (cheap, no kernel recomputation)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(n)
        null[i] = hsic_statistic_from_K(Kc, y_int[perm])

    p = (1.0 + np.sum(null >= obs)) / (n_perm + 1.0)
    return obs, float(p), float(null.mean()), float(null.std())


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_npz_features(cache_dir: str):
    """Load disc + indisc + y + paths from a pair of NPZ files."""
    train_path = os.path.join(cache_dir, "dinov3FD_briancancer_train.npz")
    test_path  = os.path.join(cache_dir, "dinov3FD_briancancer_test.npz")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"NPZ files not found in {cache_dir}.\n"
            f"  Expected: {train_path}\n  Expected: {test_path}"
        )

    train = np.load(train_path, allow_pickle=True)
    test  = np.load(test_path,  allow_pickle=True)

    disc   = np.concatenate([train["disc"],   test["disc"]],   axis=0).astype(np.float32)
    indisc = np.concatenate([train["indisc"], test["indisc"]], axis=0).astype(np.float32)
    y      = np.concatenate([train["y"],       test["y"]],       axis=0).astype(np.int64)
    paths  = list(train["paths"]) + list(test["paths"])

    n_train = train["y"].shape[0]
    n_test  = test["y"].shape[0]
    n_total = n_train + n_test

    assert disc.shape[0] == n_total
    assert indisc.shape[0] == n_total
    assert y.shape[0] == n_total

    return disc, indisc, y, paths, n_train, n_test


def strat_train_val_split(n_train: int, y: np.ndarray, *, random_state: int = 42,
                          test_size: float = 0.1):
    """Re-create the train/val split from split_builders.py."""
    y_train_portion = y[:n_train]
    all_train_idx = np.arange(n_train, dtype=np.int64)

    tr_idx, va_idx = train_test_split(
        all_train_idx,
        test_size=test_size,
        stratify=y_train_portion,
        random_state=random_state,
    )
    train_mask = np.zeros(n_train, dtype=bool)
    val_mask   = np.zeros(n_train, dtype=bool)
    train_mask[tr_idx] = True
    val_mask[va_idx]   = True
    return train_mask, val_mask


def build_probe(X_train, y_train, X_val, y_val, *, C_values=_PROBE_C_VALUES):
    """StandardScaler → LogisticRegression, C selected by val BAcc."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)

    best_c, best_bacc, best_model = None, -1.0, None

    for C in C_values:
        model = LogisticRegression(
            C=C,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        model.fit(X_tr_s, y_train)
        va_pred = model.predict(X_va_s)
        bacc = balanced_accuracy_score(y_val, va_pred)
        if bacc > best_bacc:
            best_bacc = bacc
            best_c    = C
            best_model = model

    X_full_s = scaler.fit_transform(np.concatenate([X_train, X_val], axis=0))
    y_full   = np.concatenate([y_train, y_val], axis=0)
    final_model = LogisticRegression(
        C=best_c,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    final_model.fit(X_full_s, y_full)
    final_model._scaler = scaler
    return final_model, scaler, best_c


def evaluate_probe(model, scaler, X_test, y_test, task: str):
    """Return dict of metrics on the test set."""
    X_s = scaler.transform(X_test)
    y_pred = model.predict(X_s)

    acc   = accuracy_score(y_test, y_pred)
    bacc  = balanced_accuracy_score(y_test, y_pred)
    p, r, f1_arr, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    macro_f1 = float(np.mean(f1_arr))

    if task == "binary" or len(np.unique(y_test)) == 2:
        try:
            proba = model.predict_proba(X_s)
            auroc = roc_auc_score(y_test, proba[:, 1])
        except Exception:
            auroc = float("nan")
    else:
        try:
            proba = model.predict_proba(X_s)
            n_cls = proba.shape[1]
            y_onehot = np.eye(n_cls)[y_test]
            auroc = roc_auc_score(y_onehot, proba, multi_class="ovr", average="macro")
        except Exception:
            auroc = float("nan")

    return {
        "accuracy": acc,
        "bal_accuracy": bacc,
        "macro_f1": macro_f1,
        "auroc": auroc,
    }


# ── Main per-seed runner ─────────────────────────────────────────────────────

def run_single_seed(seed: int, cache_dir: str, out_dir: str,
                    n_perm: int = 200):
    """Run the full LP + HSIC pipeline for one seed."""
    print(f"\n{'='*70}")
    print(f"  SEED {seed}  |  cache: {cache_dir}")
    print(f"{'='*70}")

    disc, indisc, y_all, paths_all, n_train, n_test = load_npz_features(cache_dir)
    print(f"  Loaded: disc/indisc {disc.shape}, y {y_all.shape}")
    print(f"  n_train (train+val)={n_train}, n_test={n_test}")

    y_tumor = y_all

    train_mask, val_mask = strat_train_val_split(
        n_train, y_tumor, random_state=_SPLIT_RS, test_size=_SPLIT_TEST_SIZE,
    )
    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]
    test_idx  = np.arange(n_train, n_train + n_test, dtype=np.int64)

    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    results = {}

    # ── (A) Linear probe on the tumor (5-class) task ─────────────────────────
    for feat_name, X_all in [("f_disc", disc), ("f_indisc", indisc)]:
        X_tr = X_all[train_idx]
        X_va = X_all[val_idx]
        X_te = X_all[test_idx]
        y_tr = y_tumor[train_idx]
        y_va = y_tumor[val_idx]
        y_te = y_tumor[test_idx]

        model, scaler, best_c = build_probe(X_tr, y_tr, X_va, y_va)
        print(f"  [LP-Tumor] {feat_name}: best_C={best_c}")

        metrics = evaluate_probe(model, scaler, X_te, y_te, task="multiclass")
        metrics["best_C"] = best_c
        results[f"tumor_{feat_name}"] = metrics

        for k, v in metrics.items():
            if k != "best_C":
                print(f"    {k:>15s} = {v:.4f}")

    # ── (B) HSIC on the test set (information-theoretic measure) ─────────────
    # Use a balanced stratified subsample of the test set.  HSIC is
    # a U-statistic that converges in n very quickly: a balanced
    # ~1500-point sub-sample gives the same statistical conclusion
    # (both p-values, both the relative ordering of HSIC values)
    # while keeping the O(n^2) kernel computation tractable.
    HSIC_SUBSAMPLE = 1500
    rng = np.random.default_rng(seed)
    sub_idx_list = []
    y_te_full = y_tumor[test_idx]
    for c in np.unique(y_te_full):
        cls_idx = np.where(y_te_full == c)[0]
        take = min(len(cls_idx), HSIC_SUBSAMPLE // len(np.unique(y_te_full)))
        sub_idx_list.append(rng.choice(cls_idx, size=take, replace=False))
    sub_idx = np.concatenate(sub_idx_list)
    rng.shuffle(sub_idx)

    X_disc_sub = disc[test_idx][sub_idx]
    X_ind_sub  = indisc[test_idx][sub_idx]
    y_sub      = y_te_full[sub_idx]

    print(f"\n  Computing HSIC on stratified sub-sample "
          f"(n={len(sub_idx)}, n_perm={n_perm}) ...")

    for feat_name, X in [("f_disc", X_disc_sub), ("f_indisc", X_ind_sub)]:
        hsic_val, p_val, null_mean, null_std = hsic_permutation_test(
            X, y_sub, n_perm=n_perm, seed=seed,
        )
        results[f"hsic_{feat_name}"] = {
            "hsic":     hsic_val,
            "p_value":  p_val,
            "null_mean": null_mean,
            "null_std":  null_std,
        }
        print(f"  [HSIC]   {feat_name}: HSIC = {hsic_val:.6f} "
              f"(null mean = {null_mean:.6f} ± {null_std:.6f}, p = {p_val:.4f})")

    # ratio disc / indisc (a single number to report in the paper)
    hsic_d = results["hsic_f_disc"]["hsic"]
    hsic_i = results["hsic_f_indisc"]["hsic"]
    ratio = float(hsic_d / hsic_i) if hsic_i > 0 else float("inf")
    results["hsic_ratio"] = ratio
    print(f"  [HSIC]   ratio HSIC(f_disc)/HSIC(f_indisc) = {ratio:.3f}")

    # ── Save per-seed CSV ────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    seed_csv = os.path.join(out_dir, f"linerprobe_seed{seed}.csv")
    fieldnames = ["probe_target", "feature", "accuracy", "bal_accuracy",
                  "macro_f1", "auroc", "best_C", "hsic", "hsic_p_value"]

    with open(seed_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        # tumor rows
        for key, m in results.items():
            if not key.startswith("tumor_"):
                continue
            target, feat = key.split("_", 1)
            w.writerow({
                "probe_target":  target,
                "feature":       feat,
                "accuracy":      f"{m['accuracy']:.4f}",
                "bal_accuracy":  f"{m['bal_accuracy']:.4f}",
                "macro_f1":      f"{m['macro_f1']:.4f}",
                "auroc":         f"{m['auroc']:.4f}",
                "best_C":        m["best_C"],
                "hsic":          "",
                "hsic_p_value":  "",
            })
        # hsic rows
        for key, m in results.items():
            if not key.startswith("hsic_f"):
                continue
            feat = key.split("_", 1)[1]
            w.writerow({
                "probe_target":  "hsic",
                "feature":       feat,
                "accuracy":      "",
                "bal_accuracy":  "",
                "macro_f1":      "",
                "auroc":         "",
                "best_C":        "",
                "hsic":          f"{m['hsic']:.6f}",
                "hsic_p_value":  f"{m['p_value']:.4f}",
            })
        # ratio row
        w.writerow({
            "probe_target":  "hsic_ratio",
            "feature":       "f_disc_over_f_indisc",
            "accuracy":      "",
            "bal_accuracy":  "",
            "macro_f1":      "",
            "auroc":         "",
            "best_C":        "",
            "hsic":          f"{ratio:.4f}",
            "hsic_p_value":  "",
        })

    print(f"\n  → Saved {seed_csv}")

    seed_json = os.path.join(out_dir, f"linerprobe_seed{seed}.json")
    with open(seed_json, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"  → Saved {seed_json}")

    return results


def summarise_all_seeds(all_results: dict, out_dir: str):
    """
    Compute mean ± SD over seeds for the LP and HSIC tables and write
    ``linerprobe_summary.csv``.
    """
    import statistics

    keys_order = ["tumor_f_disc", "tumor_f_indisc", "hsic_f_disc", "hsic_f_indisc"]

    summary_rows = []
    for key in keys_order:
        row = {
            "probe_target": key.split("_", 1)[0],
            "feature":      "_".join(key.split("_")[1:]),
        }
        if key.startswith("tumor_"):
            for metric in ["accuracy", "bal_accuracy", "macro_f1", "auroc"]:
                vs = [float(all_results[sk][key][metric]) for sk in sorted(all_results)]
                m, sd = statistics.fmean(vs), statistics.stdev(vs) if len(vs) >= 2 else 0.0
                row[metric] = f"{m:.4f} ± {sd:.4f}"
            row["hsic"]         = ""
            row["hsic_p_value"] = ""
        else:  # hsic_*
            vs  = [float(all_results[sk][key]["hsic"])    for sk in sorted(all_results)]
            pvs = [float(all_results[sk][key]["p_value"]) for sk in sorted(all_results)]
            m, sd = statistics.fmean(vs), statistics.stdev(vs) if len(vs) >= 2 else 0.0
            row["accuracy"]     = ""
            row["bal_accuracy"] = ""
            row["macro_f1"]     = ""
            row["auroc"]        = ""
            row["hsic"]         = f"{m:.6f} ± {sd:.6f}"
            row["hsic_p_value"] = f"{statistics.fmean(pvs):.4f}"

        # ratio row appended separately
        summary_rows.append(row)

    # ratio row (mean ± SD over seeds)
    ratios = [float(all_results[sk]["hsic_ratio"]) for sk in sorted(all_results)]
    m, sd = statistics.fmean(ratios), statistics.stdev(ratios) if len(ratios) >= 2 else 0.0
    summary_rows.append({
        "probe_target":  "hsic_ratio",
        "feature":       "f_disc_over_f_indisc",
        "accuracy":      "",
        "bal_accuracy":  "",
        "macro_f1":      "",
        "auroc":         "",
        "hsic":          f"{m:.4f} ± {sd:.4f}",
        "hsic_p_value":  "",
    })

    summary_csv = os.path.join(out_dir, "linerprobe_summary.csv")
    fieldnames  = ["probe_target", "feature", "accuracy", "bal_accuracy",
                   "macro_f1", "auroc", "hsic", "hsic_p_value"]

    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n{'='*70}")
    print("  SUMMARY  (mean ± SD, 3 seeds)")
    print(f"{'='*70}")
    print(f"{'Probe':<12} {'Feature':<22} {'Acc.':>14} {'BAcc.':>14} {'HSIC':>20}")
    print("-" * 88)
    for r in summary_rows:
        acc  = r.get("accuracy", "") or "-"
        bacc = r.get("bal_accuracy", "") or "-"
        hsic = r.get("hsic", "") or "-"
        print(f"{r['probe_target']:<12} {r['feature']:<22} "
              f"{acc:>14} {bacc:>14} {hsic:>20}")
    print("-" * 88)
    print(f"\nSaved: {summary_csv}")
    return summary_rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    SEED = int(os.environ.get("SEED", "0"))   # 0 → run all three seeds
    if SEED == 0:
        seeds_to_run = [42, 123, 666]
    else:
        seeds_to_run = [SEED]

    all_results = {}
    for seed in seeds_to_run:
        cache_dir = CACHE_DIRS.get(seed)
        if cache_dir is None:
            print(f"[SKIP] No cache dir configured for seed {seed}")
            continue
        results = run_single_seed(seed, cache_dir, OUT_DIR)
        all_results[str(seed)] = results

    if len(all_results) >= 2:
        summarise_all_seeds(all_results, OUT_DIR)
    else:
        print("[info] Only one seed run — summary CSV not generated (need ≥ 2 seeds).")

    print(f"\n[done] All results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
