"""Per-seed bootstrap from migration/bootstrap_fusion_metrics.py."""

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)


CI_ALPHA     = 0.05


RNG_SEED     = 2026


def _acc(y, p, s):    return accuracy_score(y, p)


def _bacc(y, p, s):   return balanced_accuracy_score(y, p)


def _mp(y, p, s):     return precision_score(y, p, average="macro", zero_division=0)


def _mr(y, p, s):     return recall_score(y, p, average="macro", zero_division=0)


def _mf(y, p, s):     return f1_score(y, p, average="macro", zero_division=0)


def _auc(y, p, s):
    return roc_auc_score(y, s, multi_class="ovr", average="macro")


METRICS = {
    "Accuracy":          _acc,
    "Balanced Accuracy": _bacc,
    "Macro Precision":   _mp,
    "Macro Recall":      _mr,
    "Macro F1":          _mf,
    "AUC (OvR macro)":   _auc,
}


def _bootstrap_one(data: dict, n_boot: int, alpha: float, rng: np.random.Generator) -> dict:
    """For a single seed, return point estimate + 95% BS-CI for each metric."""
    n   = data["y"].shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))   # (n_boot, n) with replacement
    out = {"n_test": int(n), "n_boot": int(n_boot), "metrics": {}}
    for name, fn in METRICS.items():
        samples = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            b_idx = idx[b]
            samples[b] = fn(data["y"][b_idx], data["pred"][b_idx], data["score"][b_idx])
        lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
        out["metrics"][name] = {
            "point": float(samples.mean()),
            "std":   float(samples.std(ddof=1)),
            "ci_low":  float(lo),
            "ci_high": float(hi),
            "ci_half": float(hi - lo) / 2.0,
            # for sanity: also store the per-replicate distribution as a
            # 16-bin histogram; reviewer can compare against the raw samples
            # file if they want.
            "samples_path": None,
        }
    return out
