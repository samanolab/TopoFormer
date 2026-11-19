"""Metric engine and Holm correction from table_s10_v8_three_seeds.py."""

import numpy as np


BOOT_SEED = 20261110


def _group_layout(scores):
    """Pre-compute a tie-group layout for one score vector.

    Returns (order, group_id) such that scores[order] is sorted ascending and
    group_id labels equal-score blocks.
    """
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    # boundaries of equal-value runs
    new = np.empty(len(s), dtype=bool)
    new[0] = True
    new[1:] = s[1:] != s[:-1]
    group_id = np.cumsum(new) - 1
    return order, group_id


def _weighted_auc(y_bin, w, order, group_id, n_groups):
    """Weighted one-vs-rest AUC from a multiplicity vector w.

    `y_bin` is given in the ORIGINAL sample order; it is re-ordered here to
    match the ascending-score `order`.
    """
    y_sorted = y_bin[order]
    w_sorted = w[order]
    pos_g = np.bincount(group_id, weights=w_sorted * y_sorted, minlength=n_groups)
    neg_g = np.bincount(group_id, weights=w_sorted * (1 - y_sorted), minlength=n_groups)
    w_pos = pos_g.sum()
    w_neg = neg_g.sum()
    if w_pos <= 0 or w_neg <= 0:
        return np.nan
    cum_neg_before = np.cumsum(neg_g) - neg_g
    contrib = pos_g * (cum_neg_before + 0.5 * neg_g)
    return float(contrib.sum() / (w_pos * w_neg))


class FileMetrics:
    """Pre-computed, reusable weighted metric evaluator for one prediction file."""

    def __init__(self, y_true, probs, n_classes):
        self.n = len(y_true)
        self.K = n_classes
        self.y = np.asarray(y_true, dtype=np.int64)
        self.probs = np.asarray(probs, dtype=np.float64)
        self.y_pred = self.probs.argmax(axis=1)
        self.flat = self.y * self.K + self.y_pred
        # per-class tie layouts for the OvR AUC
        self.layouts = []
        for c in range(self.K):
            order, gid = _group_layout(self.probs[:, c])
            self.layouts.append((order, gid, int(gid.max()) + 1,
                                 (self.y == c).astype(np.float64)))

    def compute(self, w):
        """Six Table I metrics (percent) under sample multiplicities w."""
        n_eff = float(w.sum())
        cm = np.bincount(self.flat, weights=w, minlength=self.K * self.K).reshape(self.K, self.K)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - tp - fp - fn

        acc = float(np.trace(cm)) / n_eff * 100.0

        w_true_cls = cm.sum(axis=1)          # total weight of each true class
        w_pred_cls = cm.sum(axis=0)          # total weight of each predicted class

        # --- balanced accuracy: mean recall over classes present in y_true
        present = w_true_cls > 0
        rec_den = tp + fn
        with np.errstate(invalid="ignore", divide="ignore"):
            rec = np.where(rec_den > 0, tp / np.maximum(rec_den, 1e-300), 0.0)
        bacc = float(rec[present].mean()) * 100.0 if present.any() else np.nan

        # --- macro precision / F1 over union(present in y_true, present in y_pred)
        union = (w_true_cls > 0) | (w_pred_cls > 0)
        prec_den = tp + fp
        f1_den = 2 * tp + fp + fn
        with np.errstate(invalid="ignore", divide="ignore"):
            prec_c = np.where(prec_den > 0, tp / np.maximum(prec_den, 1e-300), 0.0)
            rec_c = np.where(rec_den > 0, tp / np.maximum(rec_den, 1e-300), 0.0)
            f1_c = np.where(f1_den > 0, 2 * tp / np.maximum(f1_den, 1e-300), 0.0)
        prec = float(prec_c[union].mean()) * 100.0 if union.any() else np.nan
        f1 = float(f1_c[union].mean()) * 100.0 if union.any() else np.nan

        # --- macro specificity (OvR) over classes present in y_true
        spec_den = tn + fp
        with np.errstate(invalid="ignore", divide="ignore"):
            spec_c = np.where(spec_den > 0, tn / np.maximum(spec_den, 1e-300), 0.0)
        spec = float(spec_c[present].mean()) * 100.0 if present.any() else np.nan

        # --- macro OvR AUC over classes present in y_true
        aucs = []
        for c in range(self.K):
            if w_true_cls[c] <= 0:
                continue
            order, gid, ng, ybin = self.layouts[c]
            a = _weighted_auc(ybin, w, order, gid, ng)
            if not np.isnan(a):
                aucs.append(a)
        auc = float(np.mean(aucs)) * 100.0 if aucs else np.nan

        return np.array([acc, bacc, prec, spec, f1, auc], dtype=np.float64)


def holm_bonferroni(pvals):
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, p[idx] * (n - rank))
        adj[idx] = min(running, 1.0)
    return adj
