import argparse
from pathlib import Path

import numpy as np
from scipy.stats import t

from materials import map_paths, read_config
from legacy_tables import dnfd_representation, s9_bootstrap, s10_v8
from report import METRICS, measures, write_json


def predictions(files, config, reference=None):
    scores = []
    for file in files:
        with np.load(file, allow_pickle=False) as z:
            ids = map_paths(z["paths"].astype(str), config)
            classes = z["classes"].astype(str)
            label = classes[z["y_true"]]
            prob = z["probabilities"].astype(np.float64)
        if len(set(ids)) != len(ids) or prob.shape != (len(ids), len(classes)):
            raise ValueError(f"Invalid prediction cache: {file}")
        if reference is None:
            reference = ids, classes, label
        want_ids, want_classes, want_label = reference
        row = {name: idx for idx, name in enumerate(ids)}
        col = {name: idx for idx, name in enumerate(classes)}
        if set(ids) != set(want_ids) or set(classes) != set(want_classes):
            raise ValueError(f"Prediction dataset or class IDs differ: {file}")
        take = np.asarray([row[key] for key in want_ids])
        np.testing.assert_array_equal(label[take], want_label)
        scores.append(prob[take][:, [col[name] for name in want_classes]])
    return reference, np.asarray(scores)


def stats(files, baselines, config, replicates):
    if baselines:
        # Use the original Virchow2 manifest order for the seeded S10 bootstrap.
        reference, other = predictions(baselines, config)
        _, ours = predictions(files, config, reference=reference)
    else:
        reference, ours = predictions(sorted(files), config)
    _, classes, labels = reference
    lookup = {name: index for index, name in enumerate(classes)}
    y = np.asarray([lookup[name] for name in labels])
    if not baselines:
        rng = np.random.default_rng(s9_bootstrap.RNG_SEED)
        rows = [
            s9_bootstrap._bootstrap_one({"y": y, "pred": prob.argmax(1), "score": prob},
                                        replicates, s9_bootstrap.CI_ALPHA, rng)["metrics"]
            for prob in ours
        ]
        observed = np.stack([measures(y, scores) for scores in ours])
        mean, sd = observed.mean(0), observed.std(0, ddof=1)
        margin = t.ppf(0.975, 2) * sd / np.sqrt(3)
        names = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced Accuracy",
                 "macro_precision": "Macro Precision", "macro_f1": "Macro F1",
                 "macro_auroc_ovr": "AUC (OvR macro)"}
        result = {}
        for key, name in names.items():
            i = METRICS.index(key)
            result[key] = {"mean": mean[i], "sd": sd[i],
                           "bs_ci": [float(np.mean([r[name][bound] for r in rows]))
                                     for bound in ("ci_low", "ci_high")],
                           "t_ci": [mean[i] - margin[i], mean[i] + margin[i]]}
        return result
    engines = [s10_v8.FileMetrics(y, prob, len(classes)) for prob in (*ours, *other)]
    observed = np.asarray([engine.compute(np.ones(len(y))) for engine in engines])
    mean, sd = observed[:3].mean(0), observed[:3].std(0, ddof=1)
    baseline_mean, baseline_sd = observed[3:].mean(0), observed[3:].std(0, ddof=1)
    delta = mean - baseline_mean
    rng = np.random.default_rng(s10_v8.BOOT_SEED)
    draws = np.empty((replicates, len(METRICS)))
    for index in range(replicates):
        weights = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)).astype(np.float64)
        values = np.asarray([engine.compute(weights) for engine in engines])
        draws[index] = values[:3].mean(0) - values[3:].mean(0)
    ci = np.nanpercentile(draws, (2.5, 97.5), axis=0).T
    centered = draws - delta
    p = (1 + (np.abs(centered) >= np.abs(delta) - 1e-12).sum(0)) / (1 + np.isfinite(centered).sum(0))
    adjusted = s10_v8.holm_bonferroni(p)
    return {key: {"topoformer": [mean[i] / 100, sd[i] / 100],
                  "virchow2": [baseline_mean[i] / 100, baseline_sd[i] / 100],
                  "delta": delta[i] / 100, "ci": (ci[i] / 100).tolist(), "p_holm": adjusted[i]}
            for i, key in enumerate(METRICS)}


def main():
    parser = argparse.ArgumentParser(description="Update supplementary Tables S2, S9 and S10")
    parser.add_argument("--config", default="config.local.json")
    commands = parser.add_subparsers(dest="action", required=True)
    s2 = commands.add_parser("s2")
    s2.add_argument("run", type=Path)
    s2.add_argument("seed", type=int)
    s2.add_argument("output", type=Path)
    for action in ("s9", "s10"):
        sub = commands.add_parser(action)
        sub.add_argument("output", type=Path)
        sub.add_argument("paths", nargs=3, type=Path)
        sub.add_argument("--replicates", type=int, default=10000 if action == "s10" else 2000)
        if action == "s10":
            sub.add_argument("--virchow", nargs=3, type=Path, required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    if args.action == "s2":
        result = dnfd_representation.run_single_seed(args.seed, str(args.run), str(args.output.parent))
    else:
        result = stats(args.paths, getattr(args, "virchow", None), config, args.replicates)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
