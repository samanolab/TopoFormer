import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from materials import asset, map_paths, read_config

METRICS = ("accuracy", "balanced_accuracy", "macro_precision", "macro_specificity_ovr", "macro_f1", "macro_auroc_ovr")
ALIASES = {"astrocytoma": "astro", "glioblastoma": "gbm", "oligodendroglioma": "oligo"}


def measures(y, probability):
    nclass = probability.shape[1]
    pred = probability.argmax(axis=1)
    matrix = np.bincount(y * nclass + pred, minlength=nclass * nclass).reshape(nclass, nclass)
    positives = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = matrix.diagonal()
    if (positives == 0).any() or (positives == len(y)).any():
        raise ValueError("Metrics require complete test label coverage")
    return np.asarray([
        tp.sum() / len(y), np.mean(tp / positives),
        np.mean(tp / np.maximum(predicted, 1)),
        np.mean(1 - (predicted - tp) / (len(y) - positives)),
        np.mean(2 * tp / (predicted + positives)),
        roc_auc_score(y, probability, multi_class="ovr", average="macro"),
    ])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_feature(path, config, feature):
    with np.load(path, allow_pickle=True) as cache:
        ids = map_paths(cache["paths"].astype(str), config)
        x = cache[feature].astype(np.float32)
        y = cache["y"].astype(np.int64)
        classes = cache["classes"].astype(str)
    if len(set(ids)) != len(ids) or len(x) != len(y) or len(x) != len(ids) or not np.isfinite(x).all():
        raise ValueError(f"Invalid feature cache: {path}")
    return x, y, ids, classes


def ph_rows(path, split, ids, classes, y, config):
    with np.load(path, allow_pickle=True) as cache:
        labels = cache[f"{split}_labels"].astype(str)
        keys = map_paths(cache[f"{split}_files"].astype(str), config)
        lookup = {key: idx for idx, key in enumerate(keys)}
        if len(lookup) != len(keys):
            raise ValueError(f"Duplicate PH IDs in {path}")
        take = [lookup[name] for name in ids]
        actual = [ALIASES.get(name.lower(), name) for name in labels[take]]
        np.testing.assert_array_equal(actual, classes[y])
        values = cache[f"{split}_phc"][take].astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite pooled PH in {path}")
    return values


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def save_predictions(output, prob, y, ids, classes):
    output.mkdir(parents=True, exist_ok=True)
    name = output / "predictions.npz"
    tmp = output / "predictions.tmp.npz"
    np.savez_compressed(tmp, paths=ids, y_true=y, y_pred=prob.argmax(axis=1), probabilities=prob, classes=classes)
    tmp.replace(name)
    result = dict(zip(METRICS, measures(y, prob)))
    result["weighted_f1"] = f1_score(y, prob.argmax(axis=1), average="weighted")
    return result


def score(args):
    config = read_config(args.config)
    data = config["datasets"][args.dataset]
    run = args.run
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if any(metadata.get(key) != getattr(args, key) for key in ("dataset", "threshold", "seed", "variant")):
        raise ValueError(f"Wrong run metadata: {run}")
    condition = args.condition
    splits = [args.split] if isinstance(args.split, str) else args.split
    if len(splits) > 1 and (condition != "clean" or len(set(splits)) != len(splits)):
        raise ValueError("Multiple score splits must be distinct and use the clean condition")
    prefix = f"dinov3FD_{args.dataset}"
    source = run if condition == "clean" else run / condition
    with np.load(run / "splits.npz", allow_pickle=False) as z:
        expected = {name: map_paths(z[name], config) for name in ("train", "val", "test")}
    if len(set(np.concatenate(tuple(expected.values())))) != sum(map(len, expected.values())):
        raise ValueError("The training/validation/test split overlaps")

    train_id_set = set(expected["train"])
    if not args.head_only:
        feature = config.get("fusion_feature", "feat")
        x_train, y_train, train_ids, train_classes = load_feature(run / f"{prefix}_train.npz", config, feature)
        np.testing.assert_array_equal(np.sort(train_ids), np.sort(expected["train"]))
        pool_key = "concat_pooled" if args.variant == "simpleconcat" else "pooled"
        train_pool = asset(data[pool_key], args.threshold)
        test_pool = train_pool if condition == "clean" else asset(data["conditions"][condition]["pooled"], args.threshold)
        h_train = ph_rows(train_pool, "train", train_ids, train_classes, y_train, config)
        mu, sd = x_train.mean(axis=0), x_train.std(axis=0) + 1e-6
        lower = h_train.min(axis=0)
        span = h_train.max(axis=0) - lower
        span[span == 0] = 1e-6
        def fuse(x, h):
            return np.concatenate((unit((x - mu) / sd), 0.3 * unit((h - lower) / span)), axis=1)
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight="balanced", solver="lbfgs")
        clf.fit(fuse(x_train, h_train), y_train)
        np.testing.assert_array_equal(clf.classes_, np.arange(len(train_classes)))
        del x_train, h_train

    for split in splits:
        output = run / "scores" / (condition if condition != "clean" else split)
        cache = source / f"{prefix}_{split}.npz"
        logits, y, ids, classes = load_feature(cache, config, "logitsA")
        if condition == "clean":
            np.testing.assert_array_equal(np.sort(ids), np.sort(expected[split]))
        if set(ids) & train_id_set:
            raise ValueError("Evaluation intersects LR fitting samples")
        result = {
            "dataset": args.dataset, "seed": args.seed, "variant": args.variant, "threshold": args.threshold,
            "condition": condition, "split": split, "n": len(y),
            "decoder_a": dict(zip(METRICS, measures(y, softmax(logits, axis=1)))),
        }
        if args.variant not in ("no-dnfd", "dino-cti"):
            logits_b, _, _, _ = load_feature(cache, config, "logitsB")
            result["decoder_b"] = dict(zip(METRICS, measures(y, softmax(logits_b, axis=1))))
        if args.head_only:
            result["primary"] = result["decoder_a"]
        else:
            np.testing.assert_array_equal(train_classes, classes)
            x_test, _, _, _ = load_feature(cache, config, feature)
            h_test = ph_rows(test_pool, "train" if split == "val" else "test", ids, classes, y, config)
            if h_test.shape[1] != len(lower):
                raise ValueError("Pooled PH feature dimensions differ; use the same scale protocol")
            prob = clf.predict_proba(fuse(x_test, h_test))
            result["primary"] = save_predictions(output, prob, y, ids, classes)
            result["fusion_feature"] = feature
        write_json(output / "metrics.json", result)
        print(output / "metrics.json")


def select(paths, output):
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(records) != 18 or len({row.get("fusion_feature") for row in records}) != 1:
        raise ValueError("All 18 threshold scores must use one common fusion feature")
    seeds = {"briancancer": {42, 123, 666}, "crc": {12, 42, 666}}
    metrics = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_auroc_ovr")
    chosen = {}
    table = [["Dataset", "Threshold", "Selected", "Seeds", *metrics]]
    for dataset, seed_set in seeds.items():
        means = {}
        sds = {}
        for threshold in (160, 180, 200):
            group = [row for row in records if row["dataset"] == dataset and row["threshold"] == threshold]
            if (len(group) != 3 or {row["seed"] for row in group} != seed_set
                    or any(row["variant"] != "full" or row["split"] != "val" for row in group)
                    or len({row.get("fusion_feature") for row in group}) != 1):
                raise ValueError(f"Need one comparable validation score per seed for {dataset}/{threshold}")
            values = {metric: [row["primary"][metric] for row in group] for metric in metrics}
            means[threshold] = {metric: float(np.mean(values[metric])) for metric in metrics}
            sds[threshold] = {metric: float(np.std(values[metric], ddof=1)) for metric in metrics}
        chosen[dataset] = {
            "threshold": max(means, key=lambda t: (means[t]["balanced_accuracy"], means[t]["accuracy"], means[t]["macro_f1"])),
            "validation_mean": means,
            "validation_sample_sd": sds,
        }
        for threshold in means:
            table.append(["DeepHisto" if dataset == "briancancer" else "CRC", str(threshold),
                          "yes" if threshold == chosen[dataset]["threshold"] else "no",
                          ",".join(map(str, sorted(seed_set))),
                          *[f"{100*means[threshold][m]:.4f} ± {100*sds[threshold][m]:.4f}" for m in metrics]])
    write_json(output, chosen)
    tsv = output.with_name("s5_threshold_table.tsv")
    pending = tsv.with_suffix(".tmp")
    pending.write_text("\n".join("\t".join(row) for row in table) + "\n", encoding="utf-8")
    pending.replace(tsv)
    print(json.dumps(chosen, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Score decoder heads or train-fit LR and select thresholds")
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("score")
    run.add_argument("dataset", choices=("briancancer", "crc"))
    run.add_argument("threshold", type=int, choices=(160, 180, 200))
    run.add_argument("seed", type=int)
    run.add_argument("variant")
    run.add_argument("run", type=Path)
    run.add_argument("--config", default="config.local.json")
    run.add_argument("--split", choices=("val", "test"), nargs="+", default=["test"])
    run.add_argument("--condition", default="clean")
    run.add_argument("--head-only", action="store_true")
    selection = commands.add_parser("select")
    selection.add_argument("--output", type=Path, required=True)
    selection.add_argument("paths", nargs=18, type=Path)
    args = parser.parse_args()
    if args.action == "select":
        select(args.paths, args.output)
    else:
        score(args)


if __name__ == "__main__":
    main()
