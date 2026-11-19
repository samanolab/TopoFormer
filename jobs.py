import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from zipfile import is_zipfile

from materials import asset, read_config

SEEDS = {"briancancer": (42, 123, 666), "crc": (12, 42, 666)}
THRESHOLDS = (160, 180, 200)
ABLATIONS = ("no-convnext", "no-dnfd", "dino-dnfd-cti", "dino-cti")
PH_VARIANTS = ("L0", "L1", "L2", "L3", "simpleconcat")


def full_path(config, dataset, threshold, seed):
    return Path(config["run_root"]) / "threshold" / dataset / str(threshold) / str(seed)


def variant_path(config, dataset, threshold, seed, variant):
    return Path(config["run_root"]) / variant / dataset / str(threshold) / str(seed)


def results_seeds(chosen):
    return {ds: tuple(chosen[ds].get("seeds", initial)) for ds, initial in SEEDS.items()}


def stage_jobs(stage, seeds=SEEDS):
    if stage == "threshold":
        return [(ds, t, seed) for ds, seeds in SEEDS.items() for t in THRESHOLDS for seed in seeds]
    if stage == "main":
        return [(ds, seed) for ds, group in seeds.items() for seed in group]
    if stage == "ablation":
        return [(ds, seed, kind) for ds, group in seeds.items() for kind in ABLATIONS for seed in group]
    if stage == "robust":
        return [(ds, seed) for ds, group in seeds.items() for seed in group]
    if stage == "cam":
        return [(ds, kind) for ds in SEEDS for kind in ("full", "no-convnext", "no-dnfd")]
    if stage == "external":
        return [(seed,) for seed in seeds["briancancer"]]
    if stage == "s4":
        return [(seed, kind) for kind in PH_VARIANTS for seed in seeds["briancancer"]]
    if stage == "s2":
        return [(seed,) for seed in seeds["briancancer"]]
    return [(ds,) for ds in SEEDS]


def call(script, *items):
    command = [sys.executable, script, *map(str, items)]
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def adopt(source, dest, dataset, threshold, seed):
    source = source.resolve()
    checkpoint = source / "best_val.pt"
    manifest = source / "splits.npz"
    epoch = source / "best_epoch.txt"
    if epoch.is_file():
        best = int(epoch.read_text())
    else:
        log = source / "train.log"
        selected = re.findall(r"\[cti\] selected_epoch=(\d+)", log.read_text(errors="replace")) if log.is_file() else []
        if not selected:
            raise FileNotFoundError(f"Source has not finalized an epoch: {source}")
        best = int(selected[-1])
    if not checkpoint.is_file() or not is_zipfile(checkpoint) or not is_zipfile(manifest):
        raise ValueError(f"Incomplete threshold checkpoint or manifest: {source}")
    dest.mkdir(parents=True, exist_ok=False)
    for path in (checkpoint, manifest, *(source / f"dinov3FD_{dataset}_{split}.npz" for split in ("train", "val", "test"))):
        if path.exists() and is_zipfile(path):
            (dest / path.name).symlink_to(path)
    (dest / "best_epoch.txt").write_text(str(best), encoding="utf-8")
    (dest / "run.json").write_text(json.dumps({
        "dataset": dataset, "threshold": threshold, "seed": seed, "variant": "full",
        "cti": "gated-local-cls", "checkpoint": "raw-at-best-ema-val-epoch", "source": str(source),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Reusing {source}")


def train_if_missing(config_file, config, ds, t, seed, variant, run, epochs=2):
    if (run / "best_epoch.txt").is_file() and (run / "best_val.pt").is_file():
        return
    source = None
    if variant == "full" and run == full_path(config, ds, t, seed):
        source = config.get("existing_threshold_runs", {}).get(f"{ds}/{t}/{seed}")
    if source:
        adopt(Path(source), run, ds, t, seed)
    else:
        call("experiment.py", "train", ds, t, seed, variant, run, "--config", config_file, "--epochs", epochs)


def export_and_score(config_file, ds, t, seed, variant, run, *splits, condition="clean", head_only=False):
    prefix = f"dinov3FD_{ds}"
    kinds = [(split, condition) for split in splits]
    if not head_only:
        kinds.insert(0, ("train", "clean"))
    exports = {}
    for name, kind in kinds:
        cache = (run if kind == "clean" else run / kind) / f"{prefix}_{name}.npz"
        if not cache.is_file():
            exports.setdefault(kind, []).append(name)
    for kind, missing in exports.items():
        call("experiment.py", "export", ds, t, seed, variant, run,
             "--config", config_file, "--split", *missing, "--condition", kind)
    extra = ["--head-only"] if head_only else []
    call("report.py", "score", ds, t, seed, variant, run, "--config", config_file,
         "--split", *splits, "--condition", condition, *extra)


def threshold_score(config, ds, threshold, seed, split):
    run = full_path(config, ds, threshold, seed)
    scores = run / "scores" / split
    metrics = scores / "metrics.json"
    cache = run / f"dinov3FD_{ds}_{split}.npz"
    if not (metrics.is_file() and is_zipfile(cache) and is_zipfile(scores / "predictions.npz")):
        return None
    record = json.loads(metrics.read_text(encoding="utf-8"))
    expected = {
        "dataset": ds, "threshold": threshold, "seed": seed, "variant": "full",
        "condition": "clean", "split": split, "fusion_feature": config.get("fusion_feature", "feat"),
    }
    if all(record.get(key) == value for key, value in expected.items()):
        return record["primary"]
    return None


def preview_thresholds(config):
    print("Dataset\tThreshold\tSeed\tVal BAcc (%)\tTest BAcc (%)\tTest Accuracy (%)\tTest Macro F1 (%)")
    for ds, threshold, seed in stage_jobs("threshold"):
        val = threshold_score(config, ds, threshold, seed, "val")
        test = threshold_score(config, ds, threshold, seed, "test")
        values = (
            (val, "balanced_accuracy"), (test, "balanced_accuracy"),
            (test, "accuracy"), (test, "macro_f1"),
        )
        fields = [f"{metrics[key] * 100:.2f}" if metrics is not None else "pending" for metrics, key in values]
        name = "DeepHisto" if ds == "briancancer" else "CRC"
        print("\t".join((name, str(threshold), str(seed), *fields)))


def main_records(config, ds, t, seeds):
    return [full_path(config, ds, t, seed) / "scores/test/predictions.npz" for seed in seeds[ds]]


def run_stage(config_file, config, stage, values):
    root = Path(config["run_root"])
    if stage == "threshold":
        ds, t, seed = values
        run = full_path(config, ds, t, seed)
        train_if_missing(config_file, config, ds, t, seed, "full", run)
        missing = [split for split in ("val", "test") if threshold_score(config, ds, t, seed, split) is None]
        if missing:
            export_and_score(config_file, ds, t, seed, "full", run, *missing)
        return

    chosen = json.loads((root / "selected_thresholds.json").read_text(encoding="utf-8"))
    seeds = results_seeds(chosen)
    thresholds = {ds: chosen[ds]["threshold"] for ds in SEEDS}
    if stage == "main":
        ds, seed = values
        t = thresholds[ds]
        if threshold_score(config, ds, t, seed, "test") is None:
            export_and_score(config_file, ds, t, seed, "full", full_path(config, ds, t, seed), "test")
        return

    if stage == "s9":
        ds, = values
        call("analysis_tables.py", "--config", config_file, "s9", root / f"s9_{ds}.json",
             *main_records(config, ds, thresholds[ds], seeds))
        return

    for ds, t in thresholds.items():
        required = [full_path(config, ds, t, seed) / "scores/test/metrics.json" for seed in seeds[ds]]
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(f"Finish Table I scoring for {ds} before the ablations")
    if stage == "ablation":
        ds, seed, kind = values
        t = thresholds[ds]
        run = variant_path(config, ds, t, seed, kind)
        train_if_missing(config_file, config, ds, t, seed, kind, run)
        export_and_score(config_file, ds, t, seed, kind, run, "test", head_only=kind in ("dino-cti", "dino-dnfd-cti"))
        return

    for ds, seed, kind in stage_jobs("ablation", seeds):
        t = thresholds[ds]
        if not (variant_path(config, ds, t, seed, kind) / "scores/test/metrics.json").is_file():
            raise FileNotFoundError("Finish all Table IV ablations before stage 4")

    if stage == "robust":
        ds, seed = values
        t = thresholds[ds]
        run = full_path(config, ds, t, seed)
        for condition in ("stain_shift", "patch_shuffle"):
            export_and_score(config_file, ds, t, seed, "full", run, "test", condition=condition)
    elif stage == "cam":
        ds, kind = values
        t = thresholds[ds]
        seed = 42 if 42 in seeds[ds] else seeds[ds][0]
        run = full_path(config, ds, t, seed) if kind == "full" else variant_path(config, ds, t, seed, kind)
        call("gradcam.py", ds, t, kind, run, "--config", config_file)
    elif stage in ("external", "s4"):
        seed = values[0]
        ds = "briancancer"
        t = thresholds[ds]
        kind = "external" if stage == "external" else values[1]
        run = variant_path(config, ds, t, seed, kind)
        train_if_missing(config_file, config, ds, t, seed, kind, run)
        for condition in ("external_raw", "external_reinhard") if kind == "external" else ("clean",):
            export_and_score(config_file, ds, t, seed, kind, run, "test",
                             condition=condition, head_only=kind in ("L0", "L1", "L2", "L3"))
    elif stage == "s2":
        seed, = values
        call("analysis_tables.py", "--config", config_file, "s2",
             full_path(config, "briancancer", thresholds["briancancer"], seed), seed, root / f"s2_{seed}.json")
    else:
        ds, = values
        files = sorted(Path("assets/virchow").glob(("deephisto" if ds == "briancancer" else "crc") + "_virchow2*.npz"))
        call("analysis_tables.py", "--config", config_file, "s10", root / f"s10_{ds}.json",
             *main_records(config, ds, thresholds[ds], seeds), "--virchow", *files)


def doctor(config):
    missing = []
    def check(path):
        if not Path(path).exists():
            missing.append(str(path))
    check(config["dinov3"])
    check(config["convnext"])
    for ds in SEEDS:
        data = config["datasets"][ds]
        for key in ("train", "test", "split_manifest", "faithfulness", "panels"):
            check(data[key])
        for t in THRESHOLDS:
            check(asset(data["pooled"], t))
            raw = asset(data["maps"], t)
            for i in range(4):
                filename = Path(f"{raw}_L{i}.npz")
                if filename.with_suffix(".mmap").is_dir():
                    continue
                check(filename)
    for item in sorted(set(missing)):
        print(f"MISSING {item}")
    if missing:
        raise SystemExit(1)
    print("Core inputs exist. Verify map shape/path alignment before GPU work.")


def main():
    stages = ("threshold", "main", "s9", "ablation", "s4", "robust", "cam", "external", "s2", "s10")
    parser = argparse.ArgumentParser(description="List or run one explicitly selected TopoFormer task")
    parser.add_argument("--config", default="config.local.json")
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("doctor")
    commands.add_parser("select")
    commands.add_parser("preview", help="Show completed validation and held-out test results by threshold and seed")
    commands.add_parser("summary")
    for action in ("list", "run"):
        part = commands.add_parser(action)
        part.add_argument("stage", choices=stages)
        if action == "run":
            part.add_argument("items", nargs="+")
    args = parser.parse_args()
    if args.action == "list":
        for values in stage_jobs(args.stage):
            print(shlex.join(["python", "jobs.py", "--config", args.config, "run", args.stage, *map(str, values)]))
        return
    config = read_config(args.config)
    if args.action == "doctor":
        doctor(config)
    elif args.action == "preview":
        preview_thresholds(config)
    elif args.action == "select":
        reports = [full_path(config, ds, t, seed) / "scores/val/metrics.json" for ds, t, seed in stage_jobs("threshold")]
        call("report.py", "select", "--output", Path(config["run_root"]) / "selected_thresholds.json", *reports)
    elif args.action == "summary":
        call("summarize.py", "--config", args.config)
    else:
        values = tuple(int(value) if value.isdecimal() else value for value in args.items)
        chosen_path = Path(config["run_root"]) / "selected_thresholds.json"
        seeds = results_seeds(json.loads(chosen_path.read_text(encoding="utf-8"))) if args.stage != "threshold" else SEEDS
        if values not in stage_jobs(args.stage, seeds):
            parser.error("Unknown task. List the stage to see the allowed dataset, threshold, seed and variant.")
        run_stage(args.config, config, args.stage, values)


if __name__ == "__main__":
    main()
