import argparse
import csv
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from materials import asset, canonical, map_paths, read_config
from topoformer.data import datasets
from topoformer.data.datasets import AlfredoFDDataset, make_loaders
from topoformer.data.split_builders import (
    _scan_class_folders_png, _scan_files_by_prefix_label,
    build_briancancer_train_test, build_nctcrc100k_trainval_crcval7k_test,
)
from topoformer.data.topo_maps import TopoMapStore
from topoformer.engine.preflight import preflight
from topoformer.engine.runtime import build_train_args, save_npz, setup_logger
from topoformer.model.checkpoints import dino_sha256, load_selected_weights
from topoformer.model.disentanglement import set_seed, train
from topoformer.model.model_builder import build_topoformer_model

ROOT = Path(__file__).resolve().parent
SEEDS = {"briancancer": (42, 123, 666), "crc": (12, 42, 666)}
ABLATED = ("no-convnext", "dino-dnfd-cti", "dino-cti", "no-dnfd")
VARIANTS = ("full", "no-cti", *ABLATED, "L0", "L1", "L2", "L3", "simpleconcat", "external")
CONDITIONS = ("clean", "stain_shift", "patch_shuffle", "external_raw", "external_reinhard")
TCGA_CLASSES = {"astrocytoma": "astro", "glioblastoma": "gbm", "oligodendroglioma": "oligo"}


def inputs(config, dataset, threshold, variant, condition):
    data = config["datasets"][dataset]
    maps = data["concat_maps"] if variant == "simpleconcat" else data["maps"]
    pool = data["pooled"]
    test_base = None
    if variant == "external" and condition == "clean":
        condition = "external_raw"
    if condition != "clean":
        item = data["conditions"][condition]
        test_base = asset(item["maps"], threshold)
        pool = item["pooled"]
    return asset(maps, threshold), test_base, asset(pool, threshold)


def load_images(config, dataset, variant, condition):
    paths = config["datasets"][dataset]
    external = variant == "external"
    if dataset == "crc":
        split = build_nctcrc100k_trainval_crcval7k_test(paths["train"], paths["test"])
    else:
        split = build_briancancer_train_test(paths["train"], paths["test"])
    images = np.asarray(split.paths, dtype=str)
    labels = np.asarray(split.label_names)[split.y_all]
    train_idx = np.concatenate((split.train_idx, split.val_idx)) if split.val_idx is not None else split.train_idx
    test_idx = split.test_idx

    if external:
        classes = sorted(TCGA_CLASSES.values())
        train_idx = np.sort(train_idx[np.isin(labels[train_idx], classes)])
        key = "external_raw" if condition == "clean" else condition
        root = paths["conditions"][key]["images"]
        mapping = {**TCGA_CLASSES, **{name: name for name in classes}}
        root = Path(root)
        test = [str(file) for folder in sorted(root.iterdir()) if folder.name.lower() in mapping
                for file in sorted(folder.rglob("*")) if file.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        targets = np.asarray([mapping[Path(path).relative_to(root).parts[0].lower()] for path in test])
        if condition == "external_reinhard":
            source = Path(paths["conditions"]["external_raw"]["images"])
            raw = {file.relative_to(source) for folder in source.iterdir() if folder.name.lower() in mapping
                   for file in folder.rglob("*") if file.suffix.lower() in {".png", ".jpg", ".jpeg"}}
            if {Path(path).relative_to(root) for path in test} != raw:
                raise ValueError("Raw and Reinhard TCGA image IDs differ")
        if paths.get("external_manifest"):
            with Path(paths["external_manifest"]).open(newline="", encoding="utf-8") as source:
                expected = {row["PatchPath"]: row["MappedLabel5to3"] for row in csv.DictReader(source)}
            observed = {Path(path).relative_to(root).as_posix(): label for path, label in zip(test, targets)}
            if observed != expected:
                raise ValueError("TCGA patch IDs or class labels differ from the external manifest")
        if len(test) != 786:
            raise ValueError(f"Expected TCGA manifest subset of 786 images, got {len(test)} at {root}")
        groups = (images[train_idx], np.asarray(test))
        images = np.concatenate(groups)
        labels = np.concatenate((labels[train_idx], targets))
        train_idx = np.arange(len(groups[0]))
        test_idx = np.arange(len(groups[0]), len(images))
        train_idx, val_idx = train_test_split(train_idx, test_size=0.1, stratify=labels[train_idx], random_state=42)
    else:
        classes = list(split.label_names)
        reference = paths.get("split_manifest")
        if reference:
            all_paths = map_paths(images, config)
            lookup = {name: i for i, name in enumerate(all_paths)}
            with np.load(reference, allow_pickle=False) as manifest:
                train_idx, val_idx, test_idx = (
                    np.asarray([lookup[name] for name in map_paths(manifest[key], config)])
                    for key in ("train", "val", "test")
                )
            covered = np.concatenate((train_idx, val_idx, test_idx))
            if len(covered) != len(images) or len(set(covered)) != len(images):
                raise ValueError("Official split manifest does not cover the dataset exactly once")
        elif split.val_idx is not None:
            train_idx, val_idx = split.train_idx, split.val_idx
        else:
            raise ValueError("CRC requires its original split manifest to preserve order across servers")
        if condition != "clean":
            root = paths["conditions"][condition]["images"]
            if dataset == "crc":
                test, test_labels = _scan_files_by_prefix_label(root, exts=(".tif", ".tiff", ".png"))
            else:
                test, test_labels = _scan_class_folders_png(root, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff"))
            official = {Path(path).stem: label for path, label in zip(images[test_idx], labels[test_idx])}
            if len(official) != len(test_idx) or len(set(map(lambda p: Path(p).stem, test))) != len(test):
                raise ValueError("Duplicate official or perturbed patch IDs")
            if any(official.get(Path(path).stem) != label for path, label in zip(test, test_labels)):
                raise ValueError("Perturbed test IDs or class labels disagree with the official set")
            expected = 7179 if dataset == "crc" else 4174
            if len(test) != expected:
                raise ValueError(f"Expected {expected} perturbed patches, got {len(test)}")
            test_idx = np.arange(len(images), len(images) + len(test))
            images = np.concatenate((images, test))
            labels = np.concatenate((labels, test_labels))

    return map_paths(images, config), np.searchsorted(classes, labels), classes, {
        "train": train_idx, "val": val_idx, "test": test_idx,
    }


def build_model(config, variant, store, classes, device):
    sys.path.insert(0, str(ROOT / "Backbone" / "dinov3" / "dinov3"))
    levels = (int(variant[1]),) if variant in ("L0", "L1", "L2", "L3") else (0, 1, 2, 3)
    return build_topoformer_model(
        device=device, fd_label_names=classes, dino_local_ckpt=config["dinov3"],
        encoder_name="dinov3_vit7b16", topo_c=400 if store is None else store.topo_c,
        fd_finetune_mode="lora", fd_unfreeze_last_k=0, fd_lora_r=16, fd_lora_alpha=16.0, fd_lora_dropout=0.05,
        use_topo_inject=False, topo_attn_dropout=0.1, topo_proj_dropout=0.1, topo_num_heads=None,
        convnext_ckpt=None if variant in ("no-convnext", "dino-dnfd-cti", "dino-cti") else config["convnext"],
        convnext_name="convnextv2_base", cti_grid=None if store is None else store.topo_hw[0], cti_levels=levels,
        topo_inject_mode="raw_concat" if variant == "simpleconcat" else "per_level",
        dnfd=variant not in ("no-dnfd", "dino-cti"),
    )


def load_checkpoint(model, run, config=None):
    load_selected_weights(model, run / "best_val.pt", config)
    model.eval()


def check_run(run, dataset, threshold, seed, variant):
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    expected = {"dataset": dataset, "threshold": threshold, "seed": seed, "variant": variant,
                "cti": "disabled" if variant == "no-cti" else "gated-local-cls",
                "checkpoint": "raw-at-best-ema-val-epoch"}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Run metadata does not match the requested experiment: {run}")


def check_manifest(run, images, indices, config, condition):
    with np.load(run / "splits.npz", allow_pickle=False) as manifest:
        for split in ("train", "val", "test") if condition == "clean" else ("train", "val"):
            np.testing.assert_array_equal(map_paths(manifest[split], config), images[indices[split]])


def run_model(args):
    config = read_config(args.config)
    run = args.run
    condition = args.condition
    set_seed(args.seed)
    # A process gets one physical GPU through CUDA_VISIBLE_DEVICES.
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    datasets.NUM_WORKERS = args.workers
    base, test_base, _ = inputs(config, args.dataset, args.threshold, args.variant, condition)
    path_key = partial(canonical, config=config)
    is_export = args.action == "export"
    splits = [args.split] if isinstance(args.split, str) else args.split
    test_only = is_export and splits == ["test"]
    store = None if args.variant == "no-cti" else TopoMapStore(
        base, path_key=path_key, concat=args.variant == "simpleconcat", test_base=test_base, test_only=test_only,
    )
    images, y, classes, indices = load_images(config, args.dataset, args.variant, condition)
    subjects = indices["test"] if test_only else np.concatenate(list(indices.values()))
    if store is not None:
        store.check_alignment(images[subjects])
    if is_export:
        check_run(run, args.dataset, args.threshold, args.seed, args.variant)
        check_manifest(run, images, indices, config, condition)
    model = build_model(config, args.variant, store, classes, device)
    batch = 8
    train_args = build_train_args(device, False, args.epochs, 5e-5, "lora", 0, 16, 16.0, 0.05)
    if not model.dnfd:
        train_args.lambda_adv = train_args.lambda_ortho = 0.0
        train_args.adv_mode = "none"
        train_args.decouple_mode = "none"
    if args.action in ("preflight", "train"):
        train_loader, val_loader, _ = make_loaders(
            images, y, indices["train"], indices["test"], store, device, batch,
            val_idx=indices["val"], balance_sampler=True,
        )
        hook = getattr(model.vit.encoder, "_topo_hook", None)
        if args.variant == "no-cti":
            if hook is not None:
                raise RuntimeError("Unexpected token interaction hook in no-cti")
            print("NO_CTI_PREFLIGHT_PASS", flush=True)
        else:
            preflight(model, train_loader, device, amp=train_args.amp)
        if args.action == "preflight":
            return
        if config.get("compact_checkpoints"):
            train_args.cti_dino_sha256 = dino_sha256(config)
        run.mkdir(parents=True, exist_ok=False)
        (run / "run.json").write_text(json.dumps({
            "dataset": args.dataset, "threshold": args.threshold, "seed": args.seed, "variant": args.variant,
            "cti": "disabled" if args.variant == "no-cti" else "gated-local-cls",
            "checkpoint": "raw-at-best-ema-val-epoch", "classes": classes, "epochs": args.epochs,
        }, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(run / "splits.npz", **{key: images[idx] for key, idx in indices.items()})
        train_args.cti_audit = hook
        train_args.cti_output = run
        history, best = train(model, train_loader, val_loader, device, train_args, setup_logger("topoformer"))
        if len(history) != args.epochs or best is None:
            raise RuntimeError(f"Expected {args.epochs} complete epochs and a validation checkpoint")
        best_epoch = max(history, key=lambda row: row["bal_acc_A"])["epoch"]
        (run / "best_epoch.txt").write_text(str(best_epoch), encoding="utf-8")
        print(f"TRAIN_COMPLETE {run} epoch={best_epoch}")
        return

    load_checkpoint(model, run, config)
    best_epoch = int((run / "best_epoch.txt").read_text())
    output = run if condition == "clean" else run / condition
    output.mkdir(exist_ok=True)
    for split in splits:
        file = output / f"dinov3FD_{args.dataset}_{split}.npz"
        if file.exists():
            raise FileExistsError(f"Export already exists: {file}")
    for split in splits:
        loader = DataLoader(
            AlfredoFDDataset(indices[split], images, y, store, device, return_path=True),
            batch_size=batch, shuffle=False, num_workers=args.workers, pin_memory=True,
        )
        save_npz(split, loader, model, device, str(output), args.dataset, dict(enumerate(classes)), classes,
                 "dinov3_vit7b16", config["dinov3"], batch, best_epoch, test_base or base,
                 0 if store is None else store.topo_c, None if store is None else store.topo_hw)


def main():
    parser = argparse.ArgumentParser(description="Train or export a gated TopoFormer experiment")
    parser.add_argument("action", choices=("preflight", "train", "export"))
    parser.add_argument("dataset", choices=SEEDS)
    parser.add_argument("threshold", type=int, choices=(160, 180, 200))
    parser.add_argument("seed", type=int)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("run", type=Path)
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument("--condition", choices=CONDITIONS, default="clean")
    parser.add_argument("--split", choices=("train", "val", "test"), nargs="+", default=["test"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be an unsigned 32-bit integer")
    if args.variant in {"L0", "L1", "L2", "L3", "simpleconcat", "external"} and args.dataset != "briancancer":
        parser.error("S4 and external training use DeepHisto")
    if len(set(args.split)) != len(args.split):
        parser.error("Specify each export split once")
    if args.condition != "clean" and (args.action != "export" or args.split != ["test"]):
        parser.error("Transformed inputs are held-out test exports")
    if args.condition.startswith("external_") != (args.variant == "external") and args.condition != "clean":
        parser.error("External test inputs require the three-class model")
    if args.action == "train" and args.run.exists():
        parser.error("Training needs a new directory. Use export to reuse a checkpoint.")
    run_model(args)


if __name__ == "__main__":
    main()
