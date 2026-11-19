import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from experiment import inputs, load_images
from materials import asset, read_config
from ph.features import WINDOWS, extract


def main():
    parser = argparse.ArgumentParser(description="Prepare PH maps and pooled features at the chosen threshold")
    parser.add_argument("dataset", choices=("briancancer", "crc"))
    parser.add_argument("threshold", type=int, choices=(160, 180, 200))
    parser.add_argument("kind", choices=("clean", "simpleconcat", "stain_shift", "patch_shuffle", "external_raw", "external_reinhard"))
    parser.add_argument("--config", default="config.local.json")
    args = parser.parse_args()
    config = read_config(args.config)
    concat = args.kind == "simpleconcat"
    condition = "clean" if concat else args.kind
    variant = args.kind if concat else ("external" if condition.startswith("external") else "full")
    base, test_maps, pool = inputs(config, args.dataset, args.threshold, variant, condition)
    if concat:
        pool = asset(config["datasets"][args.dataset]["concat_pooled"], args.threshold)
    elif condition != "clean":
        base = test_maps
    images, y, classes, indices = load_images(config, args.dataset, variant, condition)
    splits = {"test": indices["test"]}
    if condition == "clean":
        splits["train"] = np.concatenate((indices["train"], indices["val"]))
    pool_levels = 4 if concat else 3
    if condition != "clean":
        original_pool = asset(config["datasets"][args.dataset]["pooled"], args.threshold)
        with np.load(original_pool, allow_pickle=True) as existing:
            width = existing["test_phc"].shape[1]
        if width not in (1200, 1600):
            raise ValueError(f"Expected three or four pooled scales, got {width} features")
        pool_levels = width // 400
    dirs = [Path(f"{base}_L{i}.mmap") for i in range(4)]
    if Path(pool).exists() or any(path.exists() or path.with_suffix(".npz").exists() for path in dirs):
        raise FileExistsError("PH target already exists; choose a new destination or reuse the existing cache")
    work = [path.with_suffix(".partial") for path in dirs]
    for path in work:
        path.mkdir(parents=True)
    # Figure 3 recomputes occluded maps with the same per-window PH settings.
    metadata = {
        "phc_img_hw": 224 if args.dataset == "crc" else 128, "phc_threshold": args.threshold,
        "phc_dilation_kernel": 2, "phc_dilation_iterations": 1, "vec_res": 20,
        "scales": [(width, width) for width in WINDOWS], "persist_type": "alpha", "vec_type": "PI",
        "max_hw": None, "max_windows_per_scale": 256, "full_windows_only": True, "seed": 42,
        "use_phc_fpn": not concat, "fpn_upsample_mode": "nearest", "fpn_smooth": False,
        "map_align_to_l0": True,
    }
    for level, path in enumerate(work):
        np.save(path / "meta.npy", [json.dumps(dict(metadata, level_id=level))])
    vectors = {}
    grid = 14 if args.dataset == "crc" else 8
    for name, index in splits.items():
        paths = images[index]
        sizes = (len(index), 400, grid, grid)
        arrays = [np.lib.format.open_memmap(d / f"{name}_phc_map_level.npy", mode="w+", dtype="float32", shape=sizes)
                  for d in work]
        for item in work:
            np.save(item / f"{name}_files.npy", paths)
        pooled = np.empty((len(index), pool_levels * 400), dtype=np.float32)
        for i, path in enumerate(tqdm(paths, desc=name)):
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise OSError(path)
            maps, pooled[i] = extract(gray, args.dataset, args.threshold, pool_levels, concat)
            for k, arr in enumerate(arrays):
                arr[i] = maps[k]
        for array in arrays:
            array.flush()
        vectors.update({f"{name}_files": paths, f"{name}_labels": np.asarray(classes)[y[index]], f"{name}_phc": pooled})
    output = Path(pool)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(".partial.npz")
    np.savez_compressed(pending, **vectors, phc_threshold=args.threshold)
    for source, target in zip(work, dirs):
        source.rename(target)
    pending.rename(output)


if __name__ == "__main__":
    main()
