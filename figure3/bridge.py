import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiment import build_model, check_run, load_checkpoint
from materials import canonical, map_paths, read_config
from topoformer.data.topo_maps import open_mmap_rows

VARIANTS = {"none": "full", "noConvNext": "no-convnext", "noDNFD": "no-dnfd"}


def config(path):
    return read_config(path)


def test_maps(base, level, cfg):
    source = Path(f"{base}_L{level}.npz")
    mapped = source.with_suffix(".mmap")
    if mapped.is_dir():
        maps = open_mmap_rows(mapped, "test")
        files = np.load(mapped / "test_files.npy", allow_pickle=False)
        if (mapped / "meta.npy").is_file():
            meta = np.load(mapped / "meta.npy", allow_pickle=False)
        else:
            with np.load(source, allow_pickle=False) as cache:
                meta = cache["meta"]
    else:
        with np.load(source, allow_pickle=False) as cache:
            meta = cache["meta"]
            files = cache["test_files"]
            maps = cache["test_phc_map_level"].astype(np.float32, copy=False)
    metadata = json.loads(str(meta[0]))
    paths = map_paths(files.astype(str), cfg)
    if len(maps) != len(paths) or maps.shape[1] != 400:
        raise ValueError(f"Invalid map shape for {source}")
    return paths, maps, metadata


def load_gated(cfg, dataset, threshold, legacy_variant, checkpoint, classes, maps_shape):
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    variant = VARIANTS[legacy_variant]
    run = checkpoint.parent
    check_run(run, dataset, threshold, 42, variant)
    if checkpoint.name != "best_val.pt":
        raise ValueError("Expected the selected best_val.pt")
    store = SimpleNamespace(topo_c=maps_shape[1], topo_hw=maps_shape[-2:])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_model(cfg, variant, store, classes, torch.device("cuda:0"))
    load_checkpoint(model, run, cfg)
    model.requires_grad_(False)
    return model
