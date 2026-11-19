"""Full frozen TopoFormer faithfulness with fresh PH for every occluded RGB.

Original PH comes from its verified cache. Every blurred image is written only
inside this experiment's scratch directory, processed by the original PH
functions with the original sample seed, and passed with the blurred RGB.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from figure3.audit_ph_occlusion_recompute import _load_original_functions
from faithfulness_occlusion import (BLUR_KERNEL, BLUR_SIGMA, GRID, apply_blur,
                                     make_masks, tissue_enrichment)
from figure3.run_dino_faithfulness import CLASSES, RATIOS, sha256, write_json_atomic


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--threshold", type=int, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dinov3-weight", type=Path, required=True)
    p.add_argument("--convnext-weight", type=Path, required=True)
    p.add_argument("--ph-base", type=Path, required=True)
    p.add_argument("--ph-source", type=Path, required=True)
    p.add_argument("--mask-manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start-index", type=int, default=1,
                   help="Inclusive one-based manifest position for this worker")
    p.add_argument("--end-index", type=int, default=200,
                   help="Inclusive one-based manifest position for this worker")
    p.add_argument("--expected-count", type=int, default=200,
                   help="Total expected number of manifest rows; 200 for DeepHisto, 198 for CRC.")
    p.add_argument("--variant", choices=("none", "noConvNext", "noDNFD"),
                   default="none",
                   help="Ablation variant. 'none' = full TopoFormer (default).")
    p.add_argument("--target-layer", type=str, default="model.vit.encoder.blocks[-2].norm1",
                   help="Hook target layer for Grad-CAM. CRC uses blocks[-5].norm1.")
    a = p.parse_args()
    from figure3.bridge import config, load_gated, test_maps, canonical
    cfg = config(a.config)
    if not 1 <= a.start_index <= a.end_index <= a.expected_count:
        raise ValueError(f"Worker range must lie within manifest positions 1..{a.expected_count}")
    a.out.mkdir(parents=True, exist_ok=True)
    with a.mask_manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != a.expected_count:
        raise ValueError(f"Expected {a.expected_count} fixed test images, found {len(rows)}")
    if a.limit:
        rows = rows[:a.limit]
    ph_maps = {}
    cache_paths = None
    meta = None
    for level in range(4):
        paths, ph_maps[f"L{level}"], level_meta = test_maps(a.ph_base, level, cfg)
        if level == 0:
            meta = level_meta
        if cache_paths is None:
            cache_paths = paths
        elif not np.array_equal(cache_paths, paths):
            raise ValueError("PH cache level path order differs")
    assert cache_paths is not None and meta is not None
    path_index = {path: i for i, path in enumerate(cache_paths)}
    if len(path_index) != len(cache_paths):
        raise ValueError("Duplicate PH cache paths")
    if int(meta["phc_threshold"]) != a.threshold:
        raise ValueError("PH cache threshold disagrees with the selected run")
    original_ph = _load_original_functions(a.ph_source, meta)
    protocol = {
        "model": ("TopoFormer" if a.variant == "none" else f"TopoFormer-{a.variant}"),
        "variant": a.variant,
        "checkpoint": str(a.checkpoint),
        "checkpoint_sha256": sha256(a.checkpoint),
        "dino_weight": str(a.dinov3_weight), "convnext_weight": str(a.convnext_weight),
        "ph_cache_base": str(a.ph_base), "ph_source": str(a.ph_source),
        "ph_source_sha256": original_ph["source_sha256"], "ph_metadata": meta,
        "mask_manifest": str(a.mask_manifest), "mask_manifest_sha256": sha256(a.mask_manifest),
        "sample_count": len(rows), "ratios": RATIOS, "random_tissue_repeats": 5,
        "target_layer": a.target_layer,
        "cam_scope": "post-topology visual stream; final logits also include ConvNeXt fusion",
        "target": "true class", "grid_pixels": GRID,
        "blur_kernel": BLUR_KERNEL, "blur_sigma": BLUR_SIGMA,
        "occluded_ph": "recomputed from each blurred RGB using original PH source and original image seed",
    }
    protocol_path = a.out / "protocol.json"
    serialized = (json.dumps(protocol, indent=2, default=str) + "\n").encode()
    if protocol_path.exists() and protocol_path.read_bytes() != serialized:
        raise ValueError("Output directory contains incompatible protocol")
    protocol_path.write_bytes(serialized)
    protocol_hash = sha256(protocol_path)
    from topoformer.data.image_utils import read_image_any, to_01_3chw_224
    model = load_gated(cfg, "briancancer", a.threshold, a.variant, a.checkpoint, CLASSES,
                       ph_maps["L0"].shape)
    mean = torch.tensor((.485, .456, .406), dtype=torch.float32)[:, None, None]
    std = torch.tensor((.229, .224, .225), dtype=torch.float32)[:, None, None]
    print(f"READY TopoFormer samples={len(rows)} ph_cache_paths={len(cache_paths)}", flush=True)
    for position, row in enumerate(rows, 1):
        if position < a.start_index or position > a.end_index:
            continue
        tic = time.perf_counter()
        sample_id = f"{row['Class']}/{row['SampleID']}"
        sample_dir = a.out / row["Class"] / row["SampleID"].replace(".", "_")
        sample_dir.mkdir(parents=True, exist_ok=True)
        result_path = sample_dir / "result.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous.get("protocol_sha256") == protocol_hash and previous.get("status") in ("ok", "zero_cam"):
                print(f"RESUME {position}/{len(rows)} {sample_id}", flush=True)
                continue
            raise ValueError(f"Existing incompatible result: {result_path}")
        image_path = Path(canonical(row["ImagePath"], cfg))
        mask_path = a.mask_manifest.parent / row["MaskPath"]
        if str(image_path) not in path_index or sha256(image_path) != row["ImageSHA256"] or sha256(mask_path) != row["MaskSHA256"]:
            raise ValueError(f"Image/mask/cache provenance mismatch: {sample_id}")
        index = path_index[str(image_path)]
        ph_seed = int(meta["seed"]) + 10_000 + index + 1
        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        tissue = np.asarray(Image.open(mask_path).convert("L")) > 0
        if tissue.shape != rgb.shape[:2] or image_path.parent.name != row["Class"]:
            raise ValueError(f"Image/mask/class mismatch: {sample_id}")
        image = read_image_any(str(image_path))
        x = torch.from_numpy(to_01_3chw_224(image)).float()
        x = ((x - mean) / std).half().unsqueeze(0).cuda().requires_grad_(True)
        cached_topo = {key: torch.from_numpy(np.nan_to_num(values[index])).half().unsqueeze(0).cuda()
                       for key, values in ph_maps.items()}
        truth = CLASSES.index(row["Class"])
        captured = {}

        def hook(_module, _inputs, output):
            captured["activation"] = output
            output.register_hook(lambda gradient: captured.update(gradient=gradient))

        # Resolve --target-layer to a live nn.Module via dotted traversal.
        target_module = model
        for part in a.target_layer.removeprefix("model.").split("."):
            if part.startswith("blocks[") and part.endswith("]"):
                target_module = target_module.blocks[int(part[len("blocks["):-1])]
                continue
            target_module = getattr(target_module, part)
        handle = target_module.register_forward_hook(hook)
        output = model(x, grl_lambda=0.0, topo_levels=cached_topo)
        logits = output["logitsA"]
        original_logits = logits.detach().float().cpu().numpy()[0]
        original_probs = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()[0]
        logits[0, truth].backward()
        handle.remove()
        start = 1 + model.vit.encoder.n_storage_tokens
        act = captured["activation"].detach().float()[0, start:]
        grad = captured["gradient"].detach().float()[0, start:]
        if act.shape != (196, 4096) or grad.shape != act.shape or not torch.isfinite(grad).all():
            raise ValueError(f"Bad spatial tokens/gradient: {sample_id}")
        signed = (act @ grad.mean(0)).reshape(14, 14)
        positive = torch.relu(signed)
        record = {
            "model": "TopoFormer", "sample_id": sample_id, "true_label": row["Class"],
            "predicted_label": CLASSES[int(original_logits.argmax())],
            "checkpoint_sha256": protocol["checkpoint_sha256"],
            "ph_source_sha256": protocol["ph_source_sha256"], "ph_seed": ph_seed,
            "image_path": str(image_path), "image_sha256": row["ImageSHA256"],
            "mask_path": str(mask_path), "mask_sha256": row["MaskSHA256"],
            "protocol_sha256": protocol_hash, "target_class": truth,
            "tissue_area_ratio": float(tissue.mean()),
            "original_probability": float(original_probs[truth]),
            "original_logit": float(original_logits[truth]),
            "gradient_norm": float(grad.norm().item()),
            "signed_cam_max": float(signed.max().item()), "ratios": {},
        }
        if grad.norm().item() == 0 or not torch.isfinite(positive).all() or positive.max().item() <= 0:
            record["status"] = "zero_cam"
            record["elapsed_seconds"] = time.perf_counter() - tic
            write_json_atomic(result_path, record)
            print(f"ZERO_CAM {position}/{len(rows)} {sample_id}", flush=True)
            continue
        cam = (positive / positive.max()).cpu().numpy()
        aligned = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        np.savez_compressed(sample_dir / "cam.npz", cam_lowres=cam, cam_aligned=aligned,
                            image_sha256=np.str_(row["ImageSHA256"]), target_class=np.int64(truth))
        record["tissue_enrichment"] = tissue_enrichment(aligned, tissue)
        for ratio in RATIOS:
            masks = make_masks(tissue, aligned, ratio, sample_id)
            conditions = [("high", 0, masks["high"])]
            conditions += [("random_tissue", j, mask) for j, mask in enumerate(masks["random_tissue"])]
            if masks["random_background"] is not None:
                conditions.append(("random_background", 0, masks["random_background"]))
            occluded_logits = []
            occluded_probs = []
            packed = {}
            ph_saved = {}
            scratch = sample_dir / "occluded_rgb_scratch.png"
            for name, repeat, mask in conditions:
                packed[f"{name}_{repeat}"] = np.packbits(mask.ravel())
                masked_rgb = apply_blur(rgb, mask)
                if not cv2.imwrite(str(scratch), cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"Could not write isolated scratch RGB: {scratch}")
                computed = original_ph["_extract_one"](str(scratch), seed=ph_seed)[2]
                if computed is None or len(computed) != 4:
                    raise ValueError(f"Original PH source did not return four levels: {sample_id}")
                topo = {}
                for level, ph_value in enumerate(computed):
                    arr = np.asarray(ph_value, dtype=np.float32)
                    if arr.shape != ph_maps[f"L{level}"][index].shape or not np.isfinite(arr).all():
                        raise ValueError(f"Invalid recomputed PH level: {sample_id} L{level}")
                    topo[f"L{level}"] = torch.from_numpy(arr).half().unsqueeze(0).cuda()
                    ph_saved[f"{name}_{repeat}_L{level}"] = arr
                visual = read_image_any(str(scratch))
                inp = torch.from_numpy(to_01_3chw_224(visual)).float()
                inp = ((inp - mean) / std).half().unsqueeze(0).cuda()
                with torch.no_grad():
                    output = model(inp, grl_lambda=0.0, topo_levels=topo)
                    score = output["logitsA"].float()
                occluded_logits.append(float(score[0, truth].item()))
                occluded_probs.append(float(torch.softmax(score, dim=1)[0, truth].item()))
            scratch.unlink(missing_ok=True)
            np.savez_compressed(sample_dir / f"masks_{ratio:.2f}.npz",
                                image_shape=np.asarray(tissue.shape, dtype=np.int64),
                                selected_pixels=np.int64(masks["target_pixels"]), **packed)
            np.savez_compressed(sample_dir / f"ph_{ratio:.2f}.npz", **ph_saved)
            raw = []
            for i, (name, repeat, mask) in enumerate(conditions):
                raw.append({"condition": name, "repeat": repeat,
                            "selected_pixels": int(mask.sum()),
                            "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                            "true_class_probability": occluded_probs[i],
                            "true_class_logit": occluded_logits[i]})
            if any(item["selected_pixels"] != masks["target_pixels"] for item in raw):
                raise RuntimeError("Equal area QC failed")
            p0, l0 = record["original_probability"], record["original_logit"]
            high_drop = p0 - raw[0]["true_class_probability"]
            random_drop = float(np.mean([p0 - item["true_class_probability"] for item in raw[1:6]]))
            record["ratios"][f"{ratio:.2f}"] = {
                "target_pixels": masks["target_pixels"], "conditions": raw,
                "high_drop_probability": high_drop, "random_drop_probability": random_drop,
                "faithfulness_gap_probability": high_drop - random_drop,
                "high_drop_logit": l0 - raw[0]["true_class_logit"],
                "random_drop_logit": float(np.mean([l0 - item["true_class_logit"] for item in raw[1:6]])),
                "random_background_available": masks["random_background"] is not None,
            }
        record["status"] = "ok"
        record["elapsed_seconds"] = time.perf_counter() - tic
        write_json_atomic(result_path, record)
        print(f"DONE {position}/{len(rows)} {sample_id} seconds={record['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
