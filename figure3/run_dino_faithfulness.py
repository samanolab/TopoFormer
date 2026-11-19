"""Grad-CAM guided DeepHisto faithfulness using the newly fitted DINO heads.

Checkpoint, test images and tissue masks are read only. Each sample is written
atomically so a nohup run can resume without recomputing completed samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from faithfulness_occlusion import (BLUR_KERNEL, BLUR_SIGMA, GRID, apply_blur,
                                     make_masks, tissue_enrichment)


CLASSES = ("astro", "gbm", "necro", "normal", "oligo")
RATIOS = (0.05, 0.10, 0.20)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess(rgb: np.ndarray, model_name: str, device: torch.device) -> torch.Tensor:
    x = rgb.astype(np.float32)
    if model_name == "dinov2":
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    else:
        x = x / 255.0
    x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
    mean = torch.tensor((.485, .456, .406), dtype=torch.float32)[:, None, None]
    std = torch.tensor((.229, .224, .225), dtype=torch.float32)[:, None, None]
    return ((x - mean) / std).to(device)


def score_batch(images: list[np.ndarray], model, name: str, weight: torch.Tensor,
                bias: torch.Tensor, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    logits_all = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = torch.stack([preprocess(rgb, name, weight.device)
                             for rgb in images[i:i + batch_size]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feature = model.forward_features(x)["x_norm_clstoken"]
                logits = feature.float() @ weight.T + bias
            logits_all.append(logits.float().cpu().numpy())
    logits_array = np.concatenate(logits_all)
    shifted = logits_array - logits_array.max(axis=1, keepdims=True)
    exponents = np.exp(shifted)
    probability = exponents / exponents.sum(axis=1, keepdims=True)
    return logits_array, probability


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("dinov2", "dinov3"), required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--mask-manifest", type=Path, required=True)
    p.add_argument("--config", type=Path, default="config.local.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    from materials import canonical, read_config
    config = read_config(a.config)
    if a.batch_size < 1:
        raise ValueError("batch-size must be positive")
    a.out.mkdir(parents=True, exist_ok=True)
    with a.mask_manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 200 or sorted({r["Class"] for r in rows}) != sorted(CLASSES):
        raise ValueError("Fixed 200-sample DeepHisto manifest mismatch")
    if a.limit:
        rows = rows[:a.limit]
    protocol = {
        "status": "frozen_model_new_logistic_head",
        "model": a.model, "checkpoint": str(a.weights), "checkpoint_sha256": sha256(a.weights),
        "head": str(a.head), "head_sha256": sha256(a.head),
        "mask_manifest": str(a.mask_manifest), "mask_manifest_sha256": sha256(a.mask_manifest),
        "sample_count": len(rows), "ratios": RATIOS, "random_tissue_repeats": 5,
        "target_layer": "blocks[-1].norm1" if a.model == "dinov2" else "blocks[-2].norm1",
        "target": "true class", "grid_pixels": GRID,
        "blur_kernel": BLUR_KERNEL, "blur_sigma": BLUR_SIGMA,
        "occlusion_space": "original RGB pixel grid, before backbone preprocessing",
        "background_rule": "N/A if background pixels < selected tissue pixel count",
        "resume_rule": "existing sample JSON with matching protocol hash is not recomputed",
    }
    protocol_path = a.out / "protocol.json"
    protocol_bytes = (json.dumps(protocol, indent=2) + "\n").encode("utf-8")
    if protocol_path.exists() and protocol_path.read_bytes() != protocol_bytes:
        raise ValueError("Output directory already contains a different protocol")
    protocol_path.write_bytes(protocol_bytes)
    protocol_hash = sha256(protocol_path)
    with np.load(a.head, allow_pickle=False) as data:
        if tuple(data["classes"].tolist()) != CLASSES:
            raise ValueError("Head class order mismatch")
        weight = torch.from_numpy(data["weight_input"].astype(np.float32)).cuda()
        bias = torch.from_numpy(data["bias_input"].astype(np.float32)).cuda()
    if a.model == "dinov2":
        model = torch.hub.load(str(a.repo), "dinov2_vitb14", source="local",
                               pretrained=False, trust_repo=True)
        model.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True), strict=True)
        extra, grid, dim, target = 1, 16, 768, -1
    else:
        sys.path.insert(0, str(a.repo))
        from dinov3.hub.backbones import dinov3_vit7b16
        model = dinov3_vit7b16(weights=str(a.weights))
        extra, grid, dim, target = 1 + model.n_storage_tokens, 14, 4096, -2
    model = model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(f"READY {a.model} samples={len(rows)} grid={grid}x{grid}", flush=True)
    for position, row in enumerate(rows, 1):
        sample_id = f"{row['Class']}/{row['SampleID']}"
        sample_dir = a.out / row["Class"] / row["SampleID"].replace(".", "_")
        sample_dir.mkdir(parents=True, exist_ok=True)
        result_path = sample_dir / "result.json"
        if result_path.exists():
            old = json.loads(result_path.read_text(encoding="utf-8"))
            if old.get("protocol_sha256") == protocol_hash and old.get("status") in ("ok", "zero_cam"):
                print(f"RESUME {position}/{len(rows)} {sample_id}", flush=True)
                continue
            raise ValueError(f"Existing incompatible result: {result_path}")
        image_path = Path(canonical(row["ImagePath"], config))
        mask_path = a.mask_manifest.parent / row["MaskPath"]
        if sha256(image_path) != row["ImageSHA256"] or sha256(mask_path) != row["MaskSHA256"]:
            raise ValueError(f"Image/mask hash mismatch: {sample_id}")
        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        tissue = np.asarray(Image.open(mask_path).convert("L")) > 0
        if rgb.shape[:2] != tissue.shape or row["Class"] != image_path.parent.name:
            raise ValueError(f"Image/mask/class alignment mismatch: {sample_id}")
        truth = CLASSES.index(row["Class"])
        captured = {}

        def hook(_module, _inputs, output):
            captured["activation"] = output
            output.register_hook(lambda grad: captured.update(gradient=grad))

        handle = model.blocks[target].norm1.register_forward_hook(hook)
        x = preprocess(rgb, a.model, weight.device).unsqueeze(0).requires_grad_(True)
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feature = model.forward_features(x)["x_norm_clstoken"]
            logits = feature.float() @ weight.T + bias
        original_logits = logits.detach().float().cpu().numpy()[0]
        original_probs = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()[0]
        logits[0, truth].backward()
        handle.remove()
        act = captured["activation"].detach().float()[0, extra:]
        grad = captured["gradient"].detach().float()[0, extra:]
        if act.shape != (grid * grid, dim) or grad.shape != act.shape or not torch.isfinite(grad).all():
            raise ValueError(f"Invalid activation or gradient: {sample_id}")
        signed = (act @ grad.mean(0)).reshape(grid, grid)
        positive = torch.relu(signed)
        record = {
            "model": a.model, "sample_id": sample_id, "true_label": row["Class"],
            "predicted_label": CLASSES[int(original_logits.argmax())],
            "checkpoint_sha256": protocol["checkpoint_sha256"],
            "head_sha256": protocol["head_sha256"],
            "image_path": str(image_path), "image_sha256": row["ImageSHA256"],
            "mask_path": str(mask_path), "mask_sha256": row["MaskSHA256"],
            "protocol_sha256": protocol_hash, "target_class": truth,
            "tissue_area_ratio": float(tissue.mean()),
            "original_probability": float(original_probs[truth]),
            "original_logit": float(original_logits[truth]),
            "gradient_norm": float(grad.norm().item()),
            "signed_cam_max": float(signed.max().item()),
            "ratios": {},
        }
        if grad.norm().item() == 0 or not torch.isfinite(positive).all() or positive.max().item() <= 0:
            record["status"] = "zero_cam"
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
            condition = [("high", 0, masks["high"])]
            condition += [("random_tissue", j, mask) for j, mask in enumerate(masks["random_tissue"])]
            if masks["random_background"] is not None:
                condition.append(("random_background", 0, masks["random_background"]))
            masked_images = [apply_blur(rgb, mask) for _, _, mask in condition]
            occluded_logits, occluded_probs = score_batch(masked_images, model, a.model,
                                                            weight, bias, a.batch_size)
            rows_out = []
            mask_arrays = {}
            for i, (name, repeat, mask) in enumerate(condition):
                key = f"{name}_{repeat}"
                mask_arrays[key] = np.packbits(mask.ravel())
                rows_out.append({
                    "condition": name, "repeat": repeat, "selected_pixels": int(mask.sum()),
                    "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                    "true_class_probability": float(occluded_probs[i, truth]),
                    "true_class_logit": float(occluded_logits[i, truth]),
                })
            np.savez_compressed(sample_dir / f"masks_{ratio:.2f}.npz",
                                image_shape=np.asarray(tissue.shape, dtype=np.int64),
                                selected_pixels=np.int64(masks["target_pixels"]), **mask_arrays)
            high = rows_out[0]
            random = rows_out[1:6]
            if any(r["selected_pixels"] != masks["target_pixels"] for r in rows_out):
                raise RuntimeError(f"Equal-area QC failed: {sample_id}, {ratio}")
            p0, l0 = record["original_probability"], record["original_logit"]
            high_drop = p0 - high["true_class_probability"]
            random_drop = float(np.mean([p0 - r["true_class_probability"] for r in random]))
            record["ratios"][f"{ratio:.2f}"] = {
                "target_pixels": masks["target_pixels"], "conditions": rows_out,
                "high_drop_probability": high_drop,
                "random_drop_probability": random_drop,
                "faithfulness_gap_probability": high_drop - random_drop,
                "high_drop_logit": l0 - high["true_class_logit"],
                "random_drop_logit": float(np.mean([l0 - r["true_class_logit"] for r in random])),
                "random_background_available": masks["random_background"] is not None,
            }
        record["status"] = "ok"
        write_json_atomic(result_path, record)
        print(f"DONE {position}/{len(rows)} {sample_id}", flush=True)


if __name__ == "__main__":
    main()
