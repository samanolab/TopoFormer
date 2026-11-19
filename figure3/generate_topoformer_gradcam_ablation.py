"""Generate true-class Grad-CAMs for the fixed panels via gradcam.py."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


CLASSES = {
    "deephisto": ("astro", "gbm", "necro", "normal", "oligo"),
    "crc": ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"),
}

ABLATIONS = ("none", "noConvNext", "noDNFD")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=ABLATIONS, required=True)
    p.add_argument("--dataset", choices=CLASSES, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--threshold", type=int, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dinov3-weight", type=Path, required=True)
    p.add_argument("--convnext-weight", type=Path, required=True)
    p.add_argument("--ph-base", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--target-block", type=int, default=-1)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    classes = CLASSES[args.dataset]
    display_dataset = "DeepHisto" if args.dataset == "deephisto" else "CRC"

    from figure3.bridge import config, load_gated, test_maps, canonical
    cfg = config(args.config)

    with args.manifest.open(newline="", encoding="utf-8") as f:
        selected = [r for r in csv.DictReader(f) if r["Dataset"] == display_dataset]
    if not selected:
        raise ValueError("No rows matched the requested dataset")

    ph_by_level = {}
    paths_ref = None
    for i in range(4):
        paths, maps, meta = test_maps(args.ph_base, i, cfg)
        if int(meta["phc_threshold"]) != args.threshold:
            raise ValueError("PH cache threshold disagrees with the selected run")
        if paths_ref is None:
            paths_ref = paths
        elif not np.array_equal(paths, paths_ref):
            raise ValueError(f"PH L{i} sample order differs")
        if maps.shape[0] != len(paths) or maps.shape[1] != 400:
            raise ValueError(f"PH L{i} shape mismatch")
        ph_by_level[f"L{i}"] = maps
    assert paths_ref is not None
    index = {path: i for i, path in enumerate(paths_ref)}
    if len(index) != len(paths_ref) or any(canonical(r["ImagePath"], cfg) not in index for r in selected):
        raise ValueError("Selected images missing/duplicated in PH cache")
    print(f"PH_READY {display_dataset} {len(paths_ref)} test paths", flush=True)

    from topoformer.data.image_utils import read_image_any, to_01_3chw_224
    model = load_gated(cfg, "briancancer" if args.dataset == "deephisto" else "crc",
                       args.threshold, args.variant, args.checkpoint, classes,
                       ph_by_level["L0"].shape)
    state_count = len(model.state_dict())

    mean = torch.tensor((.485, .456, .406), dtype=torch.float32)[:, None, None]
    std = torch.tensor((.229, .224, .225), dtype=torch.float32)[:, None, None]
    results = []
    for row in selected:
        image_path = Path(canonical(row["ImagePath"], cfg))
        if "TrueClassID" in row and row["TrueClassID"]:
            truth = int(row["TrueClassID"])
            truth_check_label = row["OriginalLabel"]
        else:
            truth = classes.index(row["Class"])
            truth_check_label = row["Class"]
        if classes[truth] != truth_check_label or sha256(image_path) != row["ImageSHA256"]:
            raise ValueError(f"Image/manifest mismatch: {image_path}")
        image = read_image_any(str(image_path))
        with Image.open(image_path) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        x = torch.from_numpy(to_01_3chw_224(image)).float()
        x = ((x - mean) / std).half().unsqueeze(0).cuda()
        x.requires_grad_(True)
        j = index[str(image_path)]
        topo = {k: torch.from_numpy(np.nan_to_num(v[j])).half().unsqueeze(0).cuda()
                for k, v in ph_by_level.items()}
        captured = {}

        def hook(_module, _inputs, output):
            captured["activation"] = output
            output.register_hook(lambda grad: captured.update(gradient=grad))

        handle = model.vit.encoder.blocks[args.target_block].norm1.register_forward_hook(hook)
        out = model(x, grl_lambda=0.0, topo_levels=topo)
        logits = out["logitsA"]
        pred = int(logits.argmax(1).item())
        prob = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()[0]
        logits[0, truth].backward()
        handle.remove()
        start = 1 + model.vit.encoder.n_storage_tokens
        act = captured["activation"].detach().float()[0, start:]
        grad = captured["gradient"].detach().float()[0, start:]
        if act.shape != (196, 4096) or grad.shape != act.shape:
            raise ValueError(f"Invalid spatial tokens: {act.shape}, {grad.shape}")
        grad_norm = float(grad.norm().item())
        if not torch.isfinite(grad).all() or grad_norm == 0:
            raise ValueError(f"Nonfinite/zero gradient: {row['SampleID']}")
        signed = (act @ grad.mean(0)).reshape(14, 14)
        cam = torch.relu(signed)
        status = "ok"
        if not torch.isfinite(cam).all() or cam.max().item() == 0:
            status = "zero_cam"
        stem = (row.get("SampleID") or f"{truth_check_label}__{Path(row['ImagePath']).name}").replace("/", "__").replace(".", "_")
        sample_dir = args.out / display_dataset / stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        if status == "ok":
            cam = (cam / cam.max()).cpu().numpy()
            aligned = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            np.savez_compressed(sample_dir / "cam.npz", cam_lowres=cam, cam_aligned=aligned,
                                logits=logits.detach().float().cpu().numpy()[0], probabilities=prob,
                                target_class=np.int64(truth), image_path=np.str_(str(image_path)),
                                image_sha256=np.str_(row["ImageSHA256"]))
            heat = cv2.cvtColor(cv2.applyColorMap((aligned * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
            Image.fromarray(np.clip(.55 * rgb + .45 * heat, 0, 255).astype(np.uint8)).save(sample_dir / "overlay.png")
        record = {
            "status": status, "model": f"TopoFormer-{args.variant}", "variant": args.variant,
            "dataset": display_dataset, "class": row.get("OriginalLabel", row.get("Class", "?")),
            "sample_id": row.get("SampleID", f"{row.get('Class', '?')}/{Path(row['ImagePath']).name}"),
            "image_sha256": row["ImageSHA256"], "target_class": classes[truth],
            "original_prediction": row.get("TopoFormerPrediction"),
            "predicted_class": classes[pred], "target_layer": f"model.vit.encoder.blocks[{args.target_block}].norm1",
            "branch_scope": "post-TopoCTI visual stream; final score also includes ConvNeXt fusion (post-ablation)",
            "patch_grid": [14, 14], "gradient_norm": grad_norm,
            "signed_cam_min": float(signed.min().item()), "signed_cam_max": float(signed.max().item()),
            "checkpoint_path": str(args.checkpoint), "checkpoint_state_keys": state_count,
            "ph_base": str(args.ph_base), "true_class_probability": float(prob[truth]),
        }
        (sample_dir / "report.json").write_text(json.dumps(record, indent=2) + "\n")
        results.append(record)
        print(f"TOPO_CAM {display_dataset} {args.variant} {row.get('OriginalLabel', row.get('Class', '?'))} {status} grad={grad_norm:.6g}", flush=True)
    summary = {
        "model": f"TopoFormer-{args.variant}", "variant": args.variant, "dataset": display_dataset,
        "target_block": args.target_block, "manifest_sha256": sha256(args.manifest),
        "total": len(results),
        "ok": sum(r["status"] == "ok" for r in results),
        "status_counts": {s: sum(r["status"] == s for r in results) for s in sorted(set(r["status"] for r in results))},
        "results": results,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("variant", "dataset", "total", "ok", "status_counts")}), flush=True)


if __name__ == "__main__":
    main()
