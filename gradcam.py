import argparse
import csv
import hashlib
from pathlib import Path

from figure3 import summarize_faithfulness as dh_summary
from figure3 import summarize_faithfulness_crc as crc_summary
from jobs import call
from materials import asset, canonical, read_config
from report import write_json

ROOT = Path(__file__).resolve().parent
PANELS_SHA256 = "1e5d6f00b2c11898b8fefd7476e9c321f6b27452161594277bcebc5d19a87b93"
VARIANT = {"full": "none", "no-convnext": "noConvNext", "no-dnfd": "noDNFD"}


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def check_inputs(cfg, dataset):
    data = cfg["datasets"][dataset]
    panel_path = Path(data["panels"])
    if hashlib.sha256(panel_path.read_bytes()).hexdigest() != PANELS_SHA256:
        raise ValueError("Expected the distributed fixed-panel manifest")
    name = "CRC" if dataset == "crc" else "DeepHisto"
    displayed = [r for r in rows(panel_path) if r["Dataset"] == name]
    mask_path = Path(data["faithfulness"])
    cohort = rows(mask_path)
    expected = 198 if dataset == "crc" else 200
    if len(displayed) != 5 or len(cohort) != expected:
        raise ValueError(f"Incomplete fixed Figure 3 cohort for {dataset}")
    for entry in [*displayed, *cohort]:
        path = Path(canonical(entry["ImagePath"], cfg))
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["ImageSHA256"]:
            raise ValueError(f"Changed original patch: {path}")
    for entry in cohort:
        path = mask_path.parent / entry["MaskPath"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["MaskSHA256"]:
            raise ValueError(f"Changed original tissue mask: {path}")
    print(f"FIG3_FIXED_INPUTS_PASS {dataset} panels=5 faithfulness={expected}")


def main():
    parser = argparse.ArgumentParser(description="Run the original Figure 3 scripts with the gated checkpoint")
    parser.add_argument("dataset", choices=("briancancer", "crc"))
    parser.add_argument("threshold", type=int, choices=(160, 180, 200))
    parser.add_argument("variant", choices=VARIANT)
    parser.add_argument("run", type=Path)
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument("--check-inputs", action="store_true", help="Verify fixed images and tissue masks on CPU")
    args = parser.parse_args()
    cfg = read_config(args.config)
    check_inputs(cfg, args.dataset)
    if args.check_inputs:
        return
    data = cfg["datasets"][args.dataset]
    crc = args.dataset == "crc"
    target = "-5" if crc else "-2"
    output = args.run / "figure3"
    common = (
        "--config", args.config, "--checkpoint", args.run / "best_val.pt",
        "--threshold", args.threshold, "--variant", VARIANT[args.variant],
        "--dinov3-weight", cfg["dinov3"], "--convnext-weight", cfg["convnext"],
        "--ph-base", asset(data["maps"], args.threshold),
    )
    call("-m", "figure3.generate_topoformer_gradcam_ablation", *common,
         "--dataset", "crc" if crc else "deephisto", "--target-block", target,
         "--manifest", data["panels"], "--out", output / "panels")
    module = "figure3.run_crc_topo_faithfulness" if crc else "figure3.run_topo_faithfulness"
    source = "TopoFPN_layer_CRC_180.py" if crc else "TopoFPN_layer.py"
    call("-m", module, *common, "--ph-source", ROOT / "notebooks" / source,
         "--mask-manifest", data["faithfulness"], "--out", output / "faithfulness")
    original = crc_summary if crc else dh_summary
    records = original.read_run(output / "faithfulness")
    if len(records) != (198 if crc else 200):
        raise ValueError("Incomplete original faithfulness cohort")
    original.validate_repeats({args.variant: records})
    result = {ratio: original.summarize_model(args.variant, records, ratio, 2000, 2026) for ratio in original.RATIOS}
    write_json(output / "faithfulness_summary.json", result)


if __name__ == "__main__":
    main()
