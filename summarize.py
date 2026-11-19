import argparse
import json
from pathlib import Path

import numpy as np

from jobs import full_path, results_seeds, variant_path
from materials import read_config
from report import write_json


def aggregate(files, metric_group):
    if not all(file.is_file() for file in files):
        return None
    data = [json.loads(file.read_text(encoding="utf-8"))[metric_group] for file in files]
    return {name: {"mean": float(np.mean([row[name] for row in data])),
                   "sample_sd": float(np.std([row[name] for row in data], ddof=1))}
            for name in data[0]}


def main():
    parser = argparse.ArgumentParser(description="Generate table rows from available three-seed results")
    parser.add_argument("--config", default="config.local.json")
    args = parser.parse_args()
    cfg = read_config(args.config)
    root = Path(cfg["run_root"])
    chosen = json.loads((root / "selected_thresholds.json").read_text(encoding="utf-8"))
    tables = {}
    for ds, seeds in results_seeds(chosen).items():
        threshold = chosen[ds]["threshold"]
        full = [full_path(cfg, ds, threshold, seed) for seed in seeds]
        output = tables[ds] = {}
        def record(name, folders, key="primary", condition="test"):
            values = aggregate([folder / "scores" / condition / "metrics.json" for folder in folders], key)
            if values is not None:
                output[name] = values
        def runs(kind):
            return [variant_path(cfg, ds, threshold, seed, kind) for seed in seeds]
        record("I / IV Full", full)
        record("IV Full - Global Vector", full, "decoder_a")
        record("IV DINOv3 + DNFD + TopoCTI", runs("dino-dnfd-cti"))
        record("IV Full - ConvNeXt", runs("no-convnext"))
        record("IV DINOv3 + TopoCTI", runs("dino-cti"))
        record("IV Full - DNFD", runs("no-dnfd"))
        record("IV TopoFormer - TopoCTI", runs("no-cti"))
        for condition in ("stain_shift", "patch_shuffle"):
            record(f"II {condition}", full, condition=condition)
        if ds == "briancancer":
            record("S1 Decoder A", full, "decoder_a")
            record("S1 Decoder B", full, "decoder_b")
            for kind in ("L0", "L1", "L2", "L3", "simpleconcat"):
                record(f"S4 {kind}", runs(kind))
            for condition in ("external_raw", "external_reinhard"):
                record(f"III {condition}", runs("external"), condition=condition)
    write_json(root / "table_rows.json", tables)


if __name__ == "__main__":
    main()
