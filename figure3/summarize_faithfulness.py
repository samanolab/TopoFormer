"""Render Table X/S1 from completed frozen-model, fixed-cohort occlusion runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("TopoFormer", "DINOv3", "DINOv2", "Swin")
CLASSES = ("astro", "gbm", "necro", "normal", "oligo")
RATIOS = ("0.05", "0.10", "0.20")
METRICS = ("high_drop_probability", "random_drop_probability",
           "faithfulness_gap_probability", "tissue_enrichment")


def read_run(root: Path) -> dict[str, dict]:
    files = sorted(root.glob("*/*/result.json"))
    results = {}
    for path in files:
        value = json.loads(path.read_text(encoding="utf-8"))
        sample_id = value["sample_id"]
        if sample_id in results:
            raise ValueError(f"Duplicate sample ID: {sample_id}")
        results[sample_id] = value
    return results


def validate_repeats(all_records: dict[str, dict[str, dict]]) -> dict:
    """Check every saved probability, area and common random mask before tables."""
    checked = 0
    background_counts = Counter()
    for sample_id in sorted(next(iter(all_records.values()))):
        for ratio in RATIOS:
            reference_random_hashes = None
            for model, records in all_records.items():
                row = records[sample_id]
                if row["status"] == "zero_cam":
                    continue
                if row["status"] != "ok" or ratio not in row["ratios"]:
                    raise ValueError(f"Missing complete ratios: {model}, {sample_id}")
                values = row["ratios"][ratio]
                condition = values["conditions"]
                high = [item for item in condition if item["condition"] == "high"]
                random = sorted((item for item in condition if item["condition"] == "random_tissue"),
                                key=lambda item: item["repeat"])
                background = [item for item in condition if item["condition"] == "random_background"]
                if len(high) != 1 or len(random) != 5 or [item["repeat"] for item in random] != list(range(5)):
                    raise ValueError(f"Condition/repeat count mismatch: {model}, {sample_id}")
                if len(background) != int(values["random_background_available"]):
                    raise ValueError(f"Background availability mismatch: {model}, {sample_id}")
                if any(item["selected_pixels"] != values["target_pixels"] for item in condition):
                    raise ValueError(f"Occlusion areas differ: {model}, {sample_id}")
                p0, l0 = row["original_probability"], row["original_logit"]
                drop_high = p0 - high[0]["true_class_probability"]
                drop_random = float(np.mean([p0 - item["true_class_probability"] for item in random]))
                if (abs(values["high_drop_probability"] - drop_high) > 1e-8 or
                        abs(values["random_drop_probability"] - drop_random) > 1e-8 or
                        abs(values["faithfulness_gap_probability"] - (drop_high - drop_random)) > 1e-8 or
                        abs(values["high_drop_logit"] - (l0 - high[0]["true_class_logit"])) > 1e-7):
                    raise ValueError(f"Drop arithmetic mismatch: {model}, {sample_id}")
                hashes = [item["mask_sha256"] for item in random]
                if reference_random_hashes is None:
                    reference_random_hashes = hashes
                elif reference_random_hashes != hashes:
                    raise ValueError(f"Random-tissue masks differ across models: {sample_id}, {ratio}")
                background_counts[model] += len(background)
                checked += 1
    return {"checked_model_sample_ratio_records": checked,
            "background_condition_counts": dict(background_counts),
            "common_random_masks_across_models": len(all_records) > 1,
            "equal_area_and_drop_arithmetic": True}


def slide_id(sample_id: str) -> str:
    dataset_class, filename = sample_id.split("/", 1)
    return dataset_class + "/" + filename.split(".", 1)[0]


def bootstrap_indices(ids: list[str], rng: np.random.Generator) -> list[int]:
    """Stratified hierarchical WSI/patch bootstrap, retaining five classes."""
    by_class = defaultdict(lambda: defaultdict(list))
    for i, sample_id in enumerate(ids):
        label = sample_id.split("/", 1)[0]
        by_class[label][slide_id(sample_id)].append(i)
    if set(by_class) != set(CLASSES):
        raise ValueError("A class has no valid CAM observations")
    chosen = []
    for label in CLASSES:
        slides = sorted(by_class[label])
        for slide_position in rng.integers(0, len(slides), size=len(slides)):
            members = by_class[label][slides[slide_position]]
            chosen.extend(rng.choice(members, size=len(members), replace=True).tolist())
    return chosen


def macro_mean(values: np.ndarray, ids: list[str]) -> float:
    labels = np.asarray([sample_id.split("/", 1)[0] for sample_id in ids])
    return float(np.mean([values[labels == label].mean() for label in CLASSES]))


def summarize_model(model: str, records: dict[str, dict], ratio: str,
                    replicates: int, seed: int) -> dict:
    valid_ids = sorted(key for key, row in records.items()
                       if row["status"] == "ok" and ratio in row["ratios"])
    if len(valid_ids) < 5:
        raise ValueError(f"Too few valid observations for {model}, ratio {ratio}")
    arrays = {}
    for metric in METRICS:
        arrays[metric] = np.asarray([
            records[sample_id]["tissue_enrichment"] if metric == "tissue_enrichment"
            else records[sample_id]["ratios"][ratio][metric]
            for sample_id in valid_ids], dtype=np.float64)
        if not np.isfinite(arrays[metric]).all():
            raise ValueError(f"Nonfinite {metric}: {model}")
    rng = np.random.default_rng(seed)
    bootstrap = {metric: [] for metric in METRICS}
    for _ in range(replicates):
        indexes = bootstrap_indices(valid_ids, rng)
        selected_ids = [valid_ids[i] for i in indexes]
        for metric in METRICS:
            bootstrap[metric].append(macro_mean(arrays[metric][indexes], selected_ids))
    output = {
        "model": model, "ratio": float(ratio), "n_valid": len(valid_ids),
        "n_zero_cam": sum(row["status"] == "zero_cam" for row in records.values()),
        "n_by_class": dict(Counter(sample_id.split("/", 1)[0] for sample_id in valid_ids)),
        "n_wsi": len({slide_id(sample_id) for sample_id in valid_ids}),
        "n_background_available": sum(records[sample_id]["ratios"][ratio]["random_background_available"] for sample_id in valid_ids),
        "bootstrap_replicates": replicates, "bootstrap_seed": seed,
        "bootstrap_unit": "stratified hierarchical WSI-cluster then patch resampling",
        "wsi_id_rule": "class + filename prefix before first period; proxy slide ID; not independently validated patient ID",
        "metrics_pp": {},
    }
    for metric in METRICS:
        values = np.asarray(bootstrap[metric])
        output["metrics_pp"][metric] = {
            "point": 100 * macro_mean(arrays[metric], valid_ids),
            "ci_low": 100 * float(np.quantile(values, 0.025)),
            "ci_high": 100 * float(np.quantile(values, 0.975)),
        }
    return output


def raw_rows(model: str, records: dict[str, dict]) -> list[dict]:
    result = []
    for sample_id, item in sorted(records.items()):
        base = {
            "Dataset": "DeepHisto", "Model": model, "SampleID": sample_id,
            "WSIProxyID": slide_id(sample_id), "TrueLabel": item["true_label"],
            "PredictedLabel": item["predicted_label"], "Status": item["status"],
            "CheckpointSHA256": item["checkpoint_sha256"],
            "OriginalProbability": item["original_probability"],
            "OriginalLogit": item["original_logit"],
            "TissueAreaRatio": item["tissue_area_ratio"],
            "CAMTissueMass": (item["tissue_enrichment"] + item["tissue_area_ratio"]
                              if item.get("tissue_enrichment") is not None else None),
            "CAMTissueEnrichment": item.get("tissue_enrichment"),
        }
        if item["status"] != "ok":
            result.append({**base, "Ratio": None, "Condition": None, "Repeat": None})
            continue
        for ratio, values in item["ratios"].items():
            for condition in values["conditions"]:
                result.append({
                    **base, "Ratio": ratio, "Condition": condition["condition"],
                    "Repeat": condition["repeat"], "SelectedPixels": condition["selected_pixels"],
                    "MaskSHA256": condition["mask_sha256"],
                    "OccludedProbability": condition["true_class_probability"],
                    "OccludedLogit": condition["true_class_logit"],
                    "HighDrop": values["high_drop_probability"],
                    "RandomDrop": values["random_drop_probability"],
                    "FaithfulnessGap": values["faithfulness_gap_probability"],
                    "HighLogitDrop": values["high_drop_logit"],
                    "RandomLogitDrop": values["random_drop_logit"],
                })
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def render_table(rows: list[dict], path: Path, title: str, columns: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(16, 3.6 if len(rows) == 4 else 3.1))
    ax.axis("off")
    cells = [[str(row[col]) for col in columns] for row in rows]
    table = ax.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    for (_, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
    for i in range(len(columns)):
        table[(0, i)].set_text_props(weight="bold")
        table[(0, i)].set_edgecolor("black")
    fig.suptitle(title, fontsize=11, weight="bold", y=.94)
    fig.tight_layout(rect=(.02, .02, .98, .9))
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ("topoformer", "dinov3", "dinov2", "swin"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    roots = {"TopoFormer": a.topoformer, "DINOv3": a.dinov3,
             "DINOv2": a.dinov2, "Swin": a.swin}
    all_records = {model: read_run(root) for model, root in roots.items()}
    reference_ids = set(all_records["DINOv2"])
    if len(reference_ids) != 200 or any(set(records) != reference_ids for records in all_records.values()):
        raise ValueError("All four models must contain the same fixed 200 Sample IDs")
    qc = validate_repeats(all_records)
    flat = [row for model in MODELS for row in raw_rows(model, all_records[model])]
    write_csv(a.out / "Faithfulness_AllRawRepeats.csv", flat)
    full = {}
    for ratio in RATIOS:
        full[ratio] = [summarize_model(model, all_records[model], ratio,
                                       a.replicates, a.seed) for model in MODELS]
    paired = {}
    correct_subset = {}
    for ratio in RATIOS:
        common = sorted(sample_id for sample_id in reference_ids if all(
            all_records[model][sample_id]["status"] == "ok" and ratio in all_records[model][sample_id]["ratios"]
            for model in MODELS))
        all_correct = [sample_id for sample_id in common if all(
            all_records[model][sample_id]["predicted_label"] ==
            all_records[model][sample_id]["true_label"] for model in MODELS)]
        gaps = {model: np.asarray([all_records[model][sample_id]["ratios"][ratio]["faithfulness_gap_probability"]
                                   for sample_id in common], dtype=np.float64) for model in MODELS}
        paired_rows = []
        rng = np.random.default_rng(a.seed + int(float(ratio) * 100))
        draws = [bootstrap_indices(common, rng) for _ in range(a.replicates)]
        for baseline in MODELS[1:]:
            differences = gaps["TopoFormer"] - gaps[baseline]
            boot = [macro_mean(differences[indexes], [common[i] for i in indexes]) for indexes in draws]
            paired_rows.append({"comparison": "TopoFormer minus " + baseline,
                                "point_pp": 100 * macro_mean(differences, common),
                                "ci_low_pp": 100 * float(np.quantile(boot, .025)),
                                "ci_high_pp": 100 * float(np.quantile(boot, .975))})
        paired[ratio] = {"n_common_valid": len(common), "n_wsi_proxy": len({slide_id(s) for s in common}),
                         "differences": paired_rows}
        correct_subset[ratio] = {
            "n": len(all_correct),
            "n_by_class": dict(Counter(s.split("/", 1)[0] for s in all_correct)),
            "model_gap_mean_pp": {model: 100 * float(np.mean([
                all_records[model][sample_id]["ratios"][ratio]["faithfulness_gap_probability"]
                for sample_id in all_correct])) if all_correct else None for model in MODELS},
            "aggregation": "sample mean on four-model-correct subset; not selected by CAM value",
        }
    metadata = {
        "dataset": "DeepHisto", "fixed_sample_count": 200, "model_order": MODELS,
        "ratios": RATIOS, "methods": {
            "target": "true label", "random_repeats": 5,
            "point_estimate": "macro mean across five classes",
            "confidence_interval": "percentile 95% CI from paired stratified hierarchical WSI/patch bootstrap",
            "bootstrap_replicates": a.replicates, "seed": a.seed,
            "unit_warning": "WSI ID inferred from filename prefix; four classes have one or two WSI prefixes, so confidence intervals do not establish generalization to unseen patients",
            "units": "percentage points; 100 times probability drop or CAM mass minus tissue area fraction",
        }, "quality_control": qc, "results": full,
        "paired_topoformer_differences": paired,
        "four_model_correct_sensitivity": correct_subset,
    }
    (a.out / "TableX_Faithfulness.json").write_text(json.dumps(metadata, indent=2) + "\n")
    def cell(info: dict, metric: str, ci: bool = False) -> str:
        v = info["metrics_pp"][metric]
        return f"{v['point']:.2f} [{v['ci_low']:.2f}, {v['ci_high']:.2f}]" if ci else f"{v['point']:.2f}"
    tablex = [{"Model": info["model"], "High-CAM Drop (pp)": cell(info, METRICS[0]),
               "Random Drop (pp)": cell(info, METRICS[1]),
               "Faithfulness Gap (pp, 95% CI)": cell(info, METRICS[2], True),
               "Tissue Enrichment (pp)": cell(info, METRICS[3]),
               "N": info["n_valid"]} for info in full["0.10"]]
    write_csv(a.out / "TableX_Faithfulness.csv", tablex)
    latex = [r"\begin{table*}[t]", r"\centering", r"\caption{Quantitative Grad-CAM faithfulness analysis on DeepHisto}",
             r"\label{tab:gradcam-faithfulness}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\renewcommand{\arraystretch}{1.15}", r"\begin{tabular}{lcccc}", r"\toprule",
             r"Model & High-CAM Drop (pp) $\uparrow$ & Random Drop (pp) & Faithfulness Gap (pp; 95\% CI) $\uparrow$ & Tissue Enrichment (pp) $\uparrow$ \\",
             r"\midrule"]
    for row in tablex:
        name = r"\textbf{TopoFormer}" if row["Model"] == "TopoFormer" else row["Model"]
        latex.append(f"{name} & {row['High-CAM Drop (pp)']} & {row['Random Drop (pp)']} & {row['Faithfulness Gap (pp, 95% CI)']} & {row['Tissue Enrichment (pp)']} " + r"\\")
    latex += [r"\bottomrule", r"\end{tabular}",
              r"\vspace{2pt}\parbox{0.98\textwidth}{\footnotesize True-class probability drops after 10\% tissue-area Gaussian-blur occlusion; five random-tissue repeats per image. Fixed frozen checkpoints and user-authorized new DINO logistic heads. Paired hierarchical WSI/patch bootstrap (2,000 replicates); $n$ is reported in the machine-readable table. Filename prefixes are proxy WSI IDs and most classes contain few WSIs. Background occlusion was unavailable when the background area was insufficient.}",
              r"\end{table*}"]
    (a.out / "TableX_Faithfulness.tex").write_text("\n".join(latex) + "\n")
    render_table(tablex, a.out / "TableX_Faithfulness", "TABLE X. QUANTITATIVE GRAD-CAM FAITHFULNESS ANALYSIS ON DEEPHISTO",
                 ["Model", "High-CAM Drop (pp)", "Random Drop (pp)", "Faithfulness Gap (pp, 95% CI)", "Tissue Enrichment (pp)"])
    table_s1 = []
    for ratio in RATIOS:
        entry = {"Occlusion Ratio": f"{int(float(ratio) * 100)}%"}
        for info in full[ratio]:
            entry[info["model"]] = cell(info, METRICS[2], True)
            entry[info["model"] + " N"] = info["n_valid"]
        table_s1.append(entry)
    write_csv(a.out / "TableS1_OcclusionRatioSensitivity.csv", table_s1)
    (a.out / "TableS1_OcclusionRatioSensitivity.json").write_text(json.dumps({"methods": metadata["methods"], "rows": table_s1}, indent=2) + "\n")
    supplement_latex = [r"\begin{table*}[t]", r"\centering",
                        r"\caption{Occlusion-ratio sensitivity of the Grad-CAM faithfulness gap}",
                        r"\label{tab:gradcam-ratio}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
                        r"\begin{tabular}{lcccc}", r"\toprule",
                        r"Tissue-area occlusion & TopoFormer & DINOv3 & DINOv2 & Swin \\", r"\midrule"]
    for row in table_s1:
        supplement_latex.append(" & ".join([row["Occlusion Ratio"].replace("%", r"\%"),
                                             *(row[model] for model in MODELS)]) + r" \\")
    supplement_latex += [r"\bottomrule", r"\end{tabular}",
                         r"\vspace{2pt}\parbox{0.98\textwidth}{\footnotesize Values are true-class faithfulness gaps in percentage points with paired hierarchical WSI/patch bootstrap 95\% CIs. Five random-tissue repetitions per image. Actual valid sample counts are provided in the accompanying CSV and JSON.}",
                         r"\end{table*}"]
    (a.out / "TableS1_OcclusionRatioSensitivity.tex").write_text("\n".join(supplement_latex) + "\n")
    render_table(table_s1, a.out / "TableS1_OcclusionRatioSensitivity",
                 "TABLE S1. OCCLUSION RATIO SENSITIVITY: FAITHFULNESS GAP (pp; 95% CI)",
                 ["Occlusion Ratio", *MODELS])
    print(json.dumps({"TableX": tablex, "TableS1": table_s1}, indent=2), flush=True)


if __name__ == "__main__":
    main()
