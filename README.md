# TopoFormer

**Multi-scale topology-aware visual representations for H&E histopathological classification**

TopoFormer combines persistent-homology (PH) descriptors with a DINOv3 ViT-7B/16 visual backbone. Its main components are a multi-scale topology feature pyramid (TopoFPN), stage-wise topology-conditioned token interaction (TopoCTI), discriminative–nuisance feature disentanglement (DNFD), a ConvNeXtV2-based gated visual branch, and a global topological vector for final classification.

**Scope.** This repository concerns **patch-level** tumor/tissue classification on DeepHisto and colorectal histology (CRC) benchmarks. It does not establish WSI-level diagnosis, patient-level prognosis, or clinical deployment readiness.

![TopoFormer architecture](docs/figures/Architecture.png)

## Repository layout

The revision uses the following file organization (files must be committed before these paths become available on GitHub):

```text
TopoFormer/
├── Backbone/dinov3/dinov3/            # Local DINOv3 source, with TopoCTI integration
├── topoformer/
│   ├── data/                           # Datasets, preprocessing, split and topology loaders
│   ├── engine/                         # Runtime and feature caching
│   └── model/                          # TopoCTI, LoRA, DNFD, gated fusion
├── scripts/
│   ├── train_briancancer_rgb_200_seed42.py  # DeepHisto, PH threshold 200, seed 42
│   └── train_crc_rgb_180_seed42.py          # CRC, PH threshold 180, seed 42
├── notebooks/
│   ├── TopoFPN_layer.py
│   ├── TopoFPN_vector.py
│   ├── TopoFPN_layer_CRC_180.py
│   ├── TopoFPN_vector_CRC_180.py
│   ├── Final_feature_fusion_eval_briancancer—200_seed42.ipynb
│   ├── Final_feature_fusion_eval_briancancer—200_seed123.ipynb
│   ├── Final_feature_fusion_eval_briancancer—200_seed666.ipynb
│   ├── Final_feature_fusion_eval_crc—180_seed12.ipynb
│   ├── Final_feature_fusion_eval_crc—180_seed42.ipynb
│   └── Final_feature_fusion_eval_crc—180_seed666.ipynb
├── docs/figures/
├── requirements.txt
└── README.md
```

**For the revised experiments, use the explicitly threshold- and seed-labeled files above.**

## Data availability and partitions

| Dataset | Version / original source | Experimental use |
|---|---|---|
| DeepHisto | [Zenodo v0.1](https://doi.org/10.5281/zenodo.7941080) | Official five-class glioma/brain-tissue task; preserve its published patient-wise train/test separation |
| NCT-CRC-HE-100K | [Zenodo v0.1](https://doi.org/10.5281/zenodo.1214456) | Nine-class CRC training source |
| CRC-VAL-HE-7K | [Same Zenodo v0.1 release](https://doi.org/10.5281/zenodo.1214456) | Independent CRC final test set (7,180 patches) |
| TCGA-UT | [Original Zenodo source](https://doi.org/10.5281/zenodo.5889558); [Hugging Face distribution](https://huggingface.co/datasets/dakomura/tcga-ut/tree/main/data) | External cross-source assessment; see provenance note below |

### Train / validation / test protocol

1. **DeepHisto:** preserve the official patient-wise training/test split. The official test set contains **4,174 patches** and is reserved for final evaluation.
2. **CRC:** use NCT-CRC-HE-100K as the development/training source and the independent **CRC-VAL-HE-7K** set for final testing; do not treat the latter as a tuning or early-stopping set.
3. **Internal validation:** select a **stratified 90%/10% training/validation partition from the official training data only**. Checkpoint/model selection uses **validation balanced accuracy**. The 90%/10% validation partition is *not* claimed to be an independent patient-wise split unless verified by patient identifiers.
4. **Hyperparameter selection:** tuning must use training/validation data exclusively. Use the untouched official test sets only for final reported evaluations.

**Reproducibility requirement:** documenting a split percentage and a random seed does not, by itself, uniquely identify every train/validation sample if scanning order varies. Publish the exact split indices/sample IDs or a deterministic split-generator with stable input ordering before claiming bit-for-bit reconstruction of the partitions.

### External TCGA-UT provenance

The revised external experiment evaluates **786 selected patches** (282 astrocytoma, 344 glioblastoma, 160 oligodendroglioma). The source distribution is the Hugging Face **`external_test`** split, specifically:

- Repository: https://huggingface.co/datasets/dakomura/tcga-ut
- File browser: https://huggingface.co/datasets/dakomura/tcga-ut/tree/main/data
- Archive: **`data/dataset_external_test_part000.tar`**
- Source release revision: **`d3f8bd4`** (Hugging Face dataset-file release)

The TAR is in **WebDataset** format: individual image records have image content and associated JSON metadata, including the source `label` and image key. A sample key can encode a TCGA source identifier. No need to redistribute the source images when linking to the original public archive is sufficient under its terms.

**Subtype-label limitation to resolve before final release:** the source dataset's `json.label` examples use broad TCGA classes such as `Brain_Lower_Grade_Glioma` and `Glioblastoma_multiforme`. The broad lower-grade-glioma label **does not on its own determine astrocytoma versus oligodendroglioma**. To reproduce the precise 786-image three-class cohort, the authors must provide the subtype-label source/mapping and deterministic selection criteria (or an equivalent list of source IDs). The official TAR link alone establishes source provenance but not necessarily the complete three-class curation procedure.

### External stain normalization

External testing includes **with- and without-normalization** conditions. The normalized condition uses Reinhard-style per-channel mean/standard-deviation matching to reference statistics estimated from **3,000 DeepHisto training patches**. The reference statistics are derived from training images only; stain normalization is applied **at external evaluation**, not as part of model training. Interpret the two conditions separately when discussing cross-source generalization.

## Environment and pretrained weights

The revised manuscript reports the following experiment environment:

| Component | Reported setting |
|---|---|
| Python | 3.10 (recommended project environment) |
| PyTorch | 2.6 |
| CUDA | 12.4 |
| GUDHI | 3.10.0 |
| GPU | NVIDIA A40, 48 GB |
| Visual input | 224 × 224 RGB; ImageNet normalization |
| Backbone | DINOv3 ViT-7B/16, local code with topology-injection hooks |
| Complementary visual encoder | ConvNeXtV2-Base, pretrained and frozen |

```bash
conda create -n topoformer python=3.10 -y
conda activate topoformer
# Install a CUDA-compatible PyTorch 2.6 build using official PyTorch instructions.
pip install -r requirements.txt
python -c "import torch, gudhi; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GUDHI', gudhi.__version__)"
```

**Important:** the repository `requirements.txt` uses minimum-version constraints (for example `torch>=2.4`) and is **not a frozen environment lockfile**. For exact reproduction, use the versions above and record/export the exact environment used for the submitted result:

```bash
python -m pip freeze > environment-lock.txt
nvidia-smi
```

Do not describe an environment-lock file as included in this repository until it has actually been committed.

Required pretrained weights are **not stored in this repository**:

- DINOv3: https://github.com/facebookresearch/dinov3 — checkpoint file used in the experiments: `dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth`.
- ConvNeXt V2: https://github.com/facebookresearch/ConvNeXt-V2 — base checkpoint used: `convnextv2_base_1k_224_ema.pt`.

Download pretrained checkpoints in accordance with the upstream instructions. Edit the data, topology-cache, backbone, and checkpoint paths in the relevant training/evaluation scripts. **Paths shown in the source may refer to the authors' compute environment and must be updated for your machine.**

## Persistent-homology preprocessing

The revised PH construction is dataset-specific but otherwise follows the same pipeline:

| Parameter | DeepHisto | CRC |
|---|---:|---:|
| Grayscale foreground threshold | **200** | **180** |
| Input size | 224 × 224 | 224 × 224 |
| TopoFPN spatial window sizes | 16, 32, 64, 128 px | 16, 32, 64, 128 px |
| Window stride | Equal to the window size (non-overlapping) | Same |
| Morphological processing | 2 × 2 dilation, one iteration | Same |
| Homology | H1 (one-dimensional holes) | Same |
| Complex | GUDHI alpha complex | Same |
| Persistence image | 20 × 20, Gaussian bandwidth 1.0 | Same |

For each grayscale tile, retain intensities at or below the selected foreground threshold, set other pixels to zero, and apply dilation. The coordinates of foreground pixels in each local window form the point set for alpha-complex persistence computation; PH descriptors are organized into four spatial scales by TopoFPN. No grayscale inversion or additional denoising step is specified by this procedure. The published α-complex construction does not specify a manually imposed upper filtration cutoff; avoid adding one unless explicitly selected and documented.

The training and evaluation code expects **precomputed** per-level topological maps; use the TopoFPN generation code to create the aligned `.npz` files:

```text
<topology_base>_L0.npz
<topology_base>_L1.npz
<topology_base>_L2.npz
<topology_base>_L3.npz
```

The topology loader expects the corresponding `train_files`, `test_files`, `train_phc_map_level`, and `test_phc_map_level` arrays. The source image paths/identifiers and cached maps must be aligned to prevent mismatched features.

**Preparation files:**

```text
notebooks/TopoFPN_layer.py
notebooks/TopoFPN_vector.py
notebooks/TopoFPN_layer_CRC_180.py
notebooks/TopoFPN_vector_CRC_180.py
```

These Python files currently use project-specific setup; inspect/configure their internal paths before invoking them. Do not assume undocumented CLI arguments are supported.

## Model and training settings

| Setting | Revised manuscript protocol |
|---|---|
| Four topology-injection stages | DINOv3 blocks 1–10, 11–20, 21–30, 31–40 |
| Interaction | Visual patch tokens as queries; PH/topology tokens as keys and values |
| TopoCTI | Cross-attention, 64 heads; attention and projection dropout 0.1 |
| Topology-token clipping | 0.3 × visual patch-token norm |
| Backbone adaptation | LoRA rank 16, α = 16, dropout 0.05 |
| Training epochs | **2**, fixed schedule |
| Optimizer | AdamW, learning rate 5 × 10⁻⁵, weight decay 0.05 |
| Batch size | 8 |
| Other training settings | EMA 0.999; gradient clipping 1.0; label smoothing 0.05; class-balanced sampling with replacement |
| Image augmentation | No additional random image augmentation in the revised training protocol |
| Checkpoint criterion | **Internal-validation balanced accuracy**; official test set held out |
| Final classification | Independently normalized discriminative visual and global topological features, followed by L2-regularized logistic regression |

The fixed two-epoch schedule should not be described as early stopping. The exact parameters and behavior used by the released scripts must agree with the manuscript before labeling them as the reproduction implementation.

### Independent runs and thresholds

| Dataset | Training/evaluation seeds | PH threshold |
|---|---|---:|
| DeepHisto | **42, 123, 666** | **200** |
| CRC | **12, 42, 666** | **180** |

The reported principal results use **three independent runs** and are summarized as mean ± sample standard deviation. A new seed with a different split or classifier initialization is not necessarily interchangeable with one of the paper's seed-specific runs.

## Running the revised workflow

### 1. Prepare the input data and PH caches

Download DeepHisto and CRC from the URLs above. Keep official training and testing data separate. Produce or restore the dataset-specific multi-scale PH `.npz` files, then configure their absolute paths and pretrained-weight paths in the chosen script.

### 2. Run the seed-42 training entry points

```bash
# DeepHisto: foreground threshold 200; seed 42
python scripts/train_briancancer_rgb_200_seed42.py

# CRC: foreground threshold 180; seed 42
python scripts/train_crc_rgb_180_seed42.py
```

These are **file-level entry points**, not a claim that the scripts are portable without configuration. Confirm that the training function receives an **internal validation loader**, not `test_loader`, for checkpoint selection. Also confirm that the scripts used for the reported results have been pushed to this repository.

### 3. Run seed-specific final feature-fusion evaluation

Start Jupyter Lab and open the corresponding notebook **after setting its data, feature-cache, and checkpoint paths**:

```bash
jupyter lab
```

| Dataset | Seed | Notebook |
|---|---:|---|
| DeepHisto | 42 | `Final_feature_fusion_eval_briancancer—200_seed42.ipynb` |
| DeepHisto | 123 | `Final_feature_fusion_eval_briancancer—200_seed123.ipynb` |
| DeepHisto | 666 | `Final_feature_fusion_eval_briancancer—200_seed666.ipynb` |
| CRC | 12 | `Final_feature_fusion_eval_crc—180_seed12.ipynb` |
| CRC | 42 | `Final_feature_fusion_eval_crc—180_seed42.ipynb` |
| CRC | 666 | `Final_feature_fusion_eval_crc—180_seed666.ipynb` |

The evaluation filenames use an em dash (`—`) before the threshold, not a hyphen (`-`). Run notebooks with the checkpoints and feature caches from their corresponding seed. **The listed notebooks are not substitutes for three independent training runs; the released checkpoint provenance must still be documented.**

### Reproducing reported tables

| Manuscript result | Available starting point | Remaining verification / availability |
|---|---|---|
| Table I — DeepHisto/CRC benchmarks | Seed-specific training and final-fusion evaluation above | Other baseline training/evaluation scripts and exact evaluation protocol must be released for complete table reproduction |
| Table II — stain/structure perturbations | Manuscript describes test-time Macenko-style stain shift and 4 × 4 patch shuffle | Dedicated perturbation evaluation command/file not identified among the supplied revision files |
| Table III — external TCGA-UT evaluation | Public TCGA-UT source archive and evaluation protocol | Exact 786-image selection/subtype mapping and external evaluation runner need to be provided |
| Table IV — component ablations | Core modules under `topoformer/model/` and the experiment descriptions | Not all BC/TMV/LOO/PC variant commands are currently documented by the supplied file list |
| Supplementary PH sensitivity and scale ablation | `TopoFPN_*.py` generators and supplementary settings | Confirm the precise run configurations and evaluation notebooks |

A README description is **not a substitute for executable commands**. This table records which reproduction paths are documented and which must be completed before a full manuscript-level reproduction claim can be made.

## Evaluation reporting

The revised manuscript reports Accuracy, balanced accuracy, macro Precision, macro F1, Specificity, and macro one-vs-rest AUC, with three-seed mean ± SD for the principal experiments. Balanced accuracy is the mean of class recalls (and therefore equals macro recall in this single-label multiclass evaluation).

| Dataset | Accuracy (%) | BAcc. (%) | Macro Precision (%) | Macro F1 (%) | Macro OvR AUC (%) |
|---|---:|---:|---:|---:|---:|
| DeepHisto | 97.42 ± 0.16 | 90.60 ± 1.10 | 95.84 ± 0.46 | 92.70 ± 0.87 | **99.63 ± 0.12** |
| CRC | 96.98 ± 0.23 | 95.56 ± 0.36 | 95.63 ± 0.28 | 95.53 ± 0.34 | **99.76 ± 0.09** |

These results are not uniformly superior to all alternatives across all metrics. See the revised paper for the full baseline table, variability, and uncertainty analysis. When computing significance against another model, provide matched **per-sample** predictions and specify the paired test; aggregate table means alone cannot establish a paired p-value.

## Runtime considerations

On an NVIDIA A40 GPU, the manuscript reports approximately 7.12B total parameters, 321.6M trainable parameters, and ~28.57 GB peak inference GPU memory for TopoFormer. At 224 × 224 input, the model-only forward FLOPs are ~3,249.4 G per image. PH generation adds substantial CPU preprocessing latency (~2.35 s/image on DeepHisto and ~1.85 s/image on CRC in the reported setup); the principal training and cached evaluation workflows therefore use precomputed PH features. Model-only and end-to-end timings were measured under different conditions and should not be conflated.

## Reproducibility and release identification

Source repository: **https://github.com/samanolab/TopoFormer**

To identify the exact submitted implementation, cite a **full commit SHA** or immutable release tag in the revised paper's Data Availability and Reproducibility statement. After merging the validated revision into the default branch:

```bash
git rev-parse HEAD
python -m pip freeze > environment-lock.txt
```

The SHA printed by `git rev-parse HEAD` is the local checkout revision; verify it is the precise code version used for the submission results. Do not treat an unmerged development branch or unverified file uploads as a reproducible release.

For enhanced auditability, release the exact internal split indices, all necessary seed configs and checkpoints or their generation commands, external subtype curation provenance, and complete table reproduction runners. These items should only be described as *provided* once they are actually committed.

## Third-party code and licenses

This repository contains or refers to third-party components such as [DINOv3](https://github.com/facebookresearch/dinov3), [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2), [GUDHI](https://gudhi.inria.fr/), [PyTorch](https://pytorch.org/), and public benchmark datasets. Follow the upstream licenses, checkpoint conditions, and dataset use terms. The Hugging Face TCGA-UT dataset page specifies its own distribution license.

**Project license:** A README sentence describing “academic research purposes” is **not a substitute for a formal LICENSE file**. Add an explicit top-level `LICENSE` with the rights you are authorized to grant, while preserving attribution and upstream restrictions; do not assume that upstream pretrained weights are redistributable.

## Citation

Please cite the associated TopoFormer manuscript when using this work. The bibliographic record will be updated after publication.

```bibtex
@article{li2026topoformer,
  title   = {TopoFormer: A Multi-Scale Topology-Aware Visual Transformer for H\&E Tumor Classification},
  author  = {Li, Zhibo and Peng, Sijun and Liu, Jian and Rodr{\'i}guez, Alfredo and Chan, Kuan Yoow and S{\'a}mano-S{\'a}nchez, Hugo},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Manuscript under review}
}
```
