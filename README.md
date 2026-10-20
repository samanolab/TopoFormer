# TopoFormer

## A Multi-Scale Topology-Aware Vision Transformer for H&E Tumor Classification

TopoFormer is a topology-aware histopathology classification framework that integrates a pretrained **DINOv3 ViT-7B/16** visual foundation model with **persistent-homology (PH)** representations. The framework combines a **multi-scale topology feature pyramid (TopoFPN)**, **topology-conditioned token interaction (TopoCTI)**, **ConvNeXtV2-based gated fusion**, **discriminative–nuisance feature disentanglement (DNFD)**, and a **global topological vector** for patch-level H&E tumor classification.

<p align="center">
  <img src="docs/figures/Architecture.png" alt="TopoFormer architecture" width="95%">
</p>

**Fig. 1.** Overall architecture of TopoFormer. Persistent-homology descriptors are organized into multi-scale topological representations, injected into the DINOv3 visual stream through TopoCTI, combined with complementary ConvNeXtV2 features, processed by DNFD, and fused with a global topological vector for final classification.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Environment and Installation](#environment-and-installation)
- [Pretrained Weights](#pretrained-weights)
- [Datasets](#datasets)
- [Data Partitions and Experimental Protocol](#data-partitions-and-experimental-protocol)
- [TCGA-UT External Evaluation](#tcga-ut-external-evaluation)
- [Data Preparation](#data-preparation)
- [Persistent-Homology Preprocessing](#persistent-homology-preprocessing)
- [Architecture and Training Configuration](#architecture-and-training-configuration)
- [DINOv3 Backbone Modification](#dinov3-backbone-modification)
- [Core Modules](#core-modules)
- [Training](#training)
- [Final Feature-Fusion Evaluation](#final-feature-fusion-evaluation)
- [Reported Benchmark Results](#reported-benchmark-results)
- [Additional Analyses](#additional-analyses)
- [Computational Considerations](#computational-considerations)
- [Reproducibility Notes](#reproducibility-notes)
- [Citation](#citation)
- [Third-Party Resources and Dataset Citations](#third-party-resources-and-dataset-citations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

TopoFormer is designed for classification of H&E-stained histopathology patches. It combines high-capacity pretrained visual representations with explicit topological descriptors derived from persistent homology.

The released implementation contains the following principal components:

- **DINOv3 ViT-7B/16 visual backbone** for pretrained visual representation learning.
- **LoRA adaptation** for parameter-efficient adaptation of the frozen DINOv3 backbone.
- **Persistent-homology encoder** using grayscale foreground point sets, alpha complexes, and H1 persistence.
- **TopoFPN** for organizing local PH descriptors into aligned multi-scale topology maps.
- **TopoCTI** for topology-conditioned interaction with DINOv3 patch tokens.
- **ConvNeXtV2-based gated fusion** for complementary convolutional visual representation.
- **DNFD** for encouraging task-related feature differentiation through discriminative and auxiliary representations with regularization objectives.
- **Global topological vector** for complementary sample-level topological information.
- **L2-regularized logistic regression** for the final feature-fusion classifier.
- **Dataset-specific preprocessing, training, and evaluation scripts** for DeepHisto and CRC.
- **External TCGA-UT evaluation** for three-class glioma generalization.

The reported experiments concern **patch-level classification**. The current repository does not claim a validated whole-slide clinical deployment pipeline.

---

## Repository Structure

The repository is organized around dataset preparation, topology generation, model definition, training, and post-training feature-fusion evaluation.

> The tree below lists the principal files referenced by the released experimental pipeline. Large datasets, pretrained checkpoints, cached PH maps, extracted features, and experiment logs are intentionally kept outside the Git repository.

```text
TopoFormer/
├── Backbone/
│   └── dinov3/
│       └── dinov3/
│           └── dinov3/
│               └── models/
│                   └── vision_transformer.py
│
├── docs/
│   └── figures/
│       └── Architecture.png
│
├── notebooks/
│   ├── TopoFPN_layer.py
│   ├── TopoFPN_vector.py
│   ├── TopoFPN_layer_CRC_180.py
│   ├── TopoFPN_vector_CRC_180.py
│   │
│   ├── Final_feature_fusion_eval_briancancer—200_seed42.ipynb
│   ├── Final_feature_fusion_eval_briancancer—200_seed123.ipynb
│   ├── Final_feature_fusion_eval_briancancer—200_seed666.ipynb
│   ├── Final_feature_fusion_eval_crc—180_seed12.ipynb
│   ├── Final_feature_fusion_eval_crc—180_seed42.ipynb
│   └── Final_feature_fusion_eval_crc—180_seed666.ipynb
│
├── scripts/
│   ├── train_briancancer_rgb.py
│   ├── train_crc_rgb.py
│   ├── train_briancancer_rgb_200_seed42.py
│   └── train_crc_rgb_180_seed42.py
│
├── topoformer/
│   ├── data/
│   │   └── topo_maps.py
│   ├── engine/
│   └── model/
│       ├── topo_cti_v2.py
│       ├── lora.py
│       ├── disentanglement.py
│       └── convnext_fusion.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

Seed-specific scripts and notebooks follow the same experimental configuration with the corresponding dataset-specific threshold, random seed, checkpoint, and cache paths.

### Main entry points

The principal training entry points are located under:

```text
scripts/
```

Representative commands are:

```bash
# DeepHisto: foreground threshold 200, seed 42
python scripts/train_briancancer_rgb_200_seed42.py

# CRC: foreground threshold 180, seed 42
python scripts/train_crc_rgb_180_seed42.py
```

The generic dataset-level scripts are:

```text
scripts/train_briancancer_rgb.py
scripts/train_crc_rgb.py
```

These scripts contain or call the main dataset loading, model construction, optimization, checkpointing, evaluation, and feature-cache procedures used by the project.

---

## Environment and Installation

### Reported environment

The experiments reported in the manuscript use:

| Package / Hardware | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.6.0 |
| CUDA | 12.4 |
| GUDHI | 3.10.0 |
| GPU | NVIDIA A40, 48 GB |

### Create the environment

```bash
conda create -n topoformer python=3.10 -y
conda activate topoformer
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

Verify the main software stack:

```bash
python -c "import torch, gudhi; print(torch.__version__, torch.version.cuda, gudhi.__version__)"
```

`requirements.txt` is intended as a cleaned project-level dependency list rather than a full operating-system or Conda environment export.

If the default PyTorch wheel does not match the local CUDA installation, install the appropriate PyTorch/torchvision build for the target system before installing the remaining dependencies.

---

## Pretrained Weights

TopoFormer depends on pretrained DINOv3 and ConvNeXtV2 checkpoints. These files are not redistributed by this repository and remain subject to their original licenses and distribution terms.

### DINOv3

Official repository:

- https://github.com/facebookresearch/dinov3

Checkpoint used in the reported configuration:

```text
dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
```

Configure the local path in the corresponding training script:

```python
DINO_LOCAL_CKPT = "/path/to/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
```

### ConvNeXt V2

Official repository:

- https://github.com/facebookresearch/ConvNeXt-V2

Checkpoint used in the reported configuration:

```text
convnextv2_base_1k_224_ema.pt
```

Example path configuration:

```python
CONVNEXT_CKPT = "/path/to/convnextv2_base_1k_224_ema.pt"
CONVNEXT_NAME = "convnextv2_base"
```

---

## Datasets

| Dataset | Source | Role in this project |
|---|---|---|
| **DeepHisto** | [Zenodo v0.1](https://doi.org/10.5281/zenodo.7941080) | Five-class brain histopathology classification |
| **NCT-CRC-HE-100K** | [Zenodo v0.1](https://doi.org/10.5281/zenodo.1214456) | Nine-class colorectal tissue training/development set |
| **CRC-VAL-HE-7K** | [Zenodo v0.1](https://doi.org/10.5281/zenodo.1214456) | Independent CRC final test set containing 7,180 patches |
| **TCGA-UT** | [Zenodo](https://doi.org/10.5281/zenodo.5889558) · [Hugging Face](https://huggingface.co/datasets/dakomura/tcga-ut) | External three-class glioma evaluation |

Users are responsible for downloading each dataset from its original source and complying with the corresponding license, access, attribution, and usage requirements.

---

## Data Partitions and Experimental Protocol

### DeepHisto

- The official **patient-wise training/test partition** is preserved.
- The held-out official test set contains **4,174 patches**.
- A stratified **10% of the official training set** is reserved for internal validation.
- The remaining **90% of the official training set** is used for model training.
- The official test set is not used for checkpoint selection.

Reported training seeds:

```text
42, 123, 666
```

### CRC

- **NCT-CRC-HE-100K** supplies the training/development samples.
- A stratified **10% of NCT-CRC-HE-100K** is reserved for internal validation.
- The remaining **90%** is used for training.
- **CRC-VAL-HE-7K** is retained as the independent final test set.
- CRC-VAL-HE-7K is not used for checkpoint selection.

Reported training seeds:

```text
12, 42, 666
```

### Checkpoint selection

For both datasets, checkpoints are selected according to:

```text
best validation balanced accuracy
```

The official test sets are evaluated only after model selection.

---

## TCGA-UT External Evaluation

The external evaluation uses **786 TCGA-UT patches** from three glioma subtype directories.

Importantly, this experiment uses **separately trained three-class DeepHisto models**. Training includes astrocytoma, glioblastoma, and oligodendroglioma only; the normal and necrosis classes are excluded. The external experiment therefore does **not** select three logits from a five-class classifier.

| External directory | Three-class label | Patches | Distinct WSI identifiers |
|---|---|---:|---:|
| `astrocytoma/` | `astro` | 282 | 69 |
| `glioblastoma/` | `gbm` | 344 | 79 |
| `oligodendroglioma/` | `oligo` | 160 | 39 |
| **Total** |  | **786** | **187** |

The cohort is tracked with:

```text
TCGA_UT_786_sample_manifest.csv
```

The manifest contains:

```text
Dataset
OriginalFolder
MappedLabel5to3
WSI_ID
PatchPath
```

`WSI_ID` identifies a source slide directory. The total of **187 distinct WSI identifiers must not be interpreted as 187 independently verified patients**.

Original TCGA-UT imagery can be obtained from:

- https://doi.org/10.5281/zenodo.5889558
- https://huggingface.co/datasets/dakomura/tcga-ut

When the manifest is moved to a different local environment, update `PatchPath` entries to match the local dataset location.

### External stain normalization

External images are evaluated under two conditions:

1. **Original external images**
2. **Reinhard-style color normalization**

Reference BGR-channel means and standard deviations are estimated from **3,000 sampled DeepHisto training patches**. These statistics are applied only during external evaluation.

The trained model checkpoints remain fixed across the two external-evaluation conditions.

---

## Data Preparation

### DeepHisto

Download DeepHisto from the official Zenodo record and preserve its official patient-wise train/test organization.

A simplified local layout is:

```text
/path/to/DeepHisto/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

Configure the local paths in the DeepHisto training scripts, for example:

```python
BRIANCANCER_TRAIN_DIR = "/path/to/DeepHisto/train"
BRIANCANCER_TEST_DIR = "/path/to/DeepHisto/test"
```

### CRC

A simplified local layout is:

```text
/path/to/CRC/
├── NCT-CRC-HE-100K/
└── CRC-VAL-HE-7K/
```

Configure the corresponding paths:

```python
NCTCRC100K_DIR = "/path/to/CRC/NCT-CRC-HE-100K"
CRCVAL7K_DIR = "/path/to/CRC/CRC-VAL-HE-7K"
```

### Large generated files

Datasets and the following generated files should normally be stored outside the Git repository:

- pretrained model checkpoints;
- TopoFPN `.npz` archives;
- cached feature matrices;
- model checkpoints;
- experiment logs;
- external-evaluation image copies;
- large intermediate PH outputs.

Use explicit local paths in the dataset-specific scripts or configuration files.

---

## Persistent-Homology Preprocessing

The visual and topological branches use separate input preprocessing.

### Visual branch

The visual branch uses:

```text
224 × 224 RGB
```

with ImageNet normalization.

### PH branch

The PH branch uses separately resized grayscale images.

| Configuration | DeepHisto | CRC |
|---|---:|---:|
| PH grayscale input | **128 × 128** | **224 × 224** |
| Foreground threshold | **200** | **180** |
| Grayscale range | 8-bit | 8-bit |
| Morphological dilation | 2 × 2 kernel, 1 iteration | Same |
| Local PH windows | 16, 32, 64, 128 px | Same |
| Stride | Equal to window size | Same |
| Complex | Alpha complex | Same |
| Homology dimension | **H1** | Same |
| Persistence image | 20 × 20 | Same |
| Gaussian bandwidth | 1.0 | Same |
| Persistence weighting | Uniform | Same |
| TopoFPN alignment | Nearest-neighbor top-down upsampling | Same |

### Foreground construction

The PH preprocessing procedure is:

1. Convert the source RGB image to grayscale with OpenCV.
2. Resize the grayscale image using area interpolation.
3. Apply the dataset-specific foreground threshold.
4. Pixels with intensities **above** the threshold are set to zero.
5. Intensities of retained pixels are preserved.
6. Apply one morphological dilation using a `2 × 2` kernel.
7. Within each local window, collect the spatial coordinates of nonzero pixels.
8. Treat these coordinates as the point set used to construct an alpha complex.
9. Compute **H1 persistent homology**.
10. Convert the persistence representation into a `20 × 20` persistence image.
11. Organize local descriptors into four multi-scale spatial maps.
12. Align the maps through the TopoFPN top-down pathway.

### Dataset-specific topology preprocessing

The topology preprocessing modules referenced by the released experiments are:

```text
notebooks/TopoFPN_layer.py
notebooks/TopoFPN_vector.py
notebooks/TopoFPN_layer_CRC_180.py
notebooks/TopoFPN_vector_CRC_180.py
```

The output consists of aligned, per-level topology maps saved as `.npz` files.

A typical set of cached files is:

```text
/path/to/cache/topofpn_levels_train_test_maps_L0.npz
/path/to/cache/topofpn_levels_train_test_maps_L1.npz
/path/to/cache/topofpn_levels_train_test_maps_L2.npz
/path/to/cache/topofpn_levels_train_test_maps_L3.npz
```

Example configuration:

```python
TOPO_NPZ_BASE = "/path/to/cache/topofpn_levels_train_test_maps"
```

### Cache alignment requirement

The image files, split definitions, and topology-cache entries must refer to the same samples.

Before training, verify that:

```text
image identifier
↔ split entry
↔ cached PH / TopoFPN entry
```

are aligned consistently.

Incorrect cache-to-image matching invalidates the corresponding experiment even if the array dimensions remain compatible.

---

## Architecture and Training Configuration

| Component | Reported configuration |
|---|---|
| Visual backbone | DINOv3 ViT-7B/16 |
| Backbone initialization | Official pretrained weights |
| DINOv3 backbone | Frozen |
| Adaptation | LoRA |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| TopoCTI stages | Blocks 1–10, 11–20, 21–30, 31–40 |
| TopoCTI injection locations | Before blocks 1, 11, 21, and 31 |
| TopoCTI query | Visual patch tokens |
| Topological context | Multi-scale topology tokens |
| TopoCTI attention heads | 64 |
| Attention dropout | 0.1 |
| Projection dropout | 0.1 |
| Topology update clipping | 0.3 × patch-token norm |
| ConvNeXtV2 branch | Pretrained and frozen |
| ConvNeXtV2 role | Gated complementary visual fusion |
| DNFD `λortho` | 0.05 |
| DNFD `λadv` | 0.1 |
| Optimizer | AdamW |
| Initial learning rate | 5 × 10⁻⁵ |
| Weight decay | 0.05 |
| Epochs | 2 |
| Initial batch size | 8 |
| Mixed precision | FP16 |
| Gradient clipping | 1.0 |
| Label smoothing | 0.05 |
| EMA | 0.999 |
| Sampling | Class-balanced sampling with replacement |
| Checkpoint criterion | Validation balanced accuracy |
| Final feature classifier | L2-regularized logistic regression |
| Logistic-regression `C` | 1.0 |

### Final feature representation

Before the final classifier:

- the learned discriminative visual representation and the global topological representation are normalized independently;
- the normalized representations are combined;
- an L2-regularized logistic-regression classifier is trained with `C = 1.0`.

### DNFD interpretation

DNFD is designed to encourage **task-related feature differentiation** through a discriminative branch, an auxiliary branch, and regularization objectives.

The current experimental evidence supports partial class-related representation differentiation. It should **not** be interpreted as establishing complete removal of stain, background, or other nuisance information.

---

## DINOv3 Backbone Modification

This repository includes a local DINOv3 source tree under:

```text
Backbone/dinov3/
```

The project is based on the official DINOv3 implementation:

- https://github.com/facebookresearch/dinov3

The principal local DINOv3 source file modified for TopoFormer is:

```text
Backbone/dinov3/dinov3/dinov3/models/vision_transformer.py
```

The modification enables DINOv3 feature forwarding to accept and inject multi-scale topological representations through the TopoCTI interface.

Users comparing this repository against upstream DINOv3 should therefore distinguish:

1. unchanged upstream DINOv3 source files;
2. the TopoFormer-specific modification to `vision_transformer.py`;
3. TopoFormer modules implemented under `topoformer/`.

---

## Core Modules

### 1. TopoFPN

TopoFPN organizes persistent-homology descriptors into spatially aligned multi-scale topology maps.

Relevant project files include:

```text
topoformer/data/topo_maps.py
notebooks/TopoFPN_layer.py
notebooks/TopoFPN_vector.py
notebooks/TopoFPN_layer_CRC_180.py
notebooks/TopoFPN_vector_CRC_180.py
```

The four topology levels are aligned in a top-down pyramid before injection into the visual transformer.

### 2. TopoCTI

TopoCTI performs topology-conditioned interaction inside DINOv3.

Implementation:

```text
topoformer/model/topo_cti_v2.py
```

The reported stage assignment is:

```text
Stage 1  <- L0
Stage 2  <- L1
Stage 3  <- L2
Stage 4  <- L3
```

The corresponding DINOv3 block groups are:

```text
Stage 1: blocks 1–10
Stage 2: blocks 11–20
Stage 3: blocks 21–30
Stage 4: blocks 31–40
```

Topology is injected before blocks:

```text
1, 11, 21, 31
```

Visual patch tokens act as the queries in the topology-conditioned attention operation.

### 3. LoRA adaptation

LoRA modules provide parameter-efficient adaptation of the frozen DINOv3 backbone.

Implementation:

```text
topoformer/model/lora.py
```

Reported settings:

```text
rank    = 16
alpha   = 16
dropout = 0.05
```

### 4. DNFD

DNFD separates task-oriented discriminative representations from an auxiliary feature branch and applies regularization objectives to encourage representation differentiation.

Implementation:

```text
topoformer/model/disentanglement.py
```

Reported objective weights:

```text
lambda_ortho = 0.05
lambda_adv   = 0.1
```

### 5. ConvNeXtV2-based gated fusion

A frozen pretrained ConvNeXtV2-Base branch provides complementary convolutional visual features.

Implementation:

```text
topoformer/model/convnext_fusion.py
```

The ConvNeXtV2 representation is integrated through a learned gated fusion mechanism rather than simple unweighted concatenation.

### 6. Global topological vector

In addition to local multi-scale topology maps, the pipeline computes a sample-level global topological vector.

This representation is normalized independently from the discriminative visual feature before final fusion.

For the **Global Vector Only** branch-control experiment, the global topological vector is directly used to train an L2-regularized logistic classifier.

---

## Training

Before launching training, configure:

- dataset paths;
- internal train/validation split settings;
- official test-set path;
- DINOv3 checkpoint path;
- ConvNeXtV2 checkpoint path;
- TopoFPN cache path;
- output checkpoint directory;
- feature-cache directory;
- random seed.

### DeepHisto example

```bash
python scripts/train_briancancer_rgb_200_seed42.py
```

This corresponds to:

```text
PH foreground threshold = 200
seed                    = 42
```

Reported DeepHisto seeds:

```text
42
123
666
```

### CRC example

```bash
python scripts/train_crc_rgb_180_seed42.py
```

This corresponds to:

```text
PH foreground threshold = 180
seed                    = 42
```

Reported CRC seeds:

```text
12
42
666
```

### Validation and test-set separation

The released experimental protocol uses a stratified validation subset drawn only from the official training data.

Model/epoch selection is based on **validation balanced accuracy**.

The official test sets are not used for:

- early stopping;
- checkpoint selection;
- hyperparameter selection.

---

## Final Feature-Fusion Evaluation

The seed-specific final evaluation notebooks are:

| Dataset | Seed | Notebook |
|---|---:|---|
| DeepHisto | 42 | `Final_feature_fusion_eval_briancancer—200_seed42.ipynb` |
| DeepHisto | 123 | `Final_feature_fusion_eval_briancancer—200_seed123.ipynb` |
| DeepHisto | 666 | `Final_feature_fusion_eval_briancancer—200_seed666.ipynb` |
| CRC | 12 | `Final_feature_fusion_eval_crc—180_seed12.ipynb` |
| CRC | 42 | `Final_feature_fusion_eval_crc—180_seed42.ipynb` |
| CRC | 666 | `Final_feature_fusion_eval_crc—180_seed666.ipynb` |

Each notebook must be paired with the matching:

- random seed;
- model checkpoint;
- feature cache;
- topology cache;
- dataset split;
- final test-set path.

Mixing outputs from different seeds or preprocessing thresholds is not a valid reproduction of the reported result.

---

## Reported Benchmark Results

The following results are reported as **mean ± sample standard deviation (%) over three seed-specific runs**.

| Dataset | Accuracy | Balanced Accuracy | Macro Precision | Macro-F1 | Macro OvR-AUC |
|---|---:|---:|---:|---:|---:|
| **DeepHisto** | **97.42 ± 0.16** | **90.60 ± 1.10** | **95.84 ± 0.46** | **92.70 ± 0.87** | **99.63 ± 0.12** |
| **CRC** | **96.98 ± 0.23** | **95.56 ± 0.36** | **95.63 ± 0.28** | **95.53 ± 0.33** | **99.76 ± 0.09** |

For CRC, balanced accuracy is reported together with overall accuracy because the nine tissue classes are not perfectly balanced.

---

## Additional Analyses

The manuscript and supplementary experiments additionally evaluate:

- component ablation;
- branch-control experiments;
- independent topology-module validation;
- leave-one-component-out ablations;
- progressive cumulative ablations;
- foreground-threshold sensitivity;
- grayscale robustness;
- controlled stain perturbation;
- patch-shuffle spatial-structure perturbation;
- feature discriminability;
- attribution / faithfulness analysis;
- runtime and computational profiling;
- external TCGA-UT generalization;
- paired statistical comparisons.

### Foreground-threshold settings used in the main experiments

The main reported settings are:

```text
DeepHisto: threshold 200
CRC:       threshold 180
```

These values are dataset-specific and correspond to the final preprocessing configuration used for the reported benchmark results.

---

## Computational Considerations

The reported TopoFormer configuration contains approximately:

```text
Total parameters:     7.12B
Trainable parameters: 321.6M
```

The large DINOv3 backbone is frozen and adapted with LoRA; however, freezing parameters does not remove the inference-time computational cost of executing the backbone.

### Persistent-homology cost

PH maps can be precomputed and cached.

This is the recommended setting for repeated training or evaluation because:

- topology extraction is deterministic under fixed preprocessing;
- repeated online PH construction adds substantial preprocessing latency;
- cached maps make repeated seed-specific experiments more practical.

When online PH extraction is enabled, its preprocessing time should be considered separately from neural-network inference time.

### Scope

The reported experiments evaluate patch-level histopathology classification.

The repository does not establish:

- whole-slide inference validity;
- clinical diagnostic performance;
- prospective clinical utility;
- deployment-time latency for a complete pathology workflow.

---

## Reproducibility Notes

For a reproducible run, verify all of the following before training:

- [ ] The correct official dataset version has been downloaded.
- [ ] DeepHisto preserves the official patient-wise train/test partition.
- [ ] CRC-VAL-HE-7K is not used for training or checkpoint selection.
- [ ] The internal validation subset is drawn only from the official training data.
- [ ] The correct random seed is used.
- [ ] The correct PH foreground threshold is used.
- [ ] The correct PH input resolution is used.
- [ ] All four TopoFPN levels are present.
- [ ] Image identifiers and topology-cache identifiers are aligned.
- [ ] DINOv3 and ConvNeXtV2 checkpoints correspond to the reported pretrained models.
- [ ] The local DINOv3 source tree contains the TopoFormer-specific `vision_transformer.py` modification.
- [ ] The reported optimizer, learning rate, weight decay, epoch count, and regularization settings are unchanged.
- [ ] The selected checkpoint is based on internal validation balanced accuracy.
- [ ] Final evaluation uses the untouched official test set.
- [ ] Seed-specific feature-fusion notebooks are paired with their matching checkpoints and caches.
- [ ] Mean and sample standard deviation are computed across the three reported seeds.

### Random seeds

```text
DeepHisto: 42, 123, 666
CRC:       12, 42, 666
```

### Recommended output organization

A reproducible local experiment directory can be organized as:

```text
experiments/
├── deephisto/
│   ├── seed42/
│   ├── seed123/
│   └── seed666/
└── crc/
    ├── seed12/
    ├── seed42/
    └── seed666/
```

For each seed, retain at least:

```text
training configuration
best validation checkpoint
validation metrics
final test metrics
feature cache
logistic-regression output
software / hardware information
```

---

## Citation

If you use TopoFormer or this repository in academic work, please cite the associated manuscript.

```bibtex
@article{li2026topoformer,
  title   = {TopoFormer: A Multi-Scale Topology-Aware Vision Transformer for H\&E Tumor Classification},
  author  = {Li, Zhibo and Peng, Sijun and Liu, Jian and Rodr{\'i}guez, Alfredo and Chan, Kuan Yoow and S{\'a}mano-S{\'a}nchez, Hugo},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Manuscript}
}
```

> Update the BibTeX entry with the final volume, issue, page range, DOI, and publication status after formal publication.

For an archival public release, it is also recommended to provide a repository-level `CITATION.cff` containing the final manuscript metadata.

---

## Third-Party Resources and Dataset Citations

TopoFormer builds on external software, pretrained models, and public datasets. Users should cite the original resources in addition to the TopoFormer manuscript when those resources materially contribute to their work.

### DINOv3

**Paper**

O. Siméoni *et al.*, **“DINOv3,”** arXiv:2508.10104, 2025.

- Paper: https://arxiv.org/abs/2508.10104
- Code: https://github.com/facebookresearch/dinov3

### ConvNeXt V2

**Paper**

S. Woo, S. Debnath, R. Hu, X. Chen, Z. Liu, I. S. Kweon, and S. Xie,  
**“ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders,”** 2023.

- Paper: https://arxiv.org/abs/2301.00808
- Code: https://github.com/facebookresearch/ConvNeXt-V2

### GUDHI

Persistent-homology computation in this project uses GUDHI.

- Project: https://gudhi.inria.fr/
- Citation guidance: https://gudhi.inria.fr/python/latest/citation.html

The reported software environment uses **GUDHI 3.10.0**. When preparing a publication, cite the GUDHI reference corresponding to the software version used in the experiment.

### DeepHisto

M. Mittelbronn, A.-C. Hau, S.-Y. Kim, P. V. Nazarov, V. Despotovic, A. Kakoichankava, F. B. K. Borgmann, and G. G. Klamminger,  
**“DeepHisto: Dataset for glioma subtype classification from Whole Slide Images,”** Zenodo, Version 0.1, 2023.

- DOI: https://doi.org/10.5281/zenodo.7941080

### NCT-CRC-HE-100K / CRC-VAL-HE-7K

J. N. Kather, N. Halama, and A. Marx,  
**“100,000 histological images of human colorectal cancer and healthy tissue,”** Zenodo, Version 0.1, 2018.

- DOI: https://doi.org/10.5281/zenodo.1214456

The same Zenodo record contains **CRC-VAL-HE-7K**, the 7,180-patch independent CRC evaluation set used by this project.

### TCGA-UT

D. Komura and S. Ishikawa,  
**“Histology images from uniform tumor regions in TCGA Whole Slide Images (TCGA-UT),”** Zenodo.

- Dataset DOI: https://doi.org/10.5281/zenodo.5889558
- Hugging Face mirror: https://huggingface.co/datasets/dakomura/tcga-ut

The associated publication is:

D. Komura *et al.*,  
**“Universal encoding of pan-cancer histology by deep texture representations,”** *Cell Reports*, 38, 110424, 2022.

- DOI: https://doi.org/10.1016/j.celrep.2022.110424

---

## Acknowledgements

This project builds on the following open research resources:

- **DINOv3** — https://github.com/facebookresearch/dinov3
- **ConvNeXt V2** — https://github.com/facebookresearch/ConvNeXt-V2
- **GUDHI** — https://gudhi.inria.fr/
- **PyTorch** — https://pytorch.org/
- **DeepHisto** — https://doi.org/10.5281/zenodo.7941080
- **NCT-CRC-HE-100K / CRC-VAL-HE-7K** — https://doi.org/10.5281/zenodo.1214456
- **TCGA-UT** — https://doi.org/10.5281/zenodo.5889558

We thank the maintainers and authors of these projects and datasets for making their resources available to the research community.

---

## License

This repository is released for academic research purposes under the terms specified in the repository's `LICENSE` file.

Third-party components are **not automatically relicensed** by the TopoFormer repository. Users must separately comply with the licenses and terms associated with:

- DINOv3 source code and pretrained weights;
- ConvNeXt V2 source code and pretrained weights;
- DeepHisto;
- NCT-CRC-HE-100K / CRC-VAL-HE-7K;
- TCGA-UT;
- GUDHI;
- PyTorch;
- any additional third-party package listed in `requirements.txt`.

Pretrained checkpoints and datasets should be downloaded from their original distribution sources unless redistribution is explicitly permitted.

---

## Research Use and Reporting

When reporting results obtained from this repository, please state at minimum:

1. dataset and exact split;
2. PH threshold and PH input resolution;
3. training seed;
4. pretrained backbone checkpoint;
5. validation/checkpoint-selection protocol;
6. whether PH maps were precomputed or generated online;
7. whether the reported value is a single run or mean ± SD;
8. hardware and key software versions;
9. whether the result is patch-level, slide-level, or external evaluation.

This information is important because the repository contains dataset-specific preprocessing and seed-specific evaluation configurations.

---

## Contact

For questions concerning the implementation, reproduction of the reported experiments, or the associated manuscript, please use the repository issue tracker or the contact information provided in the manuscript.
