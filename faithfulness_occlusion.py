"""Deterministic, equal-area tissue occlusion for the fixed DeepHisto cohort.

All masks are selected on the original RGB pixel grid. A candidate cell is a
32x32 region; a partial final cell gives exact equal pixel counts. Random seeds
depend on image identity, ratio and repeat, never on model identity.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np


GRID = 32
BLUR_KERNEL = 21
BLUR_SIGMA = 5.0


def stable_seed(sample_id: str, ratio: float, repeat: int) -> int:
    payload = f"2026|{sample_id}|{ratio:.6f}|{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _cells(shape: tuple[int, int], eligible: np.ndarray) -> list[np.ndarray]:
    height, width = shape
    cells = []
    for y in range(0, height, GRID):
        for x in range(0, width, GRID):
            ys = np.arange(y, min(y + GRID, height))
            xs = np.arange(x, min(x + GRID, width))
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            indices = (yy * width + xx).ravel()
            indices = indices[eligible.ravel()[indices]]
            if len(indices):
                cells.append(indices)
    return cells


def select_region(eligible: np.ndarray, count: int, *, sample_id: str,
                  ratio: float, repeat: int, cam: np.ndarray | None = None) -> np.ndarray:
    """Select exactly ``count`` eligible pixels using common spatial cells.

    ``cam`` means descending mean CAM within each nonempty cell; ``None`` means
    a seeded random cell permutation. ``repeat=-1`` is reserved for high CAM.
    """
    if eligible.dtype != np.bool_ or eligible.ndim != 2:
        raise ValueError("eligible must be a 2-D bool mask")
    if not 0 < count <= int(eligible.sum()):
        raise ValueError("requested count exceeds available pixels")
    if cam is not None and (cam.shape != eligible.shape or not np.isfinite(cam).all()):
        raise ValueError("CAM must be finite and aligned to the image")
    rng = np.random.default_rng(stable_seed(sample_id, ratio, repeat))
    cells = _cells(eligible.shape, eligible)
    if cam is None:
        order = rng.permutation(len(cells))
    else:
        scores = np.array([float(cam.ravel()[indices].mean()) for indices in cells])
        # Stable tie-breaks are spatial and prediction-independent.
        order = np.argsort(-scores, kind="stable")
    flat = np.zeros(eligible.size, dtype=np.bool_)
    remaining = count
    for cell_index in order:
        indices = cells[int(cell_index)]
        if cam is None:
            indices = rng.permutation(indices)
        else:
            indices = indices[np.argsort(-cam.ravel()[indices], kind="stable")]
        selected = indices[:remaining]
        flat[selected] = True
        remaining -= len(selected)
        if remaining == 0:
            break
    result = flat.reshape(eligible.shape)
    if remaining != 0 or int(result.sum()) != count or np.any(result & ~eligible):
        raise RuntimeError("Occlusion area or eligibility validation failed")
    return result


def make_masks(tissue: np.ndarray, cam: np.ndarray, ratio: float,
               sample_id: str, random_repeats: int = 5) -> dict:
    if tissue.dtype != np.bool_ or tissue.shape != cam.shape:
        raise ValueError("Tissue and CAM must be aligned")
    if not 0 < ratio < 1 or random_repeats != 5:
        raise ValueError("Protocol requires 5 random repeats and ratio in (0,1)")
    n_tissue = int(tissue.sum())
    count = max(1, int(round(n_tissue * ratio)))
    high = select_region(tissue, count, sample_id=sample_id,
                         ratio=ratio, repeat=-1, cam=cam)
    random_tissue = [select_region(tissue, count, sample_id=sample_id,
                                   ratio=ratio, repeat=i) for i in range(random_repeats)]
    background = None
    if int((~tissue).sum()) >= count:
        background = select_region(~tissue, count, sample_id=sample_id,
                                   ratio=ratio, repeat=100)
    return {"high": high, "random_tissue": random_tissue,
            "random_background": background, "target_pixels": count}


def apply_blur(rgb: np.ndarray, selected: np.ndarray) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB image must be uint8 HxWx3")
    if selected.dtype != np.bool_ or selected.shape != rgb.shape[:2]:
        raise ValueError("Occlusion mask must be bool and image-aligned")
    blurred = cv2.GaussianBlur(rgb, (BLUR_KERNEL, BLUR_KERNEL), BLUR_SIGMA)
    result = rgb.copy()
    result[selected] = blurred[selected]
    return result


def tissue_enrichment(cam: np.ndarray, tissue: np.ndarray) -> float:
    if cam.shape != tissue.shape or not np.isfinite(cam).all() or (cam < 0).any():
        raise ValueError("CAM must be finite, nonnegative and aligned")
    total = float(cam.sum())
    if total <= 0:
        raise ValueError("Zero CAM has no defined tissue enrichment")
    return float(cam[tissue].sum()) / total - float(tissue.mean())
