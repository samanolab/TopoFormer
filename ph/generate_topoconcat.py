"""Nearest-neighbor alignment used to prepare TopoFPN and Simple Concat maps."""
import numpy as np


def upsample_nearest(map_chw: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Matches torch.nn.functional.interpolate(mode='nearest') index mapping."""
    c, h, w = map_chw.shape
    th, tw = hw
    if (h, w) == (th, tw):
        return map_chw
    iy = np.minimum((np.arange(th, dtype=np.int64) * h) // th, h - 1)
    ix = np.minimum((np.arange(tw, dtype=np.int64) * w) // tw, w - 1)
    return map_chw[:, iy[:, None], ix[None, :]]
