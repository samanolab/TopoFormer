import cv2
import numpy as np
from gudhi.representations import PersistenceImage

from ph.generate_topoconcat import upsample_nearest
from ph.legacy_alpha_pi import alphacomplex, preprocess

WINDOWS = (16, 32, 64, 128)


def extract(gray, dataset, threshold, pool_levels, concat=False):
    """Apply the same alpha/PI and nearest-neighbor FPN recipe as the saved caches."""
    size = 224 if dataset == "crc" else 128
    gray = cv2.resize(gray.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    prep = preprocess(thresh=threshold, kernel_size=2, iterate=1)
    image = prep.dilate(prep.threshold(gray))
    raw = []
    for width in WINDOWS:
        h = size // width
        maps = np.empty((400, h, h), dtype=np.float32)
        for i in range(h):
            for j in range(h):
                diagram = alphacomplex(image[i * width:(i + 1) * width, j * width:(j + 1) * width])
                pi = PersistenceImage(resolution=(20, 20), bandwidth=1.0)
                maps[:, i, j] = pi.fit_transform(diagram)[0]
        raw.append(maps)

    def pyramid(scales):
        fused = list(scales)
        if not concat:
            for level in range(len(fused) - 2, -1, -1):
                fused[level] = fused[level] + upsample_nearest(fused[level + 1], fused[level].shape[-2:])
        return fused

    grid = raw[0].shape[-2:]
    maps = [upsample_nearest(level, grid) for level in pyramid(raw)]
    pool_maps = pyramid(raw[:pool_levels])
    if concat:
        pool_maps = [upsample_nearest(level, grid) for level in pool_maps]
    # Legacy FPN pools each native grid. On CRC, 3×3 -> 14×14 is not uniform.
    pooled = np.concatenate([level.mean(axis=(1, 2)) for level in pool_maps])
    return maps, pooled
