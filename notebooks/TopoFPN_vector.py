# ===== Notebook code cell 2 =====
import numpy as np
import cv2
from tqdm import tqdm

import gudhi as gd
from gudhi import CubicalComplex, AlphaComplex

from gudhi.representations import Silhouette, PersistenceImage


# ===== Notebook code cell 4 =====
def remove_noisy_pts(pd: np.ndarray) -> np.ndarray:
    """
    Simplify persistence diagrams from pixel-level holes.
    """
    if pd is None or len(pd) == 0:
        return pd
    value_to_remove = np.array([0.25, 0.5])  # remove redundant noise
    mask = np.any(pd != value_to_remove, axis=1)
    denoised_pd = pd[mask]
    return denoised_pd


def pointcloud2D(dilated_img: np.ndarray) -> np.ndarray:
    """
    Only needed for alpha complex.
    Convert nonzero pixels to 2D point cloud (x,y).
    """
    y_coords, x_coords = np.where(dilated_img > 0)
    pointcloud = np.column_stack((x_coords, y_coords))
    return pointcloud


def betti_curve_from_dgm(
    dgm: np.ndarray,
    resolution: int = 20,
    t_min: float | None = None,
    t_max: float | None = None,
) -> np.ndarray:
    """Betti curve (dim=1) from a persistence diagram.

    Return a vector of length resolution: For each threshold t, count the number of intervals where birth <= t < death.
    """
    if dgm is None or len(dgm) == 0:
        return np.zeros(int(resolution), dtype=np.float32)

    dgm = np.asarray(dgm, dtype=np.float32)
    if dgm.ndim != 2 or dgm.shape[1] != 2:
        raise ValueError(f"The expected shape is (N, 2), but the obtained result is {dgm.shape}.")

    births = dgm[:, 0]
    deaths = dgm[:, 1]

    finite = np.isfinite(births) & np.isfinite(deaths)
    births = births[finite]
    deaths = deaths[finite]

    if births.size == 0:
        return np.zeros(int(resolution), dtype=np.float32)

    if t_min is None:
        t_min = float(np.min(births))
    if t_max is None:
        t_max = float(np.max(deaths))

    if not np.isfinite(t_min) or not np.isfinite(t_max):
        return np.zeros(int(resolution), dtype=np.float32)

    if t_max <= t_min:
        t_max = t_min + 1e-3

    ts = np.linspace(float(t_min), float(t_max), int(resolution), dtype=np.float32)

    alive = (births[:, None] <= ts[None, :]) & (deaths[:, None] > ts[None, :])
    bc = alive.sum(axis=0).astype(np.float32)
    return bc


# ===== Notebook code cell 6 =====
def alphacomplex(img: np.ndarray, alpha_value: float = None) -> list[np.ndarray]:
    """
    Alpha complex persistence (dim=1).
    """
    points = pointcloud2D(img)
    if points is None or len(points) == 0:
        return [np.empty((0, 2), dtype=np.float32)]

    alpha_complex = AlphaComplex(points=points)

    if alpha_value is None:
        st = alpha_complex.create_simplex_tree()
    else:
        st = alpha_complex.create_simplex_tree(max_alpha_square=alpha_value**2)

    st.persistence()
    dgm = np.array(st.persistence_intervals_in_dimension(1), dtype=np.float32)
    dgm = remove_noisy_pts(dgm)
    return [dgm]


def cubicalcomplex(img: np.ndarray) -> list[np.ndarray]:
    """
    Cubical complex (lower-star on pixel values) dim=1.
    """
    cc = CubicalComplex(
        dimensions=img.shape,
        top_dimensional_cells=img.flatten()
    )
    cc.persistence(homology_coeff_field=2, min_persistence=0.01)
    dgm = np.array(cc.persistence_intervals_in_dimension(1), dtype=np.float32)
    return [dgm]


def ext_persistence(img: np.ndarray, filtration_function: str = "pixel_intensity") -> list[np.ndarray]:
    """
    Extended persistence diagram (dim=1).
    filtration_function: "pixel_intensity" or "height_function"
    """
    filtration = img.flatten()
    rows, cols = img.shape
    num_points = rows * cols

    st = gd.SimplexTree()

    if filtration_function == "pixel_intensity":
        # vertices
        for i in range(num_points):
            st.insert([i], filtration=float(filtration[i]))

        # edges + diagonals
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j

                if i + 1 < rows:
                    st.insert(
                        [idx, (i + 1) * cols + j],
                        filtration=float(max(img[i, j], img[i + 1, j]))
                    )
                if j + 1 < cols:
                    st.insert(
                        [idx, i * cols + (j + 1)],
                        filtration=float(max(img[i, j], img[i, j + 1]))
                    )
                if i + 1 < rows and j + 1 < cols:
                    st.insert(
                        [idx, (i + 1) * cols + (j + 1)],
                        filtration=float(max(img[i, j], img[i + 1, j + 1]))
                    )
                    st.insert(
                        [i * cols + (j + 1), (i + 1) * cols + j],
                        filtration=float(max(img[i, j + 1], img[i + 1, j]))
                    )

    elif filtration_function == "height_function":
        # vertices: filtration = i % rows
        for i in range(num_points):
            st.insert([i], filtration=float(i % rows))

        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j

                if i + 1 < rows:
                    st.insert([idx, (i + 1) * cols + j], filtration=float(i + 1))
                if j + 1 < cols:
                    st.insert([idx, i * cols + (j + 1)], filtration=float(i))
                if i + 1 < rows and j + 1 < cols:
                    st.insert([idx, (i + 1) * cols + (j + 1)], filtration=float(i + 1))
                    st.insert([i * cols + (j + 1), (i + 1) * cols + j], filtration=float(i + 1))
    else:
        raise ValueError("Unknown filtration_function. Use 'pixel_intensity' or 'height_function'.")

    st.extend_filtration()
    st.extended_persistence(min_persistence=1e-5)
    dgm = np.array(st.persistence_intervals_in_dimension(1), dtype=np.float32)
    return [dgm]


# ===== Notebook code cell 8 =====
class preprocess:
    """
    Image conditioning for topological extraction.
    """

    def __init__(self, thresh: int = None, kernel_size: int = 2, iterate: int = 1):
        self.thresh = thresh
        self.kernel = kernel_size
        self.iterate = iterate

    def threshold(self, img: np.ndarray) -> np.ndarray:
        """
        Keep pixels <= thresh, set others to 0.
        """
        if self.thresh is None:
            raise ValueError("preprocess(thresh=...) must be set.")
        thresh_img = np.where(img <= self.thresh, img, 0)
        return thresh_img

    def dilate(self, thresh_img: np.ndarray) -> np.ndarray:
        """
        Dilate to emphasize boundaries.
        """
        kernel = np.ones((self.kernel, self.kernel), np.uint8)
        dilated_img = cv2.dilate(cv2.convertScaleAbs(thresh_img), kernel, iterations=self.iterate)
        return np.array(dilated_img)


# ===== Notebook code cell 10 =====
class PHC:

    def __init__(
        self,
        persistence_type: str = "cubical",
        window_size: int = 32,
        stride: int = 32,
        vectorization: str = "PI",
        vector_resolution: int = 20,
    ):
        self.persistence_type = str(persistence_type)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.vectorization = str(vectorization)
        self.vector_resolution = int(vector_resolution)

    def process_window(
        self,
        img: np.ndarray,
        x_cord: int,
        y_cord: int,
        window_width: int,
        window_length: int,
    ) -> np.ndarray:

        subimg = img[x_cord : x_cord + window_width, y_cord : y_cord + window_length]

        if subimg is None or subimg.size == 0:
            vec = self.vectorization.upper().strip()
            if vec in ("PI", "PERSISTENCEIMAGE"):
                return np.zeros((self.vector_resolution, self.vector_resolution), dtype=np.float32).reshape(-1)
            else:
                return np.zeros(int(self.vector_resolution), dtype=np.float32)

        if self.persistence_type == "cubical":
            pd = cubicalcomplex(subimg)
        elif self.persistence_type == "alpha":
            pd = alphacomplex(subimg)
        elif self.persistence_type == "extended":
            pd = ext_persistence(subimg, filtration_function="height_function")
        else:
            raise ValueError("Unknown persistence_type: choose 'cubical', 'alpha', or 'extended'.")

        vec = self.vectorization.upper().strip()

        if vec in ("PL", "SILHOUETTE"):
            SH = Silhouette(
                resolution=self.vector_resolution,
                weight=lambda x: np.power(x[1] - x[0], 1),
            )
            vectorized_pd = SH.fit_transform(pd)[0]

        elif vec in ("PI", "PERSISTENCEIMAGE"):
            PI = PersistenceImage(
                resolution=(self.vector_resolution, self.vector_resolution),
                bandwidth=1.0,
            )
            PI.fit(pd)
            vectorized_pd = PI.transform(pd)[0]

        elif vec in ("BC", "BETTI", "BETTICURVE"):
            dgm = pd[0]
            vectorized_pd = betti_curve_from_dgm(dgm, resolution=self.vector_resolution)

        else:
            raise ValueError("Unknown vectorization: choose 'PI' / 'PL' / 'BC'.")

        return np.asarray(vectorized_pd, dtype=np.float32)

    def convolve(self, img: np.ndarray) -> list[np.ndarray]:
        width, length = img.shape
        w = self.window_size
        windows = []

        for x_cord in range(0, width, self.stride):
            for y_cord in range(0, length, self.stride):
                vec = self.process_window(img, x_cord, y_cord, w, w)
                windows.append(vec)

        return windows


# ===== Notebook code cell 12 =====

TRAIN_IMG_ROOT = "/home/imagea/DBs/Briancancer/train/train"
TEST_IMG_ROOT  = "/home/imagea/DBs/Briancancer/test/test"
OUT_PATH = "/home/imagea/zhibo/TopoFormer/Github/checkpoint/PHC/phc_briancancer_multiscale_pooled_train_test.npz"

PHC_IMG_HW = 128 

# PHC preprocessing: the PHC study evaluated thresholds in the 160--200 range
# and adopted k=200. The final extraction here uses k=200.
PHC_THRESHOLD = 200
PHC_DILATION_KERNEL = 2
PHC_DILATION_ITERATIONS = 1

SCALES = [
    (16, 16),
    (32, 32),
    (64, 64),
]

VEC_RES = 20
PERSIST_TYPE = "alpha"   # "extended" / "cubical" / "alpha"
VEC_TYPE = "PI"             # "PI" / "PL" / "BC" (Betti curve)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

"""
if CRC
TRAIN_IMG_ROOT = "/home/imagea/DBs/NCT-CRC-HE-100K"
TEST_IMG_ROOT  = "/home/imagea/DBs/CRC-VAL-HE-7K"

PHC_IMG_HW = 224

"""


# ===== Notebook code cell 15 =====

import os
import time
import json
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


MAX_HW = None  
MAX_WINDOWS_PER_SCALE = 256  
FULL_WINDOWS_ONLY = True  
SEED = 42
PHC_WORKERS = 0  
USE_GPU_POOLING = True  
POOL_DEVICE = "cuda" 
USE_PHC_FPN = True
FPN_UPSAMPLE_MODE = "nearest"  # "nearest" / "bilinear"
FPN_SMOOTH = False  

assert os.path.isdir(TRAIN_IMG_ROOT), f"not a dir: {TRAIN_IMG_ROOT}"
assert os.path.isdir(TEST_IMG_ROOT), f"not a dir: {TEST_IMG_ROOT}"


def _ensure_2d(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, but got shape={img.shape}")
    return img


def _resize_if_needed(img: np.ndarray, max_hw: int | None) -> np.ndarray:
    if max_hw is None:
        return img
    img = _ensure_2d(img)
    h, w = img.shape
    m = max(h, w)
    if m <= int(max_hw):
        return img
    scale = float(max_hw) / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def _read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img.astype(np.float32, copy=False)


def _scan_class_folders(root: str, exts=IMG_EXTS) -> tuple[list[str], list[str]]:
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes = sorted(classes)
    files: list[str] = []
    labels: list[str] = []
    for c in classes:
        d0 = os.path.join(root, c)
        fs = [
            f
            for f in os.listdir(d0)
            if os.path.isfile(os.path.join(d0, f)) and f.lower().endswith(exts)
        ]
        fs = sorted(fs)
        for f in fs:
            files.append(os.path.join(d0, f))
            labels.append(c)
    return files, labels


def _validate_scales(scales) -> list[tuple[int, int]]:
    if scales is None:
        raise ValueError("SCALES cannot be None")
    out = []
    for it in list(scales):
        if not (isinstance(it, (list, tuple)) and len(it) == 2):
            raise ValueError(f"Each element of SCALES must be (window, stride), but what was obtained was: {it}")
        w, s = int(it[0]), int(it[1])
        if w <= 0 or s <= 0:
            raise ValueError(f"Invalid scale (window,stride)=({w},{s})")
        out.append((w, s))
    return out


SCALES = _validate_scales(globals().get("SCALES", [(32, 32)]))
print("SCALES:", SCALES)

train_files, train_labels = _scan_class_folders(TRAIN_IMG_ROOT)
test_files, test_labels = _scan_class_folders(TEST_IMG_ROOT)

print(f"[train] images={len(train_files)} classes={sorted(set(train_labels))}")
print(f"[test ] images={len(test_files)} classes={sorted(set(test_labels))}")
assert len(train_files) > 0 and len(test_files) > 0, "No images found in train or test directories"

train_classes = sorted(set(train_labels))
test_classes = sorted(set(test_labels))
if set(test_classes) - set(train_classes):
    print("[WARNING] The test data contains categories that are not present in the training data:", sorted(set(test_classes) - set(train_classes)))


def _count_windows(h: int, w: int, window: int, stride: int, full_only: bool) -> int:
    if full_only:
        nx = max(0, (h - window) // stride + 1)
        ny = max(0, (w - window) // stride + 1)
    else:
        nx = (h + stride - 1) // stride
        ny = (w + stride - 1) // stride
    return int(nx * ny)


def _phc_windows(
    localhom: "PHC",
    img: np.ndarray,
    window: int,
    stride: int,
    full_only: bool = True,
    max_windows: int | None = None,
    seed: int = 0,
    workers: int = 0,
    return_meta: bool = False,
 ) -> np.ndarray:
    img = _ensure_2d(img)
    h, w = img.shape
    if full_only:
        xs = list(range(0, max(0, h - window + 1), stride))
        ys = list(range(0, max(0, w - window + 1), stride))
    else:
        xs = list(range(0, h, stride))
        ys = list(range(0, w, stride))

    coords = [(x, y) for x in xs for y in ys]
    if len(coords) == 0:
        coords = [(0, 0)]

    if max_windows is not None and len(coords) > int(max_windows):
        rng = np.random.default_rng(int(seed))
        sel = rng.choice(len(coords), size=int(max_windows), replace=False)
        coords = [coords[i] for i in sel]

    def _calc_one(xy):
        x, y = xy
        return localhom.process_window(img, x, y, window, window)

    workers = int(workers or 0)
    if workers <= 1:
        feats = [_calc_one(xy) for xy in coords]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            feats = list(ex.map(_calc_one, coords))

    feats = np.asarray(feats)
    if not return_meta:
        return feats
    return feats, xs, ys, coords


def _pool_windows(windows: np.ndarray) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim == 0:
        return np.asarray(windows, dtype=np.float32)
    if windows.size == 0:
        return np.zeros((1,), dtype=np.float32)
    if not USE_GPU_POOLING:
        return windows.mean(axis=0)
    try:
        import torch
        if POOL_DEVICE == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        pooled = torch.from_numpy(windows).to(device).mean(dim=0).detach().cpu().numpy()
        return pooled
    except Exception:
        return windows.mean(axis=0)


def _windows_to_feature_map(
    windows: np.ndarray,
    xs: list[int],
    ys: list[int],
    coords: list[tuple[int, int]],
 ) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float32)
    if windows.size == 0:
        return np.zeros((1, max(1, len(xs)), max(1, len(ys))), dtype=np.float32)
    n = int(windows.shape[0])
    wflat = windows.reshape(n, -1)
    c = int(wflat.shape[1])
    hx, wy = int(len(xs)), int(len(ys))
    fmap = np.zeros((c, hx, wy), dtype=np.float32)
    x_to_i = {int(x): i for i, x in enumerate(xs)}
    y_to_j = {int(y): j for j, y in enumerate(ys)}
    for k, (x, y) in enumerate(coords):
        ii = x_to_i.get(int(x), None)
        jj = y_to_j.get(int(y), None)
        if ii is None or jj is None:
            continue
        fmap[:, ii, jj] = wflat[k]
    return fmap


def _upsample_to(x: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """x: (C,H,W) -> upsample 到 size_hw (H,W)。"""
    x = np.asarray(x, dtype=np.float32)
    c, h, w = x.shape
    th, tw = int(size_hw[0]), int(size_hw[1])
    if h == th and w == tw:
        return x
    try:
        import torch
        import torch.nn.functional as F
        mode = str(FPN_UPSAMPLE_MODE)
        xt = torch.from_numpy(x[None, ...])  # (1,C,H,W)
        if mode == "nearest":
            yt = F.interpolate(xt, size=(th, tw), mode=mode)
        else:
            yt = F.interpolate(xt, size=(th, tw), mode=mode, align_corners=False)
        return yt[0].detach().cpu().numpy()
    except Exception:
        rh = max(1, th // max(1, h))
        rw = max(1, tw // max(1, w))
        y = np.repeat(np.repeat(x, rh, axis=1), rw, axis=2)
        return y[:, :th, :tw]


def _fpn_fuse_maps(maps: list[np.ndarray]) -> list[np.ndarray]:
    if len(maps) == 0:
        return []
    pyr = [None for _ in range(len(maps))]
    pyr[-1] = np.asarray(maps[-1], dtype=np.float32)
    for i in range(len(maps) - 2, -1, -1):
        ci = np.asarray(maps[i], dtype=np.float32)
        up = _upsample_to(pyr[i + 1], (ci.shape[1], ci.shape[2]))
        pyr[i] = ci + up
    if FPN_SMOOTH:
        try:
            import torch
            import torch.nn.functional as F
            out = []
            for p in pyr:
                pt = torch.from_numpy(p[None, ...])
                sm = F.avg_pool2d(pt, kernel_size=3, stride=1, padding=1)
                out.append(sm[0].detach().cpu().numpy())
            pyr = out
        except Exception:
            pass
    return pyr


phc_preprocess = preprocess(
    thresh=PHC_THRESHOLD,
    kernel_size=PHC_DILATION_KERNEL,
    iterate=PHC_DILATION_ITERATIONS,
)


localhom = PHC(
    persistence_type=PERSIST_TYPE,
    window_size=(SCALES[0][0] if len(SCALES) else 32),
    stride=(SCALES[0][1] if len(SCALES) else 32),
    vectorization=VEC_TYPE,
    vector_resolution=VEC_RES,
 )


def _extract_one(path: str, seed: int) -> np.ndarray:
    img = _read_gray(path)
    img = _resize_if_needed(img, MAX_HW)
    if PHC_IMG_HW is not None:
        img = cv2.resize(img, (int(PHC_IMG_HW), int(PHC_IMG_HW)), interpolation=cv2.INTER_AREA).astype(np.float32)
    img = _ensure_2d(img)
    img = phc_preprocess.threshold(img)
    img = phc_preprocess.dilate(img)
    img = _ensure_2d(img)

    feats_scales: list[np.ndarray] = []
    maps_scales: list[np.ndarray] = []

    for (window, stride) in SCALES:
        if USE_PHC_FPN:
            windows, xs, ys, coords = _phc_windows(
                localhom,
                img,
                window=window,
                stride=stride,
                full_only=FULL_WINDOWS_ONLY,
                max_windows=MAX_WINDOWS_PER_SCALE,
                seed=int(seed + window * 10_000 + stride * 100),
                workers=PHC_WORKERS,
                return_meta=True,
            )
            fmap = _windows_to_feature_map(windows, xs=xs, ys=ys, coords=coords)
            maps_scales.append(fmap)
        else:
            windows = _phc_windows(
                localhom,
                img,
                window=window,
                stride=stride,
                full_only=FULL_WINDOWS_ONLY,
                max_windows=MAX_WINDOWS_PER_SCALE,
                seed=int(seed + window * 10_000 + stride * 100),
                workers=PHC_WORKERS,
            )
            pooled = _pool_windows(windows)
            feats_scales.append(np.asarray(pooled, dtype=np.float32).reshape(-1))

    if USE_PHC_FPN:
        pyr = _fpn_fuse_maps(maps_scales)
        feats_scales = [p.mean(axis=(1, 2)).reshape(-1).astype(np.float32) for p in pyr]

    topo_vec = np.concatenate(feats_scales, axis=0)
    return topo_vec.astype(np.float32, copy=False)


def _extract_split(files: list[str], split_name: str, seed0: int) -> np.ndarray:
    pooled_list = []
    t0_all = time.time()
    for i, p in enumerate(tqdm(files, desc=f"Topo multiscale ({split_name})"), start=1):
        pooled = _extract_one(p, seed=seed0 + i)
        pooled_list.append(pooled)
        if i <= 2 or i % 500 == 0 or i == len(files):
            print(f"[{split_name} {i}/{len(files)}] topo_dim={int(pooled.shape[0])}")
    X = np.stack(pooled_list, axis=0)
    print(f"[{split_name}] topo_data:", X.shape, X.dtype, "total_time=", f"{time.time()-t0_all:.1f}s")
    return X

try:
    img0 = _read_gray(train_files[0])
    img0 = _resize_if_needed(img0, MAX_HW)
    if PHC_IMG_HW is not None:
        img0 = cv2.resize(img0, (int(PHC_IMG_HW), int(PHC_IMG_HW)), interpolation=cv2.INTER_AREA)
    for (window, stride) in SCALES:
        est = _count_windows(img0.shape[0], img0.shape[1], window, stride, FULL_WINDOWS_ONLY)
        print(f"Scale window={window:>3d} stride={stride:>3d} -> est_windows={est:<5d} | cap={MAX_WINDOWS_PER_SCALE}")
except Exception as e:
    print("[WARN] estimate windows failed:", repr(e))


X_phc_train = _extract_split(train_files, "train", seed0=SEED)
X_phc_test = _extract_split(test_files, "test", seed0=SEED + 10_000)

meta = dict(
    train_root=str(TRAIN_IMG_ROOT),
    test_root=str(TEST_IMG_ROOT),
    phc_img_hw=(None if PHC_IMG_HW is None else int(PHC_IMG_HW)),
    phc_threshold=int(PHC_THRESHOLD),
    phc_dilation_kernel=int(PHC_DILATION_KERNEL),
    phc_dilation_iterations=int(PHC_DILATION_ITERATIONS),
    scales=[(int(w), int(s)) for (w, s) in SCALES],
    vec_res=int(VEC_RES),
    persist_type=str(PERSIST_TYPE),
    vec_type=str(VEC_TYPE),
    max_hw=(None if MAX_HW is None else int(MAX_HW)),
    max_windows_per_scale=(None if MAX_WINDOWS_PER_SCALE is None else int(MAX_WINDOWS_PER_SCALE)),
    full_windows_only=bool(FULL_WINDOWS_ONLY),
    seed=int(SEED),
    use_phc_fpn=bool(USE_PHC_FPN),
    fpn_upsample_mode=str(FPN_UPSAMPLE_MODE),
    fpn_smooth=bool(FPN_SMOOTH),
)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
np.savez_compressed(
    OUT_PATH,
    meta=np.asarray([json.dumps(meta, ensure_ascii=False)], dtype=str),
    train_files=np.asarray(train_files, dtype=str),
    train_labels=np.asarray(train_labels, dtype=str),
    train_phc=X_phc_train,
    test_files=np.asarray(test_files, dtype=str),
    test_labels=np.asarray(test_labels, dtype=str),
    test_phc=X_phc_test,
)
print("Saved:", OUT_PATH)

