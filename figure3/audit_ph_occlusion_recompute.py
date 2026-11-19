"""Load the PH functions used for Figure 3 from the released extraction source."""
import ast
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import gudhi as gd
import numpy as np
from gudhi import AlphaComplex, CubicalComplex
from gudhi.representations import PersistenceImage, Silhouette


FUNCTIONS = {
    "remove_noisy_pts", "pointcloud2D", "betti_curve_from_dgm",
    "alphacomplex", "cubicalcomplex", "ext_persistence", "preprocess", "PHC",
    "_ensure_2d", "_resize_if_needed", "_read_gray", "_phc_windows",
    "_pool_windows", "_windows_to_feature_map", "_upsample_to",
    "_fpn_fuse_maps", "_extract_one",
}

def _load_original_functions(source: Path, meta: dict) -> dict:
    source_bytes = source.read_bytes()
    parsed = ast.parse(source_bytes, filename=str(source))
    definitions = [
        node for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in FUNCTIONS
    ]
    found = {node.name for node in definitions}
    if found != FUNCTIONS:
        raise RuntimeError(f"Source functions changed: missing={FUNCTIONS - found}")
    namespace = {
        "__name__": "original_ph_functions_only",
        "np": np, "cv2": cv2, "gd": gd,
        "AlphaComplex": AlphaComplex, "CubicalComplex": CubicalComplex,
        "PersistenceImage": PersistenceImage, "Silhouette": Silhouette,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "MAX_HW": meta["max_hw"],
        "PHC_IMG_HW": meta["phc_img_hw"],
        "PHC_THRESHOLD": meta["phc_threshold"],
        "PHC_DILATION_KERNEL": meta["phc_dilation_kernel"],
        "PHC_DILATION_ITERATIONS": meta["phc_dilation_iterations"],
        "SCALES": [tuple(v) for v in meta["scales"]],
        "MAX_WINDOWS_PER_SCALE": meta["max_windows_per_scale"],
        "FULL_WINDOWS_ONLY": meta["full_windows_only"],
        "USE_PHC_FPN": meta["use_phc_fpn"],
        "FPN_UPSAMPLE_MODE": meta["fpn_upsample_mode"],
        "FPN_SMOOTH": meta["fpn_smooth"],
        "MAP_ALIGN_TO_L0": meta["map_align_to_l0"],
        "PHC_WORKERS": 0,
        "USE_GPU_POOLING": False,
        "POOL_DEVICE": "cpu",
    }
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    namespace["phc_preprocess"] = namespace["preprocess"](
        thresh=meta["phc_threshold"],
        kernel_size=meta["phc_dilation_kernel"],
        iterate=meta["phc_dilation_iterations"],
    )
    namespace["localhom"] = namespace["PHC"](
        persistence_type=meta["persist_type"],
        window_size=meta["scales"][0][0],
        stride=meta["scales"][0][1],
        vectorization=meta["vec_type"],
        vector_resolution=meta["vec_res"],
    )
    namespace["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    return namespace
