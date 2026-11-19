import json
import os
from pathlib import Path
from typing import Dict

import numpy as np


def norm_path(p: str) -> str:
    return os.path.realpath(str(p))


class ExpandedMapRows:
    """Expose the old aligned grid while keeping the native grid on disk."""

    def __init__(self, rows, height, width):
        self.rows = rows
        self.shape = (*rows.shape[:-2], height, width)
        self.row_index = np.arange(height) * rows.shape[-2] // height
        self.col_index = np.arange(width) * rows.shape[-1] // width

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        values = self.rows[index][..., self.row_index[:, None], self.col_index[None, :]]
        return values.astype(np.float32, copy=False)


def open_mmap_rows(mmap, split):
    maps = np.load(mmap / f"{split}_phc_map_level.npy", mmap_mode="r", allow_pickle=False)
    geometry = mmap / "full_grid.json"
    if geometry.is_file():
        height, width = json.loads(geometry.read_text(encoding="utf-8"))[split]
        if maps.shape[-2:] != (height, width) or maps.dtype == np.float16:
            return ExpandedMapRows(maps, height, width)
    return maps


class TopoMapStore:
    def __init__(self, topo_npz_base: str, path_key=norm_path, concat=False, test_base=None, test_only=False):
        self.path_key = path_key
        self.concat = concat
        self.test_base = test_base
        self.test_only = test_only
        self.topo_npz_base = str(topo_npz_base).strip()
        self.levels = {f"L{i}": self._load_level_npz(i) for i in [0, 1, 2, 3]}
        self.topo_c = sum(level["C"] for level in self.levels.values()) if concat else int(self.levels["L0"]["C"])
        self.topo_hw = tuple(self.levels["L0"]["HW"])

        print(
            "[topo] loaded maps:",
            {k: v["path"] for k, v in self.levels.items()},
            "| C=",
            self.topo_c,
            "HW=",
            self.topo_hw,
        )

    def _load_level_npz(self, level: int) -> Dict:
        path = f"{self.topo_npz_base}_L{int(level)}.npz"
        def load_split(base, split):
            file = Path(f"{base}_L{level}.npz")
            mmap = file.with_suffix(".mmap")
            if mmap.is_dir():
                files = np.load(mmap / f"{split}_files.npy", allow_pickle=False)
                maps = open_mmap_rows(mmap, split)
            else:
                with np.load(file, allow_pickle=False) as archive:
                    files = archive[f"{split}_files"]
                    maps = archive[f"{split}_phc_map_level"]
            return np.asarray(files, dtype=str), maps

        test_files, test_maps = load_split(self.test_base or self.topo_npz_base, "test")
        assert test_maps.shape[0] == test_files.shape[0]
        if self.test_only:
            train_files, train_maps = np.empty(0, dtype=str), None
        else:
            train_files, train_maps = load_split(self.topo_npz_base, "train")
            assert train_maps.shape[0] == train_files.shape[0]
            assert train_maps.shape[1:] == test_maps.shape[1:]
        _, c, h, w = test_maps.shape
        n_train = int(train_files.shape[0])

        paths = [self.path_key(p) for p in np.concatenate((train_files, test_files))]
        path2idx = {p: i for i, p in enumerate(paths)}
        if len(path2idx) != len(paths):
            raise ValueError(f"Duplicate image IDs in {path}")

        return {
            "path": path,
            "C": c,
            "HW": (h, w),
            "n_train": n_train,
            "train": train_maps,
            "test": test_maps,
            "path2idx": path2idx,
        }

    def get_topo_map(self, level_key: str, npath: str) -> np.ndarray:
        info = self.levels[str(level_key)]
        gi = int(info["path2idx"][npath])
        if gi < info["n_train"]:
            value = info["train"][gi]
        else:
            value = info["test"][gi - info["n_train"]]
        return np.array(value, dtype=np.float32, copy=True)

    def get_all_levels(self, path: str) -> Dict[str, np.ndarray]:
        npath = norm_path(path)
        levels = {key: self.get_topo_map(key, npath) for key in self.levels}
        if self.concat:
            merged = np.concatenate(tuple(levels.values()), axis=0)
            return dict.fromkeys(levels, merged)
        return levels

    def check_alignment(self, image_paths) -> None:
        missing = []
        for p in image_paths:
            pp = norm_path(p)
            ok = all(pp in self.levels[f"L{i}"]["path2idx"] for i in [0, 1, 2, 3])
            if not ok:
                missing.append(pp)

        if len(missing) > 0:
            print("[topo][WARN] missing topo for paths (show up to 10):")
            for x in missing[:10]:
                print("  ", x)
            raise AssertionError(f"Topo features missing for {len(missing)}/{len(image_paths)} images")

        print("[topo] path alignment OK")
