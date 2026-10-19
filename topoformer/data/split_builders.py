import os
from dataclasses import dataclass
from collections import Counter
from typing import List, Tuple, Dict, Optional

import numpy as np


@dataclass
class SplitData:
    paths: List[str]
    types_np: np.ndarray
    label_names: List[str]
    id_to_domain: Dict[int, str]
    domain_to_id: Dict[str, int]
    y_all: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    label_counts: Counter
    val_idx: Optional[np.ndarray] = None


def _scan_class_folders_png(
    root: str,
    exts=(".png",),
    exclude: Optional[Tuple[str, ...]] = None,
) -> Tuple[List[str], List[str]]:
    assert os.path.isdir(root), f"not found: {root}"

    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if exclude:
        ex_set = set(str(x) for x in exclude)
        classes = [c for c in classes if c not in ex_set]
    classes = sorted(classes)

    paths: List[str] = []
    labels: List[str] = []

    for c in classes:
        d0 = os.path.join(root, c)
        fs = [
            f
            for f in os.listdir(d0)
            if os.path.isfile(os.path.join(d0, f)) and f.lower().endswith(exts)
        ]
        fs = sorted(fs)
        for f in fs:
            paths.append(os.path.join(d0, f))
            labels.append(str(c))

    assert len(paths) > 0, f"empty split: {root}"
    return paths, labels


def _label_from_filename_prefix(path: str) -> str:
    base = os.path.basename(str(path))
    if "-" in base:
        return base.split("-", 1)[0]
    return os.path.splitext(base)[0]


def _scan_files_by_prefix_label(root: str, exts=(".tif", ".tiff")) -> Tuple[List[str], List[str]]:
    assert os.path.isdir(root), f"not found: {root}"

    paths: List[str] = []
    labels: List[str] = []
    exts_l = tuple(str(e).lower() for e in exts)

    for dp, _dns, fns in os.walk(root):
        for fn in fns:
            if not str(fn).lower().endswith(exts_l):
                continue
            p = os.path.join(dp, fn)
            paths.append(p)
            labels.append(_label_from_filename_prefix(p))

    assert len(paths) > 0, f"empty split: {root}"
    return list(paths), list(labels)


def build_briancancer_train_test(
    train_dir: str = "/home/imagea/DBs/Briancancer/train/train",
    test_dir: str = "/home/imagea/DBs/Briancancer/test/test",
) -> SplitData:
    img_exts = (".png",)

    train_paths, train_labels = _scan_class_folders_png(train_dir, exts=img_exts)
    # The test root may contain non-class subfolders (e.g. robustness_qc
    # for QC artefacts, or perturbed splits such as patch_shuffle /
    # stain_shift that mirror the class structure). Filter them out
    # explicitly so they don't pollute the test labels and trigger the
    # "unseen categories" assertion below.
    test_paths, test_labels = _scan_class_folders_png(
        test_dir, exts=img_exts,
        exclude=("robustness_qc", "patch_shuffle", "stain_shift"),
    )

    label_names = sorted(set(train_labels))
    domain_to_id = {name: i for i, name in enumerate(label_names)}
    id_to_domain = {i: name for name, i in domain_to_id.items()}

    unknown = sorted(set(test_labels) - set(label_names))
    if len(unknown) > 0:
        raise AssertionError(f"Test set contains unseen categories: {unknown}")

    y_train = np.asarray([domain_to_id[str(x)] for x in train_labels], dtype=np.int64)
    y_test = np.asarray([domain_to_id[str(x)] for x in test_labels], dtype=np.int64)

    paths = list(train_paths) + list(test_paths)
    y_all = np.concatenate([y_train, y_test], axis=0)
    types_np = np.asarray(list(train_labels) + list(test_labels), dtype=object)
    label_counts = Counter(types_np.tolist())

    n_train = int(len(train_paths))
    n_test = int(len(test_paths))
    n_all = int(n_train + n_test)

    train_idx_all = np.arange(0, n_train, dtype=np.int64)
    test_idx = np.arange(n_train, n_all, dtype=np.int64)

    from sklearn.model_selection import train_test_split

    train_idx, val_idx = train_test_split(
        train_idx_all,
        test_size=0.1,
        stratify=y_train,
        random_state=42,
    )
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)

    val_label_counts = Counter([str(id_to_domain[int(y_train[int(c)])]) for c in val_idx])
    print("[info] Briancancer train/test split")
    print("[info] train dir:", train_dir, "N=", n_train)
    print("[info] test dir:", test_dir, "N=", n_test)
    print("[info] classes:", id_to_domain)
    print("[info] type counts:", {k: int(v) for k, v in label_counts.items()})
    print(f"[info] actual train (90% of train): N={int(train_idx.size)} val (10% stratified): N={int(val_idx.size)}")
    print("[info] val per-class counts:", {k: int(v) for k, v in val_label_counts.items()})

    return SplitData(
        paths=paths,
        types_np=types_np,
        label_names=label_names,
        id_to_domain=id_to_domain,
        domain_to_id=domain_to_id,
        y_all=y_all,
        train_idx=train_idx,
        test_idx=test_idx,
        val_idx=val_idx,
        label_counts=label_counts,
    )


def build_nctcrc100k_trainval_crcval7k_test(
    trainval_dir: str = "/home/imagea/DBs/NCT-CRC-HE-100K",
    test_dir: str = "/home/imagea/DBs/CRC-VAL-HE-7K",
) -> SplitData:
    img_exts = (".tif", ".tiff")

    trainval_paths, trainval_labels = _scan_files_by_prefix_label(trainval_dir, exts=img_exts)
    test_paths, test_labels = _scan_files_by_prefix_label(test_dir, exts=img_exts)

    label_names = sorted(set([str(x) for x in trainval_labels]))
    domain_to_id = {name: i for i, name in enumerate(label_names)}
    id_to_domain = {i: name for name, i in domain_to_id.items()}

    unknown = sorted(set([str(x) for x in test_labels]) - set(label_names))
    if len(unknown) > 0:
        raise AssertionError(f"Test set contains unseen categories: {unknown}")

    y_trainval = np.asarray([domain_to_id[str(x)] for x in trainval_labels], dtype=np.int64)
    y_test = np.asarray([domain_to_id[str(x)] for x in test_labels], dtype=np.int64)

    paths = list(trainval_paths) + list(test_paths)
    y_all = np.concatenate([y_trainval, y_test], axis=0)
    types_np = np.asarray(list(trainval_labels) + list(test_labels), dtype=object)
    label_counts = Counter([str(x) for x in types_np.tolist()])

    n_trainval = int(len(trainval_paths))
    n_test = int(len(test_paths))
    n_all = int(n_trainval + n_test)

    train_idx = np.arange(0, n_trainval, dtype=np.int64)
    test_idx = np.arange(n_trainval, n_all, dtype=np.int64)

    print("[info] NCT-CRC-HE-100K train + CRC-VAL-HE-7K test split")
    print("[info] trainval dir:", trainval_dir, "N=", n_trainval)
    print("[info] test dir:", test_dir, "N=", n_test)
    print("[info] classes:", id_to_domain)
    print("[info] type counts:", {k: int(v) for k, v in label_counts.items()})

    return SplitData(
        paths=paths,
        types_np=types_np,
        label_names=label_names,
        id_to_domain=id_to_domain,
        domain_to_id=domain_to_id,
        y_all=y_all,
        train_idx=train_idx,
        test_idx=test_idx,
        val_idx=None,
        label_counts=label_counts,
    )