import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def read_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(path, config):
    value = os.fspath(path)
    for source, target in sorted(config.get("path_roots", {}).items(), key=lambda item: -len(item[0])):
        if value == source or value.startswith(source.rstrip("/") + "/"):
            value = target.rstrip("/") + value[len(source):]
            break
    return os.path.realpath(value)


def asset(template, threshold):
    if isinstance(template, dict):
        template = template.get(str(threshold), template["default"])
    directory = "PHC" if threshold == 200 else f"PHC_{threshold}"
    return template.format(threshold=threshold, threshold_dir=directory)


def map_paths(paths, config):
    return np.asarray([canonical(path, config) for path in paths], dtype=str)


def _write_map(source, destination, level, compact, fp16):
    version = np.lib.format.read_magic(source)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(source)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(source)
    else:
        raise ValueError(f"Unsupported NPY version {version}")
    if len(shape) != 4 or fortran or dtype.kind != "f":
        raise ValueError(f"Expected C-order floating PH array, got shape={shape}, dtype={dtype}")

    count, channels, height, width = shape
    native_h, native_w = (height // (2 ** level), width // (2 ** level)) if compact else (height, width)
    if min(native_h, native_w) < 1:
        raise ValueError(f"L{level} cannot be packed from {height}x{width}")
    sample_rows = (np.arange(native_h) * height + native_h - 1) // native_h
    sample_cols = (np.arange(native_w) * width + native_w - 1) // native_w
    expand_rows = np.arange(height) * native_h // height
    expand_cols = np.arange(width) * native_w // width
    output = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.float16 if fp16 else dtype,
        shape=(count, channels, native_h, native_w),
    )
    item_bytes = channels * height * width * dtype.itemsize
    batch = max(1, 8 * 1024 * 1024 // item_bytes)
    bits = np.dtype(f"u{dtype.itemsize}")
    for start in range(0, count, batch):
        stop = min(count, start + batch)
        block = np.frombuffer(source.read((stop - start) * item_bytes), dtype=dtype)
        block = block.reshape(stop - start, channels, height, width)
        if (native_h, native_w) != (height, width):
            samples = block[..., sample_rows[:, None], sample_cols[None, :]]
            expanded = samples[..., expand_rows[:, None], expand_cols[None, :]]
            if not np.array_equal(block.view(bits), expanded.view(bits)):
                raise ValueError(f"L{level} maps are not repeated nearest-neighbor native grids at row {start}")
        else:
            samples = block
        if fp16:
            samples = samples.astype(np.float32, copy=False)
            if np.any(np.isfinite(samples) & (np.abs(samples) >= 65520)):
                raise ValueError(f"L{level} maps contain finite values that overflow float16 at row {start}")
        output[start:stop] = samples
    output.flush()
    if source.read(1):
        raise ValueError(f"Unexpected trailing bytes in {destination.name}")
    return [height, width]


def prepare_maps(prefix, compact=False, fp16=False):
    for level in range(4):
        source = Path(f"{prefix}_L{level}.npz")
        target = source.with_suffix(".mmap")
        if target.is_dir():
            continue
        with tempfile.TemporaryDirectory(prefix=target.name + ".", dir=target.parent) as work:
            stage = Path(work) / "maps"
            stage.mkdir()
            with ZipFile(source) as archive:
                grids = {}
                for key in ("train_files", "test_files", "meta"):
                    if key == "meta" and "meta.npy" not in archive.namelist():
                        continue
                    with archive.open(f"{key}.npy") as item, (stage / f"{key}.npy").open("wb") as output:
                        shutil.copyfileobj(item, output, length=8 * 1024 * 1024)
                for split in ("train", "test"):
                    key = f"{split}_phc_map_level.npy"
                    path = stage / key
                    with archive.open(key) as item:
                        if compact or fp16:
                            grids[split] = _write_map(item, path, level, compact, fp16)
                        else:
                            with path.open("wb") as output:
                                shutil.copyfileobj(item, output, length=8 * 1024 * 1024)
                if grids:
                    (stage / "full_grid.json").write_text(json.dumps(grids), encoding="utf-8")
            stage.rename(target)
        print(target, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare shared read-only PH map arrays from the legacy NPZ")
    parser.add_argument("prefix", help="PH prefix before _L0.npz")
    parser.add_argument("--compact", action="store_true", help="Store native grids after checking every sample")
    parser.add_argument("--fp16", action="store_true", help="Store the same half precision used by GPU inputs")
    args = parser.parse_args()
    prepare_maps(args.prefix, compact=args.compact, fp16=args.fp16)


if __name__ == "__main__":
    main()
