"""CPU checks for the compact PH and selected-checkpoint formats."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn

from figure3.bridge import test_maps as figure_test_maps
from materials import prepare_maps
from topoformer.data.topo_maps import TopoMapStore, norm_path
from topoformer.model.checkpoints import compact_state, dino_sha256, load_selected_weights


def check_maps(root, grid, fp16):
    rng = np.random.default_rng(8)
    prefix = str(root / "maps")
    expected = []
    paths = [str(root / f"image_{index}") for index in range(4)]
    for level in range(4):
        native = grid // (2 ** level)
        values = rng.normal(size=(4, 400, native, native)).astype(np.float32)
        values[0, 0, 0, 0] = np.nan
        values[1, 1, 0, 0] = np.inf
        values[2, 0, 0, 0] = -np.inf
        values[3, 0, 0, 0] = np.nan
        index = np.arange(grid) * native // grid
        full = values[..., index[:, None], index[None, :]]
        np.savez_compressed(
            f"{prefix}_L{level}.npz", train_files=paths[:2], test_files=paths[2:],
            train_phc_map_level=full[:2], test_phc_map_level=full[2:],
            meta=['{"phc_threshold":180}'],
        )
        expected.append(full)

    prepare_maps(prefix, compact=True, fp16=fp16)
    for source in root.glob("*.npz"):
        source.unlink()
    store = TopoMapStore(prefix)
    assert store.topo_hw == (grid, grid)
    for level, full in enumerate(expected):
        _, maps, metadata = figure_test_maps(prefix, level, {})
        assert maps.shape == (2, 400, grid, grid) and metadata["phc_threshold"] == 180
        with np.errstate(over="ignore"):
            # Match the legacy figure evaluator's default NumPy sanitization.
            np.testing.assert_array_equal(
                np.nan_to_num(maps[:]).astype(np.float16),
                np.nan_to_num(full[2:]).astype(np.float16),
            )
        for index, path in enumerate(paths):
            row = store.get_topo_map(f"L{level}", norm_path(path))
            if fp16:
                # Match the actual dataset transform before training/export.
                before = torch.from_numpy(full[index]).nan_to_num(nan=0, posinf=0, neginf=0).half()
                after = torch.from_numpy(row).nan_to_num(nan=0, posinf=0, neginf=0).half()
                assert torch.equal(before, after)
            else:
                np.testing.assert_array_equal(row, full[index])


def check_checkpoint(root):
    pretrained = root / "pretrained.pt"
    torch.save(nn.Linear(4, 4).state_dict(), pretrained)
    config = {"dinov3": str(pretrained)}
    fingerprint = dino_sha256(config)
    receipt = root / "pretrained.sha256"
    receipt.write_text(f"{fingerprint}  pretrained.pt\n", encoding="ascii")
    config["dinov3_sha256_file"] = str(receipt)

    def build():
        model = nn.Module()
        model.vit = nn.Module()
        model.vit.encoder = nn.Module()
        model.vit.encoder.base = nn.Linear(4, 4)
        model.vit.encoder.base.load_state_dict(torch.load(pretrained, weights_only=True))
        model.vit.encoder.base.requires_grad_(False)
        model.vit.encoder.adapter = nn.Linear(4, 4)
        model.vit.encoder.register_buffer("counter", torch.zeros(()))
        model.head = nn.Linear(4, 2).requires_grad_(False)
        return model

    saved, from_full, from_compact = build(), build(), build()
    saved.vit.encoder.counter += 1
    full, compact = root / "full.pt", root / "compact.pt"
    torch.save(saved.state_dict(), full)
    payload = compact_state(saved, fingerprint)
    assert set(payload["omitted"]) == {"vit.encoder.base.weight", "vit.encoder.base.bias"}
    torch.save(payload, compact)
    load_selected_weights(from_full, full)
    load_selected_weights(from_compact, compact, config)
    for key, tensor in from_full.state_dict().items():
        assert torch.equal(from_compact.state_dict()[key], tensor), key
    receipt.write_text(f'{"0" * 64}  pretrained.pt\n', encoding="ascii")
    try:
        load_selected_weights(from_compact, compact, config)
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong pretrained checksum was accepted")


if __name__ == "__main__":
    for grid in (8, 14):
        for fp16 in (False, True):
            with TemporaryDirectory() as temporary:
                check_maps(Path(temporary), grid, fp16)
    with TemporaryDirectory() as temporary:
        check_checkpoint(Path(temporary))
    print("STORAGE_CHECKS_PASS")
