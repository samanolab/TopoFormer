from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from analysis_tables import predictions, stats
from jobs import stage_jobs
from materials import map_paths, prepare_maps
from report import measures
from topoformer.data.topo_maps import TopoMapStore
from topoformer.engine.preflight import preflight
from topoformer.model.four_stage_cti import FourStageCTI


def check_backbone():
    sys.path.insert(0, str(Path(__file__).resolve().parent / "Backbone" / "dinov3" / "dinov3"))
    from dinov3.models.vision_transformer import DinoVisionTransformer

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = nn.Module()
            self.vit.encoder = DinoVisionTransformer(
                embed_dim=32, num_heads=4, depth=40, ffn_ratio=2, n_storage_tokens=4,
            )
            self.vit.encoder.init_weights()
            self.vit.encoder.requires_grad_(False)

        def forward(self, x, topo_levels):
            out = self.vit.encoder.forward_features(x, topo_levels=topo_levels)
            return {"f_disc": out["x_norm_clstoken"]}

    torch.manual_seed(42)
    model = Model()
    for grid, levels, concat in (
        (8, (0, 1, 2, 3), False), (14, (0, 1, 2, 3), False),
        *((8, (i,), False) for i in range(4)), (8, (0, 1, 2, 3), True),
    ):
        hook = FourStageCTI("local", 32, 5, grid, levels=levels, concat=concat)
        model.vit.encoder._topo_hook = hook
        image = torch.randn(4, 3, 224, 224)
        maps = {}
        for i in range(4):
            size = grid if concat else grid // 2 ** i
            native = torch.randn(4, 5, size, size)
            maps[f"L{i}"] = nn.functional.interpolate(native, size=(grid, grid), mode="nearest")
        loader = DataLoader([(image[i], 0, {k: v[i] for k, v in maps.items()}) for i in range(4)], batch_size=4)
        preflight(model, loader, "cpu")
        optimizer = torch.optim.SGD(hook.parameters(), lr=0.1)
        output = model(image, maps)["f_disc"]
        (output * torch.randn_like(output)).sum().backward()
        watched = [hook.stages[i].out.weight for i in levels]
        if 3 in levels:
            watched.append(hook.stages[3].cls_out.weight)
        before = [param.detach().clone() for param in watched]
        for param in watched:
            assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().max() > 0
        optimizer.step()
        assert all(not torch.equal(param, old) for param, old in zip(watched, before))


def check_maps(root):
    prefix = str(root / "map")
    train_file, test_file = str(root / "a"), str(root / "b")
    maps = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    for i in range(4):
        np.savez_compressed(f"{prefix}_L{i}.npz", train_files=[train_file], test_files=[test_file],
                            train_phc_map_level=maps + i, test_phc_map_level=maps + 2 * i)
    eager = TopoMapStore(prefix)
    prepare_maps(prefix)
    lazy = TopoMapStore(prefix)
    concat = TopoMapStore(prefix, concat=True)
    assert isinstance(lazy.levels["L0"]["train"], np.memmap)
    for path in (train_file, test_file):
        for key, array in eager.get_all_levels(path).items():
            np.testing.assert_array_equal(array, lazy.get_all_levels(path)[key])
        assert concat.get_all_levels(path)["L0"].shape == (4, 2, 2)


def check_statistics(root):
    y = np.repeat(np.arange(3), 20)
    probabilities = np.eye(3)[y] * 0.9 + 0.1 / 3
    np.testing.assert_array_equal(measures(y, probabilities), np.ones(6))
    paths = np.array([str(root / f"{i}.tif") for i in range(len(y))])
    names = np.array(["a", "b", "c"])
    files = []
    for i in range(3):
        file = root / f"{i}.npz"
        perm = np.roll(np.arange(len(y)), i)
        np.savez_compressed(file, paths=paths[perm], y_true=y[perm], classes=names, probabilities=probabilities[perm])
        files.append(file)
    _, aligned = predictions(files, {})
    np.testing.assert_allclose(aligned, np.tile(probabilities, (3, 1, 1)))
    assert stats(files, files, {}, 8)["accuracy"]["p_holm"] == 1

    cfg = {}
    for ds, expected in (("briancancer", 4174), ("crc", 7180)):
        with np.load(f"assets/{ds}_splits.npz") as manifest:
            canon = set(map_paths(manifest["test"], cfg))
        assert len(canon) == expected


def main():
    torch.set_num_threads(2)
    assert [len(stage_jobs(name)) for name in ("threshold", "main", "ablation", "robust", "cam", "external", "s4")] == [
        18, 6, 24, 6, 6, 3, 15,
    ]
    check_backbone()
    with TemporaryDirectory() as tmp:
        check_maps(Path(tmp))
        check_statistics(Path(tmp))
    print("SELF_CHECKS_PASS")


if __name__ == "__main__":
    main()
