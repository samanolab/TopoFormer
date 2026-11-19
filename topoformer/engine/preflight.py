from contextlib import ExitStack

import torch


@torch.no_grad()
def preflight(model, loader, device, amp=False):
    device = torch.device(device)
    encoder = model.vit.encoder
    hook = encoder._topo_hook
    starts = tuple(hook.stage_starts)
    assert starts == tuple(10 * level for level in hook.levels)
    assert hook.variant == "local"
    prefix = encoder.n_storage_tokens + 1
    assert prefix == 5
    modes = [(module, module.training) for module in model.modules()]
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    produced = {}
    consumed = set()

    def record(_, args, result):
        block, tokens, _ = args
        if block not in starts:
            return
        assert len(tokens) == len(result), f"stage {block}: changed token list length"
        before, after = tokens[-1], result[-1]
        assert before.shape == after.shape and after.shape[1] == prefix + 196
        assert torch.isfinite(after).all(), f"stage {block}: nonfinite tokens"
        assert torch.equal(before[:, 1:prefix], after[:, 1:prefix]), f"stage {block}: register tokens changed"
        cls_changed = not torch.equal(before[:, :1], after[:, :1])
        assert cls_changed == (block == 30), f"stage {block}: unexpected CLS behavior"
        assert not torch.equal(before[:, prefix:], after[:, prefix:]), f"stage {block}: no patch update"
        produced[block] = after

    def verify_input(block):
        def check(_, args):
            tokens, rope = args
            assert len(tokens) == len(rope), f"stage {block}: token/RoPE length mismatch"
            assert tokens[-1] is produced.get(block), f"stage {block}: block discarded updated tokens"
            consumed.add(block)
        return check

    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.autocast(device_type=device.type, enabled=amp):
            # Collate real examples without advancing the training sampler.
            size = len(loader.dataset)
            batch = loader.collate_fn([loader.dataset[i] for i in range(0, size, max(1, (size + 7) // 8))])
            x, _, topo = batch[:3]
            x = x.to(device)
            topo = {key: value.to(device) for key, value in topo.items()}
            model.eval()
            with ExitStack() as handles:
                handles.callback(hook.register_forward_hook(record).remove)
                for block in starts:
                    handles.callback(encoder.blocks[block].register_forward_pre_hook(verify_input(block)).remove)
                base = model(x, topo_levels=topo)["f_disc"].float()
            assert tuple(produced) == starts and consumed == set(starts)
            assert torch.isfinite(base).all(), "nonfinite preflight f_disc"
            responses = {}
            for i in hook.levels:
                key = f"L{i}"
                value = topo[key]
                native = hook.stages[i].native(value)
                repeated = torch.nn.functional.interpolate(native, size=value.shape[-2:], mode="nearest")
                assert torch.equal(value, repeated), f"{key}: cache does not repeat the native grid"
                swapped = value.roll(1, dims=0)
                assert not torch.equal(value, swapped), f"{key}: preflight samples have identical maps"
                changed = model(x, topo_levels={**topo, key: swapped})["f_disc"].float()
                assert torch.isfinite(changed).all(), f"{key}: nonfinite f_disc after map swap"
                responses[key] = (changed - base).abs().max().item()
                assert responses[key] > 0, f"{key}: swapping PH did not change f_disc"
    finally:
        for module, training in modes:
            module.training = training

    print(f"CTI_PREFLIGHT_PASS variant={hook.variant} starts={starts} f_disc_swap_max={responses}", flush=True)
