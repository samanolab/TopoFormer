import os
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DINOV3_PKG_DIR = PROJECT_ROOT / "Backbone" / "dinov3" / "dinov3"

NCTCRC100K_DIR = str(globals().get("NCTCRC100K_DIR", "/home/imagea/DBs/NCT-CRC-HE-100K"))
CRCVAL7K_DIR = str(globals().get("CRCVAL7K_DIR", "/home/imagea/DBs/CRC-VAL-HE-7K"))

TOPO_NPZ_BASE = str(globals().get(
    "TOPO_NPZ_BASE",
    "/home/imagea/zhibo/TopoFormer/Github/checkpoint/PHC_180/phc_crc_topofpn_levels_train_test_maps",
)).strip()

DINO_LOCAL_CKPT = globals().get(
    "DINO_LOCAL_CKPT",
    "/home/imagea/DBs/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
)

ENCODER_NAME = str(globals().get("ENCODER_NAME", "dinov3_vit7b16"))

CONVNEXT_CKPT = str(globals().get("CONVNEXT_CKPT", "/home/imagea/DBs/convnextv2_base_1k_224_ema.pt"))
CONVNEXT_NAME = str(globals().get("CONVNEXT_NAME", "convnextv2_base"))

FD_FINETUNE_MODE = str(globals().get("FD_FINETUNE_MODE", "lora")).lower().strip()
FD_UNFREEZE_LAST_K = int(globals().get("FD_UNFREEZE_LAST_K", 0))
FD_LORA_R = int(globals().get("FD_LORA_R", 16))
FD_LORA_ALPHA = float(globals().get("FD_LORA_ALPHA", 16.0))
FD_LORA_DROPOUT = float(globals().get("FD_LORA_DROPOUT", 0.05))

USE_TOPO_INJECT = bool(globals().get("USE_TOPO_INJECT", True))
TOPO_ATTN_DROPOUT = float(globals().get("TOPO_ATTN_DROPOUT", 0.1))
TOPO_PROJ_DROPOUT = float(globals().get("TOPO_PROJ_DROPOUT", 0.1))
TOPO_NUM_HEADS = globals().get("TOPO_NUM_HEADS", None)
# Clamp topology delta magnitude to this fraction of patch-token norm.
TOPO_DELTA_NORM_FACTOR = float(globals().get("TOPO_DELTA_NORM_FACTOR", 0.3))

FD_EPOCHS = int(globals().get("FD_EPOCHS", 2))
FD_LR = float(globals().get("FD_LR", 5e-5))

USE_BALANCE_SAMPLER = bool(globals().get("USE_BALANCE_SAMPLER", True))
LABEL_SMOOTHING = float(globals().get("LABEL_SMOOTHING", 0.05))

DATASET_TAG = str(globals().get("DATASET_TAG", "crc")).strip()
CACHE_DIR = str(globals().get("CACHE_DIR", f"/home/imagea/zhibo/features_cache_{DATASET_TAG}_topoformer_fusion_RGB_180_seed42"))

CUDA_INDEX = 1
SMOKE_ONLY = bool(globals().get("SMOKE_ONLY", False))


def setup_local_paths():
    for p in [PROJECT_ROOT, DINOV3_PKG_DIR]:
        p = str(p)
        if p not in sys.path:
            sys.path.insert(0, p)


def setup_device():
    device = torch.device(
        f"cuda:{CUDA_INDEX}"
        if torch.cuda.is_available() and torch.cuda.device_count() > CUDA_INDEX
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    print("device:", device)

    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    return device


def main():
    setup_local_paths()

    from topoformer.model.disentanglement import train, evaluate, set_seed
    from topoformer.data.split_builders import build_nctcrc100k_trainval_crcval7k_test
    from topoformer.data.topo_maps import TopoMapStore
    from topoformer.data.datasets import make_loaders, make_cache_loaders
    from topoformer.model.model_builder import build_topoformer_model
    from topoformer.engine.runtime import (
        setup_logger,
        build_train_args,
        save_npz,
        print_report_and_confusion_matrix,
    )

    device = setup_device()
    set_seed(42)

    split = build_nctcrc100k_trainval_crcval7k_test(
        trainval_dir=NCTCRC100K_DIR,
        test_dir=CRCVAL7K_DIR,
    )

    fd_paths = list(split.paths)
    fd_y = np.asarray(split.y_all, dtype=np.int64)
    fd_label_names = list(split.label_names)
    fd_domain_to_id = dict(split.domain_to_id)
    fd_id_to_domain = dict(split.id_to_domain)
    fd_train_idx = np.asarray(split.train_idx, dtype=np.int64)
    fd_test_idx = np.asarray(split.test_idx, dtype=np.int64)

    # Manually carve out a 10% stratified val from the trainval pool.
    # We do this locally (without modifying split_builders.py) so that
    # the held-out 10% has the same y-stratified distribution as in the
    # Briancancer script.
    from sklearn.model_selection import train_test_split

    trainval_y = fd_y[fd_train_idx]
    fd_train_idx, fd_val_idx = train_test_split(
        fd_train_idx,
        test_size=0.1,
        stratify=trainval_y,
        random_state=42,
    )
    fd_train_idx = np.asarray(fd_train_idx, dtype=np.int64)
    fd_val_idx = np.asarray(fd_val_idx, dtype=np.int64)

    print("[info] DINO-FD CRC splits:")
    print("  train :", NCTCRC100K_DIR, "N=", int(fd_train_idx.size))
    print("  val   :", "stratified-from-trainval", "N=", int(fd_val_idx.size))
    print("  test  :", CRCVAL7K_DIR, "N=", int(fd_test_idx.size))
    print("  classes:", fd_id_to_domain)

    topo_store = TopoMapStore(TOPO_NPZ_BASE)
    topo_store.check_alignment(fd_paths)

    fd_bs_init = 8 if str(device).startswith("cuda") else 16
    fd_bs = int(globals().get("FD_BATCH", fd_bs_init))

    while True:
        try:
            loader_ret = make_loaders(
                fd_paths,
                fd_y,
                fd_train_idx,
                fd_test_idx,
                topo_store,
                device,
                fd_bs,
                val_idx=fd_val_idx,
                balance_sampler=USE_BALANCE_SAMPLER,
            )

            if fd_val_idx is not None:
                train_loader, val_loader, test_loader = loader_ret
            else:
                train_loader, test_loader = loader_ret
                val_loader = None

            xb, yb, tb = next(iter(train_loader))
            xb = xb.to(device, non_blocking=True)
            _ = xb.sum().item()
            del xb, yb, tb

            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

            break

        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or fd_bs <= 1:
                raise
            print(f"[warn] OOM with batch={fd_bs}. retry with batch={fd_bs // 2} ...")
            fd_bs = max(1, fd_bs // 2)
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

    train_cache_loader, test_cache_loader = make_cache_loaders(
        fd_paths,
        fd_y,
        fd_train_idx,
        fd_test_idx,
        topo_store,
        device,
        fd_bs,
    )

    print("[ok] FD loaders:", "train=", len(train_loader), "val=", len(val_loader) if val_loader is not None else 0, "test=", len(test_loader), "bs=", fd_bs, "balance_sampler=", USE_BALANCE_SAMPLER)

    print("[info] finetune", {
        "mode": FD_FINETUNE_MODE,
        "unfreeze_last_k": FD_UNFREEZE_LAST_K,
        "lora": (FD_LORA_R, FD_LORA_ALPHA, FD_LORA_DROPOUT),
        "topo_inject": USE_TOPO_INJECT,
        "topo_attn": "mrfp_cti_maps",
    })

    model = build_topoformer_model(
        device=device,
        fd_label_names=fd_label_names,
        dino_local_ckpt=DINO_LOCAL_CKPT,
        encoder_name=ENCODER_NAME,
        topo_c=topo_store.topo_c,
        fd_finetune_mode=FD_FINETUNE_MODE,
        fd_unfreeze_last_k=FD_UNFREEZE_LAST_K,
        fd_lora_r=FD_LORA_R,
        fd_lora_alpha=FD_LORA_ALPHA,
        fd_lora_dropout=FD_LORA_DROPOUT,
        use_topo_inject=USE_TOPO_INJECT,
        topo_attn_dropout=TOPO_ATTN_DROPOUT,
        topo_proj_dropout=TOPO_PROJ_DROPOUT,
        topo_num_heads=TOPO_NUM_HEADS,
        topo_delta_norm_factor=TOPO_DELTA_NORM_FACTOR,
        convnext_ckpt=CONVNEXT_CKPT,
        convnext_name=CONVNEXT_NAME,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable params: {trainable / 1e6:.2f}M / total {total / 1e9:.2f}B")

    has_fp16_trainable = any((p.requires_grad and p.dtype == torch.float16) for p in model.parameters())
    if has_fp16_trainable and str(device).startswith("cuda"):
        print("[warn] trainable FP16 params detected -> disabling AMP/GradScaler to avoid GradScaler errors.")
        print("       Tip: set FD_UNFREEZE_LAST_K=0 or keep trainable blocks in fp32.")

    logger = setup_logger("dinofd_nb")

    args = build_train_args(
        device=device,
        has_fp16_trainable=has_fp16_trainable,
        fd_epochs=FD_EPOCHS,
        fd_lr=FD_LR,
        fd_finetune_mode=FD_FINETUNE_MODE,
        fd_unfreeze_last_k=FD_UNFREEZE_LAST_K,
        fd_lora_r=FD_LORA_R,
        fd_lora_alpha=FD_LORA_ALPHA,
        fd_lora_dropout=FD_LORA_DROPOUT,
        label_smoothing=LABEL_SMOOTHING,
    )

    if SMOKE_ONLY:
        print("[smoke] running 1 batch forward only")
        model.eval()
        with torch.no_grad():
            batch = next(iter(train_loader))
            if isinstance(batch, (tuple, list)) and len(batch) == 4:
                images, targets, topo_levels, _paths = batch
            elif isinstance(batch, (tuple, list)) and len(batch) == 3:
                images, targets, topo_levels = batch
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")

            images = images.to(device, non_blocking=True)
            topo_levels = {k: v.to(device, non_blocking=True) for k, v in topo_levels.items()}
            out = model(images, grl_lambda=0.0, topo_levels=topo_levels)

        print("[smoke] feat=", tuple(out["feat"].shape), "logitsA=", tuple(out["logitsA"].shape), "fusion_gate=", tuple(out["fusion_gate"].shape))
        fg = out["fusion_gate"].detach().float().cpu()
        print("[smoke] fusion_gate stats: min=", float(fg.min()), "max=", float(fg.max()), "mean=", float(fg.mean()))
        return

    # train() uses Balanced Accuracy on the held-out 10% stratified val
    # to select the best epoch (train() returns best_state). We then
    # explicitly load best_val weights before evaluating on test, so that
    # the reported test metrics come from the val-selected checkpoint.
    history, best_state = train(model, train_loader, val_loader, device, args, logger)
    print("[ok] training finished. epochs_ran=", len(history))

    if best_state is not None and val_loader is not None:
        model.load_state_dict(best_state, strict=False)
        print("[ok] loaded best-val-epoch weights for downstream evaluation")
        val_eval = evaluate(model, val_loader, device, lambda_grl=0.0)
        print("[val@best]", val_eval)

    last_epoch = int(args.epochs)
    test_metrics = evaluate(model, test_loader, device, lambda_grl=0.0)
    print("[test]", test_metrics)

    model_for_cache = model
    os.makedirs(CACHE_DIR, exist_ok=True)

    train_npz = save_npz(
        "train",
        train_cache_loader,
        model_for_cache,
        device,
        CACHE_DIR,
        DATASET_TAG,
        fd_id_to_domain,
        fd_label_names,
        ENCODER_NAME,
        DINO_LOCAL_CKPT,
        fd_bs,
        last_epoch,
        TOPO_NPZ_BASE,
        topo_store.topo_c,
        topo_store.topo_hw,
    )

    test_npz = save_npz(
        "test",
        test_cache_loader,
        model_for_cache,
        device,
        CACHE_DIR,
        DATASET_TAG,
        fd_id_to_domain,
        fd_label_names,
        ENCODER_NAME,
        DINO_LOCAL_CKPT,
        fd_bs,
        last_epoch,
        TOPO_NPZ_BASE,
        topo_store.topo_c,
        topo_store.topo_hw,
    )

    print_report_and_confusion_matrix(
        model_for_cache,
        test_cache_loader,
        device,
        fd_label_names,
        fd_id_to_domain,
        DATASET_TAG,
        ENCODER_NAME,
        test_metrics,
    )

    print("\n[done] cache files:")
    print("  ", train_npz)
    print("  ", test_npz)


if __name__ == "__main__":
    main()