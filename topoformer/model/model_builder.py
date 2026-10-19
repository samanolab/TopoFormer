import types

import torch

from .disentanglement import OrthogonalDisentangler
from .topo_cti_v2 import StageSpecificCTIHook
from .lora import get_vit_blocks, inject_lora_vit
from .convnext_fusion import build_convnextv2, GatedFusionFD


def build_topoformer_model(
    device,
    fd_label_names,
    dino_local_ckpt: str,
    encoder_name: str,
    topo_c: int,
    fd_finetune_mode: str,
    fd_unfreeze_last_k: int,
    fd_lora_r: int,
    fd_lora_alpha: float,
    fd_lora_dropout: float,
    use_topo_inject: bool,
    topo_attn_dropout: float,
    topo_proj_dropout: float,
    topo_num_heads,
    convnext_ckpt: str,
    convnext_name: str,
    topo_delta_norm_factor: float = 0.3,
    topo_inject_mode: str = "per_level",
):
    # topo_inject_mode:
    #   "per_level"  -> canonical TopoFPN path; topo_c=400; each stage gets its own Lk map.
    #   "raw_concat" -> simple-concat ablation; topo_c=1600; all stages see the same
    #                   already-channel-concatenated map. The caller is expected to
    #                   provide that single 1600ch map under the "L0" key (or any key
    #                   actually; the hook forces L0 in raw_concat mode).
    if topo_inject_mode not in {"per_level", "raw_concat"}:
        raise ValueError(
            f"topo_inject_mode must be 'per_level' or 'raw_concat', got {topo_inject_mode!r}"
        )
    vit = OrthogonalDisentangler(
        encoder_source="dinov3",
        encoder_name=encoder_name,
        pretrained_path=dino_local_ckpt,
        out_dim=len(fd_label_names),
        frozen_blocks=999,
        c_mode="cls",
        head_style="mlp",
        debug_shapes=False,
    )

    for p in vit.encoder.parameters():
        p.requires_grad = False

    if str(device).startswith("cuda"):
        vit.encoder.to(dtype=torch.float16)

    if use_topo_inject:
        blocks = get_vit_blocks(vit.encoder)
        if blocks is None:
            raise RuntimeError("Cannot locate ViT blocks for topo injection.")

        n_blocks = int(len(blocks))

        token_dim = getattr(vit.encoder, "embed_dim", None) or getattr(vit.encoder, "num_features", None)
        if token_dim is None:
            token_dim = int(blocks[-1].norm1.normalized_shape[0]) if hasattr(blocks[-1], "norm1") else None
        if token_dim is None:
            raise RuntimeError("Cannot infer token_dim(embed_dim) from dinov3 encoder")

        topo_n_stages = int(globals().get("TOPO_N_STAGES", 4))

        vit.topo_injector = StageSpecificCTIHook(
            topo_in_channels=topo_c,
            token_dim=int(token_dim),
            n_blocks=n_blocks,
            n_stages=topo_n_stages,
            mixer_heads=topo_num_heads,
            attn_dropout=topo_attn_dropout,
            proj_dropout=topo_proj_dropout,
            delta_norm_factor=topo_delta_norm_factor,
            mode=topo_inject_mode,
        )

        vit.encoder._topo_hook = vit.topo_injector

        for p in vit.topo_injector.parameters():
            p.requires_grad = True

        print("[topo] injector enabled | StageSpecific | n_blocks=", n_blocks, "token_dim=", int(token_dim))

    if fd_finetune_mode == "lora":
        n_rep = inject_lora_vit(
            vit.encoder,
            r=fd_lora_r,
            alpha=fd_lora_alpha,
            dropout=fd_lora_dropout,
        )

        for n, p in vit.encoder.named_parameters():
            if "lora_" in n.lower():
                p.requires_grad = True

        print(f"[lora] injected into encoder: replaced_linears={n_rep}")

    if fd_finetune_mode in {"unfreeze", "unfreeze_last_k", "lora"} and fd_unfreeze_last_k > 0:
        blocks = get_vit_blocks(vit.encoder)
        if blocks is None:
            raise RuntimeError("Cannot locate blocks for unfreeze_last_k.")

        for blk in blocks[-int(fd_unfreeze_last_k):]:
            for p in blk.parameters():
                p.requires_grad = True

        print(f"[unfreeze] last_k={fd_unfreeze_last_k} blocks set trainable")

    vit.adapter_disc.float()
    vit.adapter_indisc.float()
    vit.decoderA.float()
    vit.decoderB.float()
    vit.decoderA_fused.float()
    vit.proj_disc_simple.float()
    vit.proj_indisc_simple.float()
    vit.pool_attn.float()

    if hasattr(vit, "topo_injector"):
        vit.topo_injector.float()

    _orig_encode = vit._encode

    def _encode_fp32(self, x, topo_levels=None):
        feat = _orig_encode(x, topo_levels=topo_levels).float()
        return feat

    vit._encode = types.MethodType(_encode_fp32, vit)

    cnx = build_convnextv2(convnext_ckpt, convnext_name, device)

    model = GatedFusionFD(vit, cnx)
    model.proj_cnx.float()
    model.gate_ln.float()
    model.gate_fc.float()

    model = model.to(device)
    # Surface any async CUDA illegal-access / ECC error that was queued
    # by an earlier op (e.g. previous `model.to()` from preflight), so
    # the user gets a clean stack instead of a misleading one.
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
        except RuntimeError:
            # Re-raise with a clearer message.
            raise RuntimeError(
                f"CUDA error during `model.to({device})` — this is almost "
                f"always a *previous* async illegal memory access (ECC / "
                f"OOB) that bubbled up here. Check earlier GPU work and "
                f"consider `CUDA_LAUNCH_BLOCKING=1`."
            ) from None
    return model