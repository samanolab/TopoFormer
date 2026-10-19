import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        B_c, N_c, C_c = context.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B_c, N_c, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


class StageSpecificCTIHook(nn.Module):
    """
   Strategy 1: Layer-by-layer injection.
- Inject L0 Map at the beginning of Stage 1
- Inject L1 Map at the beginning of Stage 2
- Inject L2 Map at the beginning of Stage 3
- Inject L3 Map at the beginning of Stage 4

Modes:
- "per_level" (default): stage k consumes `topo_levels["Lk"]`. This is the
  canonical TopoFPN path: each stage gets a 400-channel smoothed map.
- "raw_concat": all four stages consume the *same* already-channel-
  concatenated 1600-channel map. Caller supplies it via `topo_levels["L0"]`
  (and may leave L1..L3 empty / None). Used by the simple-concat ablation
  where there is no FPN smoothing and no cross-scale mixing.
    """
    def __init__(
        self,
        topo_in_channels,
        token_dim,
        n_blocks,
        n_stages=4,
        mixer_heads=None,
        attn_dropout=0.1,
        proj_dropout=0.1,
        delta_norm_factor: float = 0.3,
        mode: str = "per_level",
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.token_dim = token_dim
        # Clamp delta magnitude relative to x (prevents topology delta from
        # dominating the pre-trained DINOv3 representation).
        self.delta_norm_factor = float(delta_norm_factor)

        if mode not in {"per_level", "raw_concat"}:
            raise ValueError(f"StageSpecificCTIHook mode must be 'per_level' or 'raw_concat', got {mode!r}")
        self.mode = mode

        # Split the multi-layer blocks of DINOv3 into n_stages evenly
        blocks_per_stage = n_blocks // n_stages
        self.stage_ends_all = [blocks_per_stage * (i + 1) - 1 for i in range(n_stages)]
        self.stage_ends_all[-1] = n_blocks - 1

        self.stage_starts = [0] + [e + 1 for e in self.stage_ends_all[:-1]]
        self.stage_to_level = {0: "L0", 1: "L1", 2: "L2", 3: "L3"}
        
        # Independent feature projections and cross attention corresponding to four stages
        self.topo_projs = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        
        heads = mixer_heads if mixer_heads is not None else max(1, token_dim // 64)
        
        for _ in range(n_stages):
           # Align the dimensions of Topo Map to those of DINO
            self.topo_projs.append(
                nn.Sequential(
                    nn.Conv2d(topo_in_channels, token_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(token_dim),
                    nn.GELU()
                )
            )
            self.norms1.append(nn.LayerNorm(token_dim))
            self.cross_attns.append(
                CustomCrossAttention(
                    dim=token_dim, 
                    num_heads=heads, 
                    qkv_bias=True, 
                    attn_drop=attn_dropout, 
                    proj_drop=proj_dropout
                )
            )
            self.norms2.append(nn.LayerNorm(token_dim))

    def forward(self, block_idx, x_list, topo_levels=None):
        if topo_levels is None:
            return x_list

        # Interactions are injected only at the beginning block of each stage
        if block_idx in self.stage_starts:
            stage_idx = self.stage_starts.index(block_idx)
            if self.mode == "raw_concat":
                # All four stages consume the SAME pre-concatenated 1600ch
                # map supplied under key "L0". L1..L3 are ignored. This is
                # the simple-concat ablation path: no FPN smoothing, no
                # cross-scale mixing.
                level_key = "L0"
            else:
                level_key = self.stage_to_level.get(stage_idx, None)

            topo_map = topo_levels.get(level_key, None)
            if topo_map is None:
                return x_list

            is_list = isinstance(x_list, list)
            x_input = x_list[-1] if is_list else x_list
            orig_dtype = x_input.dtype

            x = x_input.float()
            topo_map = topo_map.float()

            # Reduce the spatial dimension to the DINO feature layer
            B, C, H, W = topo_map.shape
            topo_feat = self.topo_projs[stage_idx](topo_map) # [B, token_dim, H, W]
            topo_tokens = topo_feat.flatten(2).transpose(1, 2) # [B, H*W, token_dim]

            # CLS protection: keep the cls token untouched and only modulate
            # patch tokens. This preserves the pre-trained DINOv3 cls
            # representation that the downstream classifier relies on.
            if x.dim() == 3:
                cls_token = x[:, :1, :]              # [B, 1, C]
                patch_tokens = x[:, 1:, :]          # [B, N-1, C]
            else:
                # Fallback for non-token inputs.
                cls_token, patch_tokens = None, x

            x_norm = self.norms1[stage_idx](patch_tokens)
            topo_norm = self.norms2[stage_idx](topo_tokens)

            delta = self.cross_attns[stage_idx](x_norm, topo_norm)

            # Magnitude clamp: scale down delta so that ||delta|| does not
            # exceed `delta_norm_factor` * ||patch||. Prevents topology delta
            # from dominating the pre-trained image representation.
            if self.delta_norm_factor is not None and self.delta_norm_factor > 0:
                delta_norm = delta.norm(dim=-1, keepdim=True) + 1e-6
                x_norm_mag = patch_tokens.norm(dim=-1, keepdim=True) + 1e-6
                max_delta = self.delta_norm_factor * x_norm_mag
                scale = (max_delta / delta_norm).clamp(max=1.0)
                delta = delta * scale

            patch_out = patch_tokens + delta

            if cls_token is not None:
                x_out = torch.cat([cls_token, patch_out], dim=1)
            else:
                x_out = patch_out

            x_out = x_out.to(orig_dtype)

            if is_list:
                x_list.append(x_out)
                return x_list
            return x_out

        return x_list

    def post_block(self, block_idx, x_list, topo_levels=None):
        return x_list


