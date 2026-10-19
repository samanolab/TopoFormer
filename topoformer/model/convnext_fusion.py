import os
import re

import torch

from topoformer.model.disentanglement import grad_reverse


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _select_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_ema", "network", "net", "module"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj

    raise TypeError(f"Unexpected checkpoint object type: {type(obj)}")


def _strip_prefix(sd: dict, prefix: str = "module.") -> dict:
    out = {}
    for k, v in sd.items():
        if isinstance(k, str) and k.startswith(prefix):
            k = k[len(prefix):]
        out[k] = v
    return out


def _looks_like_official_keys(sd: dict) -> bool:
    for k in sd.keys():
        if not isinstance(k, str):
            continue
        if k.startswith("downsample_layers."):
            return True
        if k.startswith("stages.") and ".dwconv." in k:
            return True
    return False


def _looks_like_timm_keys(sd: dict) -> bool:
    for k in sd.keys():
        if not isinstance(k, str):
            continue
        if k.startswith("stem."):
            return True
    return False


def _to_1d_param(x):
    if not isinstance(x, torch.Tensor):
        return x
    if x.ndim > 1:
        x = x.squeeze()
    if x.ndim != 1:
        x = x.reshape(-1)
    return x


def convert_official_convnextv2_to_timm(sd: dict) -> dict:
    """Convert official ConvNeXtV2-style keys to timm ConvNeXtV2 keys."""
    out = {}

    for k, v in sd.items():
        if not isinstance(k, str):
            continue

        m = re.match(r"^downsample_layers\.(\d+)\.(\d+)\.(weight|bias)$", k)
        if not m:
            continue

        i = int(m.group(1))
        j = int(m.group(2))
        param = m.group(3)

        if i == 0:
            out[f"stem.{j}.{param}"] = v
        else:
            if j == 0:
                out[f"stages.{i}.downsample.0.{param}"] = v
            elif j == 1:
                out[f"stages.{i}.downsample.1.{param}"] = v
            else:
                if getattr(v, "ndim", None) == 1:
                    out[f"stages.{i}.downsample.0.{param}"] = v
                else:
                    out[f"stages.{i}.downsample.1.{param}"] = v

    for k, v in sd.items():
        if not isinstance(k, str):
            continue

        m = re.match(r"^stages\.(\d+)\.(\d+)\.(.+)$", k)
        if not m:
            continue

        s = int(m.group(1))
        b = int(m.group(2))
        rest = m.group(3)
        prefix = f"stages.{s}.blocks.{b}."

        if rest.startswith("dwconv."):
            out[prefix + "conv_dw." + rest.split(".", 1)[1]] = v
            continue
        if rest.startswith("norm."):
            out[prefix + "norm." + rest.split(".", 1)[1]] = v
            continue
        if rest.startswith("pwconv1."):
            out[prefix + "mlp.fc1." + rest.split(".", 1)[1]] = v
            continue
        if rest.startswith("pwconv2."):
            out[prefix + "mlp.fc2." + rest.split(".", 1)[1]] = v
            continue

        if rest in ("grn.gamma", "grn.weight"):
            out[prefix + "mlp.grn.weight"] = _to_1d_param(v)
            continue
        if rest in ("grn.beta", "grn.bias"):
            out[prefix + "mlp.grn.bias"] = _to_1d_param(v)
            continue

    if "norm.weight" in sd:
        out["head.norm.weight"] = sd["norm.weight"]
    if "norm.bias" in sd:
        out["head.norm.bias"] = sd["norm.bias"]

    if "head.weight" in sd:
        out["head.fc.weight"] = sd["head.weight"]
    if "head.bias" in sd:
        out["head.fc.bias"] = sd["head.bias"]

    return out


def _filter_shapes(model: torch.nn.Module, sd: dict):
    model_sd = model.state_dict()
    filtered = {}
    skipped = 0
    skipped_keys = []
    for k, v in sd.items():
        if k in model_sd and isinstance(v, torch.Tensor) and isinstance(model_sd[k], torch.Tensor):
            if tuple(v.shape) != tuple(model_sd[k].shape):
                skipped += 1
                skipped_keys.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
                continue
        filtered[k] = v
    return filtered, skipped, skipped_keys


def build_convnextv2(convnext_ckpt: str, convnext_name: str, device):
    import timm  # type: ignore

    assert os.path.exists(convnext_ckpt), f"ConvNeXtV2 ckpt not found: {convnext_ckpt}"

    names_to_try = []
    if convnext_name:
        names_to_try.append(convnext_name)

    names_to_try.extend([
        "convnextv2_base",
        "convnextv2_base.fcmae",
        "convnextv2_base.fb_in1k",
        "convnextv2_base_in1k",
    ])

    last_err = None
    cnx = None

    for nm in names_to_try:
        try:
            cnx = timm.create_model(nm, pretrained=False, num_classes=0, global_pool="avg")
            print("[cnx] timm model =", nm, "| num_features=", getattr(cnx, "num_features", "?"))
            break
        except Exception as e:
            last_err = e
            cnx = None

    if cnx is None:
        raise RuntimeError(f"Failed to create ConvNeXtV2 via timm. Tried={names_to_try}. Last error={last_err!r}")

    raw = _safe_torch_load(convnext_ckpt)
    sd = _strip_prefix(_select_state_dict(raw), prefix="module.")

    looks_official = _looks_like_official_keys(sd)
    looks_timm = _looks_like_timm_keys(sd)
    print(f"[cnx] ckpt key style: official={looks_official} timm={looks_timm}")

    if looks_official and not looks_timm:
        sd = convert_official_convnextv2_to_timm(sd)
        print("[cnx] applied official->timm key remap")
    elif not looks_official and not looks_timm:
        print("[cnx][WARN] ckpt keys do not match official ConvNeXtV2 nor timm naming; "
              "weights will likely load as random init")

    model_keys = set(cnx.state_dict().keys())
    ckpt_keys = set(k for k in sd.keys() if isinstance(k, str))
    matched = len(model_keys & ckpt_keys)
    print(f"[cnx] model_keys={len(model_keys)} ckpt_keys={len(ckpt_keys)} matched={matched}")

    filtered, skipped, skipped_keys = _filter_shapes(cnx, sd)
    if skipped:
        print(f"[cnx] skipped {skipped} keys due to shape mismatch")
        for k, src, dst in skipped_keys[:5]:
            print(f"[cnx]   skip {k}: ckpt={src} model={dst}")
        if len(skipped_keys) > 5:
            print(f"[cnx]   ... and {len(skipped_keys) - 5} more")

    missing, unexpected = cnx.load_state_dict(filtered, strict=False)
    print(f"[cnx] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 50 or len(unexpected) > 50:
        print(f"[cnx][WARN] large mismatch: pretrained weights likely NOT loaded correctly. "
              f"First 10 missing: {missing[:10]}")
        print(f"[cnx][WARN] First 10 unexpected: {unexpected[:10]}")

    for p in cnx.parameters():
        p.requires_grad = False

    cnx.eval()

    if str(device).startswith("cuda"):
        cnx.to(dtype=torch.float16)

    return cnx


class GatedFusionFD(torch.nn.Module):
    def __init__(self, vit_model: torch.nn.Module, cnx_model: torch.nn.Module):
        super().__init__()

        self.vit = vit_model
        self.cnx = cnx_model

        self.out_dim = int(getattr(vit_model, "out_dim", 1))
        self.feature_dim = int(getattr(vit_model, "feature_dim", 256))

        cnx_dim = getattr(cnx_model, "num_features", None)
        if cnx_dim is None:
            raise RuntimeError("ConvNeXtV2 model has no num_features; cannot infer feature dim.")

        self.cnx_dim = int(cnx_dim)

        self.proj_vit = torch.nn.Identity()
        self.proj_cnx = torch.nn.Linear(self.cnx_dim, self.feature_dim)

        self.gate_ln = torch.nn.LayerNorm(self.feature_dim + self.cnx_dim)
        self.gate_fc = torch.nn.Linear(self.feature_dim + self.cnx_dim, self.feature_dim)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.vit, name)

    def _encode_fused(self, x: torch.Tensor, topo_levels=None):
        feat_vit = self.vit._encode(x, topo_levels=topo_levels).float()

        feat_cnx = self.cnx(x)
        if isinstance(feat_cnx, (tuple, list)):
            feat_cnx = feat_cnx[0]
        if feat_cnx.dim() > 2:
            feat_cnx = feat_cnx.mean(dim=tuple(range(2, feat_cnx.dim())))
        feat_cnx = feat_cnx.float()

        cat = torch.cat([feat_vit, feat_cnx], dim=1)
        g = torch.sigmoid(self.gate_fc(self.gate_ln(cat)))
        fused = g * self.proj_vit(feat_vit) + (1.0 - g) * self.proj_cnx(feat_cnx)

        return fused, g

    def forward(self, x: torch.Tensor, grl_lambda: float = 1.0, use_adapter: bool = True, topo_levels=None):
        fused, g = self._encode_fused(x, topo_levels=topo_levels)

        if use_adapter:
            f_disc = self.vit.adapter_disc(fused)
            f_indisc = self.vit.adapter_indisc(fused)
        else:
            f_disc = self.vit.proj_disc_simple(fused)
            f_indisc = self.vit.proj_indisc_simple(fused)

        vit_gate = None
        if getattr(self.vit, "use_gating", False):
            vit_gate = torch.sigmoid(self.vit.gate_layer(fused))
            f_disc = f_disc * vit_gate
            f_indisc = f_indisc * (1.0 - vit_gate)

        logitsA = self.vit.decoderA(f_disc)
        logitsB = self.vit.decoderB(grad_reverse(f_indisc, grl_lambda))

        return {
            "logitsA": logitsA,
            "logitsB": logitsB,
            "f_disc": f_disc,
            "f_indisc": f_indisc,
            "feat": fused,
            "gate": vit_gate,
            "fusion_gate": g,
        }

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
