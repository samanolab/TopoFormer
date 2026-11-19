import hashlib
from pathlib import Path

import torch


COMPACT_FORMAT = "topoformer-frozen-dino-v1"


def dino_sha256(config):
    expected = config.get("dinov3_sha256")
    if config.get("dinov3_sha256_file"):
        receipt = Path(config["dinov3_sha256_file"]).read_text(encoding="ascii").split()[0]
        if expected and receipt.lower() != expected.lower():
            raise ValueError("DINOv3 fingerprint and checksum receipt disagree")
        expected = receipt
    with Path(config["dinov3"]).open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if expected and digest != expected.lower():
        raise ValueError("DINOv3 checkpoint does not match its configured checksum")
    return digest


def omitted_encoder_keys(model):
    return {
        name for name, param in model.named_parameters()
        if name.startswith("vit.encoder.") and not param.requires_grad
    }


def compact_state(model, fingerprint):
    omitted = omitted_encoder_keys(model)
    if not omitted:
        raise ValueError("No frozen DINOv3 parameters to omit")
    weights = {
        key: tensor.detach().to(device="cpu", copy=True)
        for key, tensor in model.state_dict().items() if key not in omitted
    }
    return {
        "format": COMPACT_FORMAT,
        "dinov3_sha256": fingerprint,
        "omitted": sorted(omitted),
        "weights": weights,
    }


def load_selected_weights(model, path, config=None):
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if payload.get("format") != COMPACT_FORMAT:
        model.load_state_dict(payload, strict=True)
        return

    if config is None or payload["dinov3_sha256"] != dino_sha256(config):
        raise ValueError("Compact checkpoint needs the original DINOv3 pretrained weights")
    omitted = omitted_encoder_keys(model)
    weights = payload["weights"]
    if set(payload["omitted"]) != omitted or set(weights) != set(model.state_dict()) - omitted:
        raise ValueError("Compact checkpoint tensors do not match the requested variant")
    model.load_state_dict(weights, strict=False)
