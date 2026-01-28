from typing import Any, Dict

import torch

from encoders.clip import build_clip_backbone
from encoders.dinov2 import build_dinov2_backbone
from encoders.mae import build_mae_backbone


def _get_encoder_builder(name: str):
    builders = {
        "clip": build_clip_backbone,
        "dinov2": build_dinov2_backbone,
        "mae": build_mae_backbone,
    }
    if name not in builders:
        raise ValueError(f"Unknown encoder name: {name}")
    return builders[name]


def build_backbone(cfg: Dict[str, Any]):
    enc_cfg = cfg.get("encoder", {})
    if enc_cfg.get("patch_size") is None:
        raise ValueError("encoder.patch_size must be set in the config.")
    if enc_cfg.get("embed_dim") is None:
        raise ValueError("encoder.embed_dim must be set in the config.")
    builder = _get_encoder_builder(enc_cfg.get("name", "clip"))

    kwargs: Dict[str, Any] = {
        "pretrained": enc_cfg.get("pretrained", True),
        "fpn_dim": enc_cfg.get("fpn_dim", 256),
        "tap_fractions": enc_cfg.get("tap_fractions", [0.25, 0.5, 0.75, 1.0]),
        "out_strides": enc_cfg.get("out_strides"),
        "use_topdown_fpn": enc_cfg.get("use_topdown_fpn", False),
        "use_smoothing": enc_cfg.get("use_smoothing", True),
        "img_size": enc_cfg.get("img_size"),
        "patch_size": enc_cfg.get("patch_size"),
        "embed_dim": enc_cfg.get("embed_dim"),
    }
    if enc_cfg.get("model_name"):
        kwargs["model_name"] = enc_cfg["model_name"]

    return builder(**kwargs)


def train(cfg: Dict[str, Any]):
    backbone = build_backbone(cfg)
    debug_cfg = cfg.get("debug", {})
    if debug_cfg.get("shape_check", True):
        enc_cfg = cfg.get("encoder", {})
        input_size = cfg.get("data", {}).get("input_size") or enc_cfg.get(
            "img_size", 512
        )
        backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            features = backbone(dummy)
        for name, feat in features.items():
            print(f"{name}: {tuple(feat.shape)}")
        return backbone

    raise NotImplementedError(
        "Mask2Former training is not wired here yet. Call your external trainer "
        "after build_backbone(cfg)."
    )
