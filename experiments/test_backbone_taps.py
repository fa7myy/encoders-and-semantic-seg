import argparse
import math
import os
import sys
import time
from typing import Any, Dict, Optional

import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from encoders.vit_pyramid import tokens_to_feature_map_strict
from experiments.train import build_backbone


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_device(raw: str) -> torch.device:
    if raw.lower() == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available.")
    return torch.device(raw)


def _load_image_tensor(
    path: str, target_size: Optional[int], device: torch.device
) -> torch.Tensor:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for --image.") from exc

    image = Image.open(path).convert("RGB")
    if target_size is not None:
        image = image.resize((target_size, target_size), Image.BICUBIC)

    image_bytes = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    tensor = image_bytes.view(image.size[1], image.size[0], 3).permute(2, 0, 1)
    tensor = tensor.float().div(255.0).unsqueeze(0).to(device)
    return tensor


def _make_heatmap_image(x: torch.Tensor):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for feature visualization.") from exc

    heat = x.abs().mean(1).squeeze(0)
    heat = heat - heat.min()
    denom = heat.max().clamp(min=1e-6)
    heat = (heat / denom * 255.0).byte().cpu().numpy()
    return Image.fromarray(heat, mode="L")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--check-allclose", action="store_true")
    parser.add_argument("--image", default=None)
    parser.add_argument("--dump-heatmaps", action="store_true")
    parser.add_argument("--show-heatmaps", action="store_true")
    parser.add_argument("--dump-dir", default="outputs/feature_vis")
    args = parser.parse_args()

    base_cfg = _load_yaml(args.base_config)
    override_cfg = _load_yaml(args.config) if args.config else {}
    cfg = _deep_merge(base_cfg, override_cfg)

    if args.input_size is not None:
        cfg.setdefault("data", {})["input_size"] = args.input_size

    enc_cfg = cfg.get("encoder", {})
    neck_type = enc_cfg.get("neck_type", "vit_pyramid")
    input_size = cfg.get("data", {}).get("input_size") or enc_cfg.get("img_size")

    patch_size = enc_cfg.get("patch_size")
    if patch_size is None:
        raise ValueError("encoder.patch_size must be set in config.")

    device = _parse_device(args.device)
    backbone = build_backbone(cfg).to(device)
    backbone.eval()

    if args.image:
        dummy = _load_image_tensor(args.image, input_size, device)
    else:
        if input_size is None:
            raise ValueError("input_size must be set in config or via --input-size.")
        dummy = torch.zeros(1, 3, input_size, input_size, device=device)

    input_h = int(dummy.shape[-2])
    input_w = int(dummy.shape[-1])
    if input_h % patch_size != 0 or input_w % patch_size != 0:
        raise ValueError(
            f"input size {input_h}x{input_w} must be divisible by patch_size {patch_size}."
        )

    if neck_type == "vitdet_sfp":
        with torch.no_grad():
            features = backbone(dummy)

        out_strides = enc_cfg.get("out_strides") or [4, 8, 16, 32]
        out_features = getattr(backbone, "_out_features", list(features.keys()))
        if len(out_features) != len(out_strides):
            raise ValueError(
                f"Expected {len(out_strides)} outputs, got {len(out_features)}."
            )

        for name, stride in zip(out_features, out_strides):
            feat = features[name]
            expected_h = int(math.ceil(input_h / stride))
            expected_w = int(math.ceil(input_w / stride))
            if feat.shape[-2:] != (expected_h, expected_w):
                raise ValueError(
                    f"{name} spatial shape mismatch: got {feat.shape[-2:]}, "
                    f"expected {(expected_h, expected_w)}."
                )

        print("ViTDet SimpleFeaturePyramid: tap checks skipped.")
        print("Pyramid shapes look consistent.")
        return

    tap_indices = list(backbone.tap_indices)
    hook_outputs: Dict[int, torch.Tensor] = {}
    handles = []
    for idx in tap_indices:
        def _make_hook(i: int):
            def _hook(_module, _inputs, output):
                hook_outputs[i] = output
            return _hook
        handles.append(
            backbone.model.blocks[idx].register_forward_hook(_make_hook(idx))
        )

    with torch.no_grad():
        tap_tokens, grid_hw, num_prefix = backbone._forward_taps(dummy)

    for handle in handles:
        handle.remove()

    print("Backbone tap check")
    print(f"- input_hw: {input_h}x{input_w}")
    print(f"- patch_size: {patch_size}")
    print(f"- grid_hw: {grid_hw}")
    print(f"- num_prefix_tokens: {num_prefix}")
    print(f"- tap_indices: {tap_indices}")

    if len(tap_tokens) != len(tap_indices):
        raise ValueError(
            f"Expected {len(tap_indices)} taps, got {len(tap_tokens)}."
        )

    for tap_idx, tokens in zip(tap_indices, tap_tokens):
        if tap_idx not in hook_outputs:
            raise ValueError(f"No hook output captured for tap index {tap_idx}.")
        hook_tokens = hook_outputs[tap_idx]
        if hook_tokens.shape != tokens.shape:
            raise ValueError(
                f"Shape mismatch at tap {tap_idx}: "
                f"{tuple(tokens.shape)} vs {tuple(hook_tokens.shape)}."
            )
        if args.check_allclose:
            if not torch.allclose(tokens, hook_tokens, rtol=1e-4, atol=1e-5):
                raise ValueError(f"Tensor mismatch at tap {tap_idx}.")

    with torch.no_grad():
        features = backbone(dummy)

    out_strides = enc_cfg.get("out_strides") or [4, 8, 16, 32]
    for name, stride in zip(backbone._out_features, out_strides):
        feat = features[name]
        expected_h = int(math.ceil(input_h / stride))
        expected_w = int(math.ceil(input_w / stride))
        if feat.shape[-2:] != (expected_h, expected_w):
            raise ValueError(
                f"{name} spatial shape mismatch: got {feat.shape[-2:]}, "
                f"expected {(expected_h, expected_w)}."
            )

    if args.dump_heatmaps or args.show_heatmaps:
        if args.dump_heatmaps:
            os.makedirs(args.dump_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")

        tap_maps = [
            tokens_to_feature_map_strict(t, grid_hw, num_prefix)
            for t in tap_tokens
        ]
        for idx, fmap in zip(tap_indices, tap_maps):
            label = f"tap_block{idx}"
            image = _make_heatmap_image(fmap)
            if args.dump_heatmaps:
                filename = f"{label}_{stamp}.png"
                image.save(os.path.join(args.dump_dir, filename))
            if args.show_heatmaps:
                image.show(title=label)

        for name in backbone._out_features:
            label = f"neck_{name}"
            image = _make_heatmap_image(features[name])
            if args.dump_heatmaps:
                filename = f"{label}_{stamp}.png"
                image.save(os.path.join(args.dump_dir, filename))
            if args.show_heatmaps:
                image.show(title=label)

        if args.dump_heatmaps:
            print(f"Saved heatmaps to {args.dump_dir}")

    print("Tap outputs and pyramid shapes look consistent.")


if __name__ == "__main__":
    main()
