import argparse
from typing import Any, Dict

import yaml

from experiments.evaluate import evaluate
from experiments.train import train


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


def _parse_fractions(raw: str):
    return [float(item) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--data-fraction", type=float, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--fpn-dim", type=int, default=None)
    parser.add_argument("--tap-fractions", default=None)
    parser.add_argument(
        "--use-topdown-fpn", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--shape-check", action=argparse.BooleanOptionalAction, default=None
    )
    args = parser.parse_args()

    base_cfg = _load_yaml(args.base_config)
    override_cfg = _load_yaml(args.config) if args.config else {}
    cfg = _deep_merge(base_cfg, override_cfg)

    if args.encoder:
        cfg.setdefault("encoder", {})["name"] = args.encoder
    if args.fpn_dim is not None:
        cfg.setdefault("encoder", {})["fpn_dim"] = args.fpn_dim
    if args.tap_fractions:
        cfg.setdefault("encoder", {})["tap_fractions"] = _parse_fractions(
            args.tap_fractions
        )
    if args.use_topdown_fpn is not None:
        cfg.setdefault("encoder", {})["use_topdown_fpn"] = args.use_topdown_fpn
    if args.data_fraction is not None:
        cfg.setdefault("data", {})["fraction"] = args.data_fraction
    if args.input_size is not None:
        cfg.setdefault("data", {})["input_size"] = args.input_size
    if args.shape_check is not None:
        cfg.setdefault("debug", {})["shape_check"] = args.shape_check

    if args.mode == "train":
        train(cfg)
    else:
        evaluate(cfg)


if __name__ == "__main__":
    main()
