import argparse
import os
import sys
from typing import Any, Dict, List

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.evaluate import evaluate, run_mask2former
from experiments.train import train
from utils.data import (
    validate_encoder_and_data_config,
    validate_existing_file,
    validate_mask2former_config,
)


class _CompatBooleanOptionalAction(argparse.Action):
    """
    Python 3.8-compatible replacement for argparse.BooleanOptionalAction.
    """

    def __init__(self, option_strings, dest, **kwargs):
        option_strings = list(option_strings)
        expanded = []
        for option_string in option_strings:
            expanded.append(option_string)
            if option_string.startswith("--"):
                expanded.append("--no-" + option_string[2:])
        kwargs.pop("nargs", None)
        super().__init__(option_strings=expanded, dest=dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string and option_string.startswith("--no-"):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)


BooleanOptionalAction = getattr(
    argparse,
    "BooleanOptionalAction",
    _CompatBooleanOptionalAction,
)

DEFAULT_ENCODER_CONFIG = "configs/encoder_clip.yaml"
DEFAULT_MASK2FORMER_CONFIG = "configs/mask2former_voc.yaml"


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


def _as_str_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    return [str(raw)]


def _apply_mask2former_overrides(cfg: Dict[str, Any], args) -> None:
    mask_cfg = cfg.setdefault("mask2former", {})
    if args.mask2former_config_file:
        mask_cfg["config_file"] = args.mask2former_config_file
    if args.config:
        mask_cfg.setdefault("encoder_config", args.config)
    else:
        mask_cfg.setdefault("encoder_config", args.base_config)
    if args.base_config:
        mask_cfg.setdefault("base_encoder_config", args.base_config)
    if args.weights:
        mask_cfg["weights"] = args.weights
    if args.output_dir:
        mask_cfg["output_dir"] = args.output_dir
    if args.mask2former_opts is not None:
        mask_cfg["opts"] = _as_str_list(args.mask2former_opts)
    if args.max_epochs is not None:
        mask_cfg["max_epochs"] = int(args.max_epochs)
    if args.freeze_backbone is not None:
        mask_cfg["freeze_backbone"] = bool(args.freeze_backbone)
    if args.resume is not None:
        mask_cfg["resume"] = bool(args.resume)
    if args.skip_flops is not None:
        mask_cfg["skip_flops"] = bool(args.skip_flops)
    if args.skip_inference_benchmark is not None:
        mask_cfg["skip_inference_benchmark"] = bool(args.skip_inference_benchmark)
    if args.benchmark_max_batches is not None:
        mask_cfg["benchmark_max_batches"] = int(args.benchmark_max_batches)
    if args.eval_output_file:
        mask_cfg["eval_output_file"] = args.eval_output_file


def _validate_run_config(cfg: Dict[str, Any], mode: str) -> None:
    validate_encoder_and_data_config(cfg.get("encoder", {}), cfg.get("data", {}))

    mask_cfg = cfg.get("mask2former", {})
    if mode == "eval" and not mask_cfg.get("config_file"):
        raise ValueError(
            "--mode eval requires --mask2former-config-file so segmentation metrics can be computed."
        )

    validate_mask2former_config(
        mask_cfg,
        mode=mode,
        config_required=bool(mask_cfg.get("config_file")),
    )

    if mode == "train" and not mask_cfg.get("config_file"):
        shape_check = cfg.get("debug", {}).get("shape_check", True)
        if not shape_check:
            raise ValueError(
                "--mode train without --mask2former-config-file only supports backbone shape checks. "
                "Set debug.shape_check=true or provide --mask2former-config-file."
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_ENCODER_CONFIG)
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--data-fraction", type=float, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--fpn-dim", type=int, default=None)
    parser.add_argument("--tap-fractions", default=None)
    parser.add_argument(
        "--use-topdown-fpn", action=BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--shape-check", action=BooleanOptionalAction, default=None
    )
    parser.add_argument("--mask2former-config-file", default=DEFAULT_MASK2FORMER_CONFIG)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--mask2former-opts",
        "--opts",
        dest="mask2former_opts",
        nargs="*",
        default=None,
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--freeze-backbone", action=BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action=BooleanOptionalAction, default=None)
    parser.add_argument("--skip-flops", action=BooleanOptionalAction, default=None)
    parser.add_argument(
        "--skip-inference-benchmark", action=BooleanOptionalAction, default=None
    )
    parser.add_argument("--benchmark-max-batches", type=int, default=None)
    parser.add_argument("--eval-output-file", default=None)
    args = parser.parse_args()

    validate_existing_file(args.base_config, "--base-config")
    base_cfg = _load_yaml(args.base_config)
    if args.config:
        validate_existing_file(args.config, "--config")
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

    _apply_mask2former_overrides(cfg, args)
    _validate_run_config(cfg, mode=args.mode)

    mask_cfg = cfg.get("mask2former", {})
    if args.mode == "train":
        if mask_cfg.get("config_file"):
            run_mask2former(cfg, eval_only=False)
        else:
            train(cfg)
    else:
        evaluate(cfg)


if __name__ == "__main__":
    main()
