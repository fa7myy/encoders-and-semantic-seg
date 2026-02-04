import argparse
import os
import sys
from typing import Any, Dict

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.projects.deeplab import add_deeplab_config
except ImportError as exc:
    raise ImportError("Detectron2 is required to train Mask2Former.") from exc

try:
    from mask2former import add_maskformer2_config
except ImportError as exc:
    raise ImportError("Mask2Former must be on PYTHONPATH.") from exc

from encoders.vit_backbone import add_vit_pyramid_config
import encoders.vit_backbone  # noqa: F401 - registers backbone
from utils.register_voc import maybe_register_voc2012

maybe_register_voc2012()


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


def _apply_encoder_cfg(cfg, encoder_cfg: Dict[str, Any]) -> None:
    cfg.MODEL.BACKBONE.NAME = "build_vit_pyramid_backbone"
    cfg.MODEL.VIT.MODEL_NAME = encoder_cfg["model_name"]
    cfg.MODEL.VIT.PRETRAINED = encoder_cfg.get("pretrained", True)
    cfg.MODEL.VIT.IMG_SIZE = encoder_cfg.get("img_size", 0) or 0
    cfg.MODEL.VIT.PATCH_SIZE = encoder_cfg["patch_size"]
    cfg.MODEL.VIT.EMBED_DIM = encoder_cfg["embed_dim"]
    cfg.MODEL.VIT.FPN_DIM = encoder_cfg.get("fpn_dim", 256)
    cfg.MODEL.VIT.TAP_FRACTIONS = encoder_cfg.get(
        "tap_fractions", [0.25, 0.5, 0.75, 1.0]
    )
    cfg.MODEL.VIT.OUT_STRIDES = encoder_cfg.get("out_strides", [4, 8, 16, 32])
    cfg.MODEL.VIT.USE_TOPDOWN_FPN = encoder_cfg.get("use_topdown_fpn", False)
    cfg.MODEL.VIT.USE_SMOOTHING = encoder_cfg.get("use_smoothing", True)
    cfg.MODEL.VIT.USE_GN = encoder_cfg.get("use_gn", True)
    cfg.MODEL.VIT.USE_FINAL_NORM = encoder_cfg.get("use_final_norm", True)
    if encoder_cfg.get("neck_type") is not None:
        cfg.MODEL.VIT.NECK_TYPE = encoder_cfg["neck_type"]
    if encoder_cfg.get("sfp_norm") is not None:
        cfg.MODEL.VIT.SFP_NORM = encoder_cfg["sfp_norm"]
    if encoder_cfg.get("num_prefix_tokens") is not None:
        cfg.MODEL.VIT.NUM_PREFIX_TOKENS = int(encoder_cfg["num_prefix_tokens"])

    if hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
        cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
        if hasattr(cfg.MODEL.SEM_SEG_HEAD, "DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES"):
            cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = [
                "p3",
                "p4",
                "p5",
            ]
        if hasattr(cfg.MODEL.SEM_SEG_HEAD, "PROJECT_FEATURES"):
            cfg.MODEL.SEM_SEG_HEAD.PROJECT_FEATURES = ["p2"]


def _apply_input_cfg(cfg, data_cfg: Dict[str, Any]) -> None:
    input_size = data_cfg.get("input_size")
    if input_size:
        cfg.INPUT.MIN_SIZE_TRAIN = [input_size]
        cfg.INPUT.MAX_SIZE_TRAIN = input_size
        cfg.INPUT.MIN_SIZE_TEST = input_size
        cfg.INPUT.MAX_SIZE_TEST = input_size


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_vit_pyramid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if args.encoder_config:
        base_cfg = _load_yaml(args.base_encoder_config)
        override_cfg = _load_yaml(args.encoder_config)
        merged = _deep_merge(base_cfg, override_cfg)
        encoder_cfg = merged.get("encoder", {})
        data_cfg = merged.get("data", {})
        _apply_encoder_cfg(cfg, encoder_cfg)
        _apply_input_cfg(cfg, data_cfg)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return DefaultTrainer.test(cfg, model)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def build_parser():
    parser = default_argument_parser()
    parser.add_argument("--encoder-config", default="configs/base.yaml")
    parser.add_argument("--base-encoder-config", default="configs/base.yaml")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
