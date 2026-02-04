import argparse
import itertools
import os
import sys
from typing import Any, Dict

import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
    from detectron2.data import build_detection_test_loader, build_detection_train_loader
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


def _patch_mask2former_empty_targets() -> None:
    try:
        from mask2former.modeling import matcher as m2f_matcher
    except Exception:
        return

    if getattr(m2f_matcher, "_EMPTY_TARGETS_PATCHED", False):
        return

    def _wrap_empty_targets(fn):
        def _wrapped(inputs, targets):
            if targets is None or targets.numel() == 0 or targets.shape[0] == 0:
                return inputs.new_zeros((inputs.shape[0], 0))
            return fn(inputs, targets)

        return _wrapped

    if hasattr(m2f_matcher, "batch_dice_loss_jit"):
        m2f_matcher.batch_dice_loss_jit = _wrap_empty_targets(m2f_matcher.batch_dice_loss_jit)
    if hasattr(m2f_matcher, "batch_sigmoid_ce_loss_jit"):
        m2f_matcher.batch_sigmoid_ce_loss_jit = _wrap_empty_targets(
            m2f_matcher.batch_sigmoid_ce_loss_jit
        )

    m2f_matcher._EMPTY_TARGETS_PATCHED = True


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


def _build_adamw(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or ("norm" in name.lower()) or ("bn" in name.lower()):
            no_decay.append(param)
        else:
            decay.append(param)

    params = [
        {"params": decay, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    betas = (0.9, 0.999)
    if hasattr(cfg.SOLVER, "BETA1") or hasattr(cfg.SOLVER, "BETA2"):
        betas = (
            getattr(cfg.SOLVER, "BETA1", betas[0]),
            getattr(cfg.SOLVER, "BETA2", betas[1]),
        )
    eps = getattr(cfg.SOLVER, "EPS", 1e-8)

    return torch.optim.AdamW(
        params,
        lr=cfg.SOLVER.BASE_LR,
        betas=betas,
        eps=eps,
    )


def _wrap_full_model_grad_clip(
    optimizer: torch.optim.Optimizer,
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.optim.Optimizer:
    class _Wrapped(type(optimizer)):
        def step(self, closure=None):
            params = itertools.chain(*[group["params"] for group in self.param_groups])
            torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm, norm_type=norm_type)
            return super().step(closure=closure)

    optimizer.__class__ = _Wrapped
    return optimizer


def _build_warmup_poly_lr(cfg, optimizer: torch.optim.Optimizer):
    max_iter = cfg.SOLVER.MAX_ITER
    warmup_iters = cfg.SOLVER.WARMUP_ITERS
    warmup_factor = cfg.SOLVER.WARMUP_FACTOR
    power = getattr(cfg.SOLVER, "POLY_LR_POWER", 0.9)

    def lr_lambda(iteration: int) -> float:
        if iteration < warmup_iters:
            alpha = float(iteration) / max(1, warmup_iters)
            return warmup_factor * (1 - alpha) + alpha
        t = float(iteration - warmup_iters)
        T = float(max(1, max_iter - warmup_iters))
        return (1.0 - t / T) ** power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper_name = getattr(cfg.INPUT, "DATASET_MAPPER_NAME", "")
        mapper = None
        if mapper_name == "mask_former_semantic":
            from mask2former.data.dataset_mappers.mask_former_semantic_dataset_mapper import (
                MaskFormerSemanticDatasetMapper,
            )

            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
        elif mapper_name == "mask_former_instance":
            from mask2former.data.dataset_mappers.mask_former_instance_dataset_mapper import (
                MaskFormerInstanceDatasetMapper,
            )

            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
        elif mapper_name == "mask_former_panoptic":
            from mask2former.data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
                MaskFormerPanopticDatasetMapper,
            )

            mapper = MaskFormerPanopticDatasetMapper(cfg, True)

        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper_name = getattr(cfg.INPUT, "DATASET_MAPPER_NAME", "")
        mapper = None
        if mapper_name == "mask_former_semantic":
            from mask2former.data.dataset_mappers.mask_former_semantic_dataset_mapper import (
                MaskFormerSemanticDatasetMapper,
            )

            mapper = MaskFormerSemanticDatasetMapper(cfg, False)
        elif mapper_name == "mask_former_instance":
            from mask2former.data.dataset_mappers.mask_former_instance_dataset_mapper import (
                MaskFormerInstanceDatasetMapper,
            )

            mapper = MaskFormerInstanceDatasetMapper(cfg, False)
        elif mapper_name == "mask_former_panoptic":
            from mask2former.data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
                MaskFormerPanopticDatasetMapper,
            )

            mapper = MaskFormerPanopticDatasetMapper(cfg, False)

        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        opt_name = getattr(cfg.SOLVER, "OPTIMIZER", "SGD").upper()
        clip_cfg = cfg.SOLVER.CLIP_GRADIENTS

        if opt_name == "ADAMW":
            optimizer = _build_adamw(cfg, model)
        else:
            if clip_cfg.ENABLED and clip_cfg.CLIP_TYPE == "full_model":
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=cfg.SOLVER.BASE_LR,
                    momentum=cfg.SOLVER.MOMENTUM,
                    nesterov=cfg.SOLVER.NESTEROV,
                    weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                )
            else:
                optimizer = super().build_optimizer(cfg, model)

        if clip_cfg.ENABLED and clip_cfg.CLIP_TYPE == "full_model":
            optimizer = _wrap_full_model_grad_clip(
                optimizer,
                max_norm=clip_cfg.CLIP_VALUE,
                norm_type=clip_cfg.NORM_TYPE,
            )

        opt_class = optimizer.__class__.__name__
        if opt_class == "_Wrapped" and len(optimizer.__class__.__mro__) > 1:
            base_class = optimizer.__class__.__mro__[1].__name__
            opt_class = f"{base_class} (wrapped for full_model clip)"
        print(f"[train_mask2former] Using optimizer: {opt_class}")
        return optimizer

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        name = getattr(cfg.SOLVER, "LR_SCHEDULER_NAME", "WarmupMultiStepLR")
        if name == "WarmupPolyLR":
            return _build_warmup_poly_lr(cfg, optimizer)
        return super().build_lr_scheduler(cfg, optimizer)


def setup(args):
    _patch_mask2former_empty_targets()
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_vit_pyramid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    opts = set(args.opts)
    if "SOLVER.IMS_PER_BATCH" not in args.opts:
        cfg.SOLVER.IMS_PER_BATCH = 2

    if args.encoder_config:
        base_cfg = _load_yaml(args.base_encoder_config)
        override_cfg = _load_yaml(args.encoder_config)
        merged = _deep_merge(base_cfg, override_cfg)
        encoder_cfg = merged.get("encoder", {})
        data_cfg = merged.get("data", {})
        _apply_encoder_cfg(cfg, encoder_cfg)
        _apply_input_cfg(cfg, data_cfg)

    if "INPUT.MIN_SIZE_TRAIN" not in opts:
        cfg.INPUT.MIN_SIZE_TRAIN = [384]
    if "INPUT.MAX_SIZE_TRAIN" not in opts:
        cfg.INPUT.MAX_SIZE_TRAIN = 384
    if "INPUT.MIN_SIZE_TEST" not in opts:
        cfg.INPUT.MIN_SIZE_TEST = 384
    if "INPUT.MAX_SIZE_TEST" not in opts:
        cfg.INPUT.MAX_SIZE_TEST = 384
    if "INPUT.CROP.SIZE" not in opts:
        cfg.INPUT.CROP.SIZE = [384, 384]
    if "INPUT.SIZE_DIVISIBILITY" not in opts:
        cfg.INPUT.SIZE_DIVISIBILITY = 32
    if "MODEL.MASK_FORMER.NUM_OBJECT_QUERIES" not in opts:
        cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 50
    if "MODEL.MASK_FORMER.TRAIN_NUM_POINTS" not in opts:
        cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 8192

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return Trainer.test(cfg, model)

    trainer = Trainer(cfg)
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
