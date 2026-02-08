import argparse
import itertools
import math
import os
import random
import sys
import time
from typing import Any, Dict, Optional, Tuple

import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
    from detectron2.data import (
        DatasetCatalog,
        MetadataCatalog,
        build_detection_test_loader,
        build_detection_train_loader,
    )
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.evaluation import DatasetEvaluators, SemSegEvaluator
    from detectron2.projects.deeplab import add_deeplab_config
except ImportError as exc:
    raise ImportError("Detectron2 is required to train Mask2Former.") from exc

try:
    from mask2former import add_maskformer2_config
except ImportError as exc:
    raise ImportError("Mask2Former must be on PYTHONPATH.") from exc

from encoders.vit_backbone import add_vit_pyramid_config
import encoders.vit_backbone  # noqa: F401 - registers backbone
from utils.data import (
    validate_encoder_and_data_config,
    validate_existing_file,
    validate_mask2former_config,
)
from utils.logging import write_json_file
from utils.metrics import count_parameters, summarize_timing
from utils.register_voc import maybe_register_voc2012

maybe_register_voc2012()

DEFAULT_MASK2FORMER_CONFIG = "configs/mask2former_voc.yaml"
DEFAULT_ENCODER_CONFIG = "configs/encoder_clip.yaml"
DEFAULT_BASE_ENCODER_CONFIG = "configs/base.yaml"
OVERFIT_SAMPLES_DEFAULT = 20


def _register_overfit_datasets(cfg, num_samples: int) -> None:
    if num_samples <= 0:
        return
    train_names = list(cfg.DATASETS.TRAIN)
    if not train_names:
        return

    suffix = f"_overfit_{num_samples}"
    seed = getattr(cfg, "SEED", -1)
    try:
        seed = int(seed)
    except Exception:
        seed = -1
    if seed < 0:
        seed = 0

    def _subset_dataset(dataset):
        dataset = list(dataset)
        if len(dataset) <= num_samples:
            return dataset
        rng = random.Random(seed)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        return [dataset[i] for i in indices[:num_samples]]

    def _register_subset(name: str) -> str:
        if name.endswith(suffix):
            return name
        if name not in DatasetCatalog.list():
            print(f"[train_mask2former] Overfit requested but dataset '{name}' is not registered.")
            return name
        subset_name = f"{name}{suffix}"
        if subset_name not in DatasetCatalog.list():
            DatasetCatalog.register(
                subset_name,
                lambda base_name=name: _subset_dataset(DatasetCatalog.get(base_name)),
            )
            meta = dict(MetadataCatalog.get(name).as_dict())
            meta.pop("name", None)
            MetadataCatalog.get(subset_name).set(**meta)
        return subset_name

    cfg.DATASETS.TRAIN = tuple(_register_subset(name) for name in train_names)
    print(
        f"[train_mask2former] Overfit-20 enabled: using {num_samples} samples per train dataset "
        f"(seed={seed})."
    )


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
    if not data_cfg.get("override_mask2former_input", False):
        return
    input_size = data_cfg.get("input_size")
    if input_size:
        cfg.INPUT.MIN_SIZE_TRAIN = [input_size]
        cfg.INPUT.MAX_SIZE_TRAIN = input_size
        cfg.INPUT.MIN_SIZE_TEST = input_size
        cfg.INPUT.MAX_SIZE_TEST = input_size


def _maybe_set_max_iter(cfg, max_epochs: int, opts: set) -> None:
    if max_epochs is None or max_epochs <= 0:
        return
    if "SOLVER.MAX_ITER" in opts:
        return
    dataset_names = cfg.DATASETS.TRAIN
    if not dataset_names:
        return
    dataset_size = 0
    for name in dataset_names:
        try:
            dataset_size += len(DatasetCatalog.get(name))
        except Exception as exc:
            print(f"[train_mask2former] Skipping MAX_ITER override; dataset '{name}' not registered ({exc}).")
            return
    if dataset_size == 0:
        return
    ims_per_batch = max(1, int(cfg.SOLVER.IMS_PER_BATCH))
    iters_per_epoch = int(math.ceil(dataset_size / ims_per_batch))
    cfg.SOLVER.MAX_ITER = max_epochs * iters_per_epoch
    print(
        f"[train_mask2former] MAX_ITER set to {cfg.SOLVER.MAX_ITER} "
        f"({max_epochs} epochs, {dataset_size} samples, batch {ims_per_batch})"
    )


def _build_adamw(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    weight_decay = float(cfg.SOLVER.WEIGHT_DECAY)
    weight_decay_norm = float(getattr(cfg.SOLVER, "WEIGHT_DECAY_NORM", 0.0) or 0.0)
    weight_decay_embed = float(getattr(cfg.SOLVER, "WEIGHT_DECAY_EMBED", 0.0) or 0.0)
    backbone_mult = float(getattr(cfg.SOLVER, "BACKBONE_MULTIPLIER", 1.0) or 1.0)
    base_lr = float(cfg.SOLVER.BASE_LR)

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params = []
    memo = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            if value in memo:
                continue
            memo.add(value)

            hyperparams = {"lr": base_lr, "weight_decay": weight_decay}
            if "backbone" in module_name:
                hyperparams["lr"] *= backbone_mult

            # Common "no weight decay" exceptions for transformer-style models.
            lowered = module_param_name.lower()
            if (
                "relative_position_bias_table" in lowered
                or "absolute_pos_embed" in lowered
                or "pos_embed" in lowered
                or "cls_token" in lowered
                or "dist_token" in lowered
            ):
                hyperparams["weight_decay"] = 0.0

            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = weight_decay_norm
            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = weight_decay_embed

            params.append({"params": [value], **hyperparams})

    betas = (0.9, 0.999)
    if hasattr(cfg.SOLVER, "BETA1") or hasattr(cfg.SOLVER, "BETA2"):
        betas = (
            getattr(cfg.SOLVER, "BETA1", betas[0]),
            getattr(cfg.SOLVER, "BETA2", betas[1]),
        )
    eps = getattr(cfg.SOLVER, "EPS", 1e-8)

    return torch.optim.AdamW(params, lr=base_lr, betas=betas, eps=eps)


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


def _save_eval_results(cfg, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
    if output_path:
        out_file = output_path
    else:
        out_file = os.path.join(cfg.OUTPUT_DIR, "eval_results.json")
    write_json_file(out_file, results)
    return out_file


def _default_eval_hw(cfg) -> Tuple[int, int]:
    min_size_test = cfg.INPUT.MIN_SIZE_TEST
    if isinstance(min_size_test, (list, tuple)):
        short_side = int(min_size_test[0])
    else:
        short_side = int(min_size_test)
    max_size_test = cfg.INPUT.MAX_SIZE_TEST
    max_side = int(max_size_test[0] if isinstance(max_size_test, (list, tuple)) else max_size_test)
    side = min(short_side, max_side)
    return side, side


def _estimate_model_flops(cfg, model: torch.nn.Module) -> Tuple[Optional[float], Optional[str]]:
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as exc:
        return None, f"FLOPs unavailable: {exc}"

    h, w = _default_eval_hw(cfg)
    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    dummy = torch.zeros((3, h, w), device=device)
    inputs = [{"image": dummy, "height": h, "width": w}]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            total_flops = float(FlopCountAnalysis(model, inputs).total())
    except Exception as exc:
        return None, f"FLOPs unavailable: {exc}"
    finally:
        if was_training:
            model.train()
    return total_flops, None


def _benchmark_inference(
    cfg,
    model: torch.nn.Module,
    dataset_name: str,
    max_batches: int,
) -> Dict[str, Any]:
    data_loader = Trainer.build_test_loader(cfg, dataset_name)
    warmup_batches = 5
    all_time = 0.0
    all_images = 0
    measured_time = 0.0
    measured_images = 0

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if max_batches > 0 and idx >= max_batches:
                break
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            batch_size = len(inputs) if isinstance(inputs, (list, tuple)) else 1

            all_time += elapsed
            all_images += batch_size
            if idx >= warmup_batches:
                measured_time += elapsed
                measured_images += batch_size

    if was_training:
        model.train()

    if measured_images == 0:
        measured_time = all_time
        measured_images = all_images
    timing = summarize_timing(measured_time, measured_images)
    return {
        "seconds_per_image": timing["seconds_per_image"],
        "images_per_second": timing["images_per_second"],
        "images_measured": measured_images,
        "max_batches": max_batches,
        "warmup_batches": warmup_batches,
    }


class SemSegEvaluatorWithConfusion(SemSegEvaluator):
    def evaluate(self):
        results = super().evaluate() or {}
        conf_mat = getattr(self, "_conf_matrix", None)
        if conf_mat is None:
            return results

        if conf_mat.shape[0] > 1 and conf_mat.shape[1] > 1:
            conf_mat = conf_mat[:-1, :-1]

        output_dir = getattr(self, "_output_dir", None)
        if output_dir:
            out_file = os.path.join(output_dir, "confusion_matrix.json")
            write_json_file(out_file, {"confusion_matrix": conf_mat})
            print(f"[train_mask2former] Saved confusion matrix to {out_file}")
        return results


class Trainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        if getattr(cfg.MODEL.BACKBONE, "FREEZE", False):
            for param in model.backbone.parameters():
                param.requires_grad = False
            model.backbone.eval()
            print("[train_mask2former] Backbone frozen.")
        return model

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
        # Upstream Mask2Former evaluates with Detectron2's default test loader.
        # Its custom dataset mappers are train-only in this version.
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        os.makedirs(output_folder, exist_ok=True)
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        evaluators = []
        if evaluator_type == "sem_seg":
            evaluators.append(
                SemSegEvaluatorWithConfusion(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if not evaluators:
            raise NotImplementedError(
                f"No evaluator for dataset '{dataset_name}' with evaluator_type='{evaluator_type}'."
            )
        if len(evaluators) == 1:
            return evaluators[0]
        return DatasetEvaluators(evaluators)

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
    validate_mask2former_config(
        {
            "config_file": args.config_file,
            "max_epochs": args.max_epochs,
            "benchmark_max_batches": args.benchmark_max_batches,
            "eval_output_file": args.eval_output_file,
        },
        mode="eval" if args.eval_only else "train",
        config_required=True,
    )
    validate_existing_file(args.config_file, "--config-file")
    if args.encoder_config:
        validate_existing_file(args.encoder_config, "--encoder-config")
    if args.base_encoder_config:
        validate_existing_file(args.base_encoder_config, "--base-encoder-config")

    _patch_mask2former_empty_targets()
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_vit_pyramid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    opts = set(args.opts)
    if args.freeze_backbone:
        cfg.MODEL.BACKBONE.FREEZE = True

    if args.encoder_config:
        base_cfg = _load_yaml(args.base_encoder_config)
        override_cfg = _load_yaml(args.encoder_config)
        merged = _deep_merge(base_cfg, override_cfg)
        encoder_cfg = merged.get("encoder", {})
        data_cfg = merged.get("data", {})
        validate_encoder_and_data_config(
            encoder_cfg,
            data_cfg,
            require_model_name=True,
        )
        _apply_encoder_cfg(cfg, encoder_cfg)
        _apply_input_cfg(cfg, data_cfg)

    if args.eval_only and not args.resume and not getattr(cfg.MODEL, "WEIGHTS", ""):
        raise ValueError(
            "Evaluation requires model weights. Provide MODEL.WEIGHTS via --opts or set --resume."
        )

    if args.overfit_20:
        _register_overfit_datasets(cfg, OVERFIT_SAMPLES_DEFAULT)

    _maybe_set_max_iter(cfg, args.max_epochs, opts)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def _run_eval_only(cfg, args):
    if not cfg.DATASETS.TEST:
        raise ValueError("Evaluation requires cfg.DATASETS.TEST to be set.")
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    task_metrics = Trainer.test(cfg, model)

    report: Dict[str, Any] = {
        "task_metrics": task_metrics,
        "efficiency": {
            "parameters": count_parameters(model),
        },
    }

    if not args.skip_flops:
        total_flops, flops_error = _estimate_model_flops(cfg, model)
        report["efficiency"]["flops"] = {
            "total": total_flops,
            "error": flops_error,
            "input_hw": list(_default_eval_hw(cfg)),
        }

    if not args.skip_inference_benchmark:
        datasets = list(cfg.DATASETS.TEST)
        benchmark = {}
        for dataset_name in datasets:
            benchmark[dataset_name] = _benchmark_inference(
                cfg,
                model,
                dataset_name=dataset_name,
                max_batches=args.benchmark_max_batches,
            )
        report["efficiency"]["inference"] = benchmark

    output_path = _save_eval_results(cfg, report, output_path=args.eval_output_file)
    print(f"[train_mask2former] Evaluation report saved to {output_path}")
    return report


def main(args):
    cfg = setup(args)

    if args.eval_only:
        return _run_eval_only(cfg, args)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def build_parser():
    parser = default_argument_parser()
    parser.set_defaults(config_file=DEFAULT_MASK2FORMER_CONFIG)
    parser.add_argument("--encoder-config", default=DEFAULT_ENCODER_CONFIG)
    parser.add_argument("--base-encoder-config", default=DEFAULT_BASE_ENCODER_CONFIG)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--overfit-20", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument("--skip-inference-benchmark", action="store_true")
    parser.add_argument("--benchmark-max-batches", type=int, default=50)
    parser.add_argument("--eval-output-file", default="")
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
