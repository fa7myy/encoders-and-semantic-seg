import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.train import build_backbone
from utils.data import validate_encoder_and_data_config, validate_mask2former_config
from utils.metrics import count_parameters, summarize_timing


def _as_str_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    return [str(raw)]


def _build_mask2former_argv(cfg: Dict[str, Any], eval_only: bool) -> List[str]:
    mask_cfg = cfg.get("mask2former", {})
    validate_mask2former_config(
        mask_cfg,
        mode="eval" if eval_only else "train",
        config_required=True,
    )
    config_file = mask_cfg["config_file"]

    argv: List[str] = [
        "--config-file",
        str(config_file),
        "--encoder-config",
        str(mask_cfg.get("encoder_config", "configs/encoder_clip.yaml")),
        "--base-encoder-config",
        str(mask_cfg.get("base_encoder_config", "configs/base.yaml")),
        "--max-epochs",
        str(int(mask_cfg.get("max_epochs", 30))),
    ]
    if eval_only:
        argv.append("--eval-only")

    if mask_cfg.get("freeze_backbone", False):
        argv.append("--freeze-backbone")
    if mask_cfg.get("resume", False):
        argv.append("--resume")
    if mask_cfg.get("skip_flops", False):
        argv.append("--skip-flops")
    if mask_cfg.get("skip_inference_benchmark", False):
        argv.append("--skip-inference-benchmark")

    benchmark_max_batches = mask_cfg.get("benchmark_max_batches")
    if benchmark_max_batches is not None:
        argv.extend(["--benchmark-max-batches", str(int(benchmark_max_batches))])

    eval_output_file = mask_cfg.get("eval_output_file")
    if eval_output_file:
        argv.extend(["--eval-output-file", str(eval_output_file)])

    opts = []
    if mask_cfg.get("output_dir"):
        opts.extend(["OUTPUT_DIR", str(mask_cfg["output_dir"])])
    if mask_cfg.get("weights"):
        opts.extend(["MODEL.WEIGHTS", str(mask_cfg["weights"])])
    opts.extend(_as_str_list(mask_cfg.get("opts")))
    if opts:
        argv.append("--opts")
        argv.extend(opts)

    return argv


def run_mask2former(cfg: Dict[str, Any], eval_only: bool):
    from experiments.train_mask2former import build_parser, main as train_mask2former_main

    argv = _build_mask2former_argv(cfg, eval_only=eval_only)
    parser = build_parser()
    args = parser.parse_args(argv)
    return train_mask2former_main(args)


def _benchmark_backbone_forward(
    backbone: torch.nn.Module,
    dummy: torch.Tensor,
    runs: int = 30,
    warmup: int = 5,
) -> Dict[str, Optional[float]]:
    with torch.no_grad():
        for _ in range(warmup):
            backbone(dummy)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            backbone(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return summarize_timing(elapsed, runs)


def _estimate_backbone_flops(backbone: torch.nn.Module, dummy: torch.Tensor) -> Dict[str, Any]:
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as exc:
        return {"total": None, "error": f"FLOPs unavailable: {exc}"}

    try:
        with torch.no_grad():
            total = float(FlopCountAnalysis(backbone, dummy).total())
    except Exception as exc:
        return {"total": None, "error": f"FLOPs unavailable: {exc}"}
    return {"total": total, "error": None}


def _backbone_only_eval(cfg: Dict[str, Any]) -> Dict[str, Any]:
    validate_encoder_and_data_config(cfg.get("encoder", {}), cfg.get("data", {}))
    enc_cfg = cfg.get("encoder", {})
    input_size = cfg.get("data", {}).get("input_size") or enc_cfg.get("img_size", 512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = build_backbone(cfg).to(device)
    backbone.eval()
    dummy = torch.zeros(1, 3, int(input_size), int(input_size), device=device)

    with torch.no_grad():
        features = backbone(dummy)

    report = {
        "mode": "backbone_only",
        "segmentation_metrics": {
            "available": False,
            "reason": "No Mask2Former config provided; ran backbone-only evaluation.",
        },
        "efficiency": {
            "parameters": count_parameters(backbone),
            "inference": _benchmark_backbone_forward(backbone, dummy),
            "flops": _estimate_backbone_flops(backbone, dummy),
        },
        "feature_shapes": {name: list(feat.shape) for name, feat in features.items()},
    }
    return report


def evaluate(cfg: Dict[str, Any]):
    mask_cfg = cfg.get("mask2former", {})
    if mask_cfg.get("config_file"):
        return run_mask2former(cfg, eval_only=True)
    return _backbone_only_eval(cfg)
