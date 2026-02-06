from typing import Dict, Optional

import torch


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = int(sum(param.numel() for param in model.parameters()))
    trainable = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def summarize_timing(total_seconds: float, total_items: int) -> Dict[str, Optional[float]]:
    if total_items <= 0 or total_seconds <= 0:
        return {"seconds_per_image": None, "images_per_second": None}
    seconds_per_image = float(total_seconds) / float(total_items)
    return {
        "seconds_per_image": seconds_per_image,
        "images_per_second": 1.0 / seconds_per_image,
    }
