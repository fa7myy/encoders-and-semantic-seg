"""
Compatibility re-exports for the ViT pyramid backbone and helpers.
"""

from .vit_backbone import (
    DEFAULT_OUT_FEATURES,
    DEFAULT_OUT_STRIDES,
    DEFAULT_TAP_FRACTIONS,
    TimmViTPyramidBackbone,
    add_vit_pyramid_config,
)
from .vit_neck import ConvGNAct, ViTPyramidNeck
from .vit_utils import compute_tap_indices, tokens_to_feature_map_strict, _get_num_prefix_tokens

__all__ = [
    "DEFAULT_OUT_FEATURES",
    "DEFAULT_OUT_STRIDES",
    "DEFAULT_TAP_FRACTIONS",
    "TimmViTPyramidBackbone",
    "add_vit_pyramid_config",
    "ConvGNAct",
    "ViTPyramidNeck",
    "compute_tap_indices",
    "tokens_to_feature_map_strict",
    "_get_num_prefix_tokens",
]
