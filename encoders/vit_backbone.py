"""
ViT -> Pyramid backbone for Detectron2 / Mask2Former-style heads.

- Taps intermediate transformer blocks (default: 25/50/75/100% depth)
- Converts patch tokens to (B, C, H, W) feature maps
- Builds P2-P5 by resizing each tapped map to target strides 4/8/16/32
- Optional classic top-down FPN fusion
- Safer handling of prefix tokens and positional embeddings
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

try:  # Optional Detectron2 integration.
    from detectron2.modeling import Backbone
    from detectron2.modeling import BACKBONE_REGISTRY
    from detectron2.modeling import ShapeSpec
    from detectron2.config import CfgNode as CN
    try:
        from detectron2.modeling.backbone import SimpleFeaturePyramid
    except ImportError:
        SimpleFeaturePyramid = None
except ImportError:
    Backbone = nn.Module
    BACKBONE_REGISTRY = None
    ShapeSpec = None
    CN = None
    SimpleFeaturePyramid = None

from .vit_neck import ViTPyramidNeck
from .vit_utils import compute_tap_indices, tokens_to_feature_map_strict, _get_num_prefix_tokens


DEFAULT_TAP_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_OUT_FEATURES = ("p2", "p3", "p4", "p5")
DEFAULT_OUT_STRIDES = (4, 8, 16, 32)
_SFP_SUPPORTED_SCALES = (4.0, 2.0, 1.0, 0.5)


def _infer_sfp_scales(patch_size: int, out_strides: Sequence[int]) -> List[float]:
    scales: List[float] = []
    for stride in out_strides:
        scale = float(patch_size) / float(stride)
        matched = None
        for candidate in _SFP_SUPPORTED_SCALES:
            if abs(scale - candidate) < 1e-6:
                matched = candidate
                break
        if matched is None:
            raise ValueError(
                f"SimpleFeaturePyramid requires strides derived from patch size. "
                f"Got patch_size={patch_size}, stride={stride} (scale={scale})."
            )
        scales.append(matched)
    return scales


class _TimmViTBase(Backbone):
    def __init__(
        self,
        model: nn.Module,
        patch_size: Optional[int] = None,
        embed_dim: Optional[int] = None,
        use_final_norm: bool = True,
        num_prefix_tokens_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model = model
        if patch_size is None:
            raise ValueError("patch_size must be provided in the encoder config.")
        if int(patch_size) <= 0:
            raise ValueError("patch_size must be a positive integer.")
        self.patch_size = int(patch_size)
        if embed_dim is None:
            raise ValueError("embed_dim must be provided in the encoder config.")
        if int(embed_dim) <= 0:
            raise ValueError("embed_dim must be a positive integer.")
        self.embed_dim = int(embed_dim)

        if not hasattr(model, "blocks"):
            raise ValueError("Expected model.blocks (timm-like ViT). Wrap your encoder to look timm-like.")

        self.num_layers = len(getattr(model, "blocks"))
        self.use_final_norm = use_final_norm

        self.num_prefix_tokens = (
            int(num_prefix_tokens_override)
            if num_prefix_tokens_override is not None
            else _get_num_prefix_tokens(model)
        )

    @property
    def size_divisibility(self) -> int:
        return self._size_divisibility

    def output_shape(self) -> Optional[Dict[str, "ShapeSpec"]]:
        if ShapeSpec is None:
            return None
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    # ---- positional embedding helpers ----
    def _interpolate_pos_embed_square(
        self,
        pos_embed: torch.Tensor,
        grid_hw: Tuple[int, int],
        num_prefix_tokens: int,
    ) -> torch.Tensor:
        """
        Interpolate square-grid positional embeddings to (grid_h, grid_w).
        Assumes pretrained patch grid is square.
        """
        if pos_embed.ndim != 3:
            raise ValueError("pos_embed must be (1, T, C)")

        if pos_embed.shape[1] < num_prefix_tokens:
            raise ValueError("pos_embed has fewer tokens than num_prefix_tokens")

        prefix = pos_embed[:, :num_prefix_tokens] if num_prefix_tokens > 0 else pos_embed[:, :0]
        patch = pos_embed[:, num_prefix_tokens:]

        patch_tokens = patch.shape[1]
        grid = int(round(math.sqrt(patch_tokens)))
        if grid * grid != patch_tokens:
            raise ValueError(
                "pos_embed patch tokens are not square; wrap the encoder to provide model._pos_embed "
                "or supply an interpolation function."
            )

        patch = patch.reshape(1, grid, grid, -1).permute(0, 3, 1, 2)
        patch = F.interpolate(patch, size=grid_hw, mode="bicubic", align_corners=False)
        patch = patch.permute(0, 2, 3, 1).reshape(1, grid_hw[0] * grid_hw[1], -1)
        return torch.cat((prefix, patch), dim=1)

    def _apply_pos_embed(
        self,
        x: torch.Tensor,
        grid_hw: Tuple[int, int],
        num_patches: int,
    ) -> torch.Tensor:
        """
        Apply positional embeddings in a timm-compatible way.
        Prefer model._pos_embed if present (it usually handles interpolation).
        """
        if hasattr(self.model, "_pos_embed"):
            return self.model._pos_embed(x)

        b = x.shape[0]

        prefix_tokens = []
        cls_token = getattr(self.model, "cls_token", None)
        if cls_token is not None:
            prefix_tokens.append(cls_token.expand(b, -1, -1))
        dist_token = getattr(self.model, "dist_token", None)
        if dist_token is not None:
            prefix_tokens.append(dist_token.expand(b, -1, -1))

        num_prefix_tokens = sum(t.shape[1] for t in prefix_tokens)

        pos_embed = getattr(self.model, "pos_embed", None)
        if pos_embed is not None:
            if pos_embed.shape[1] != (num_prefix_tokens + num_patches) and pos_embed.shape[1] != num_patches:
                pos_embed = self._interpolate_pos_embed_square(pos_embed, grid_hw, num_prefix_tokens)

        if getattr(self.model, "no_embed_class", False) and pos_embed is not None:
            if pos_embed.shape[1] != num_patches:
                raise ValueError("no_embed_class requires pos_embed with patch tokens only.")
            x = x + pos_embed
            if prefix_tokens:
                x = torch.cat(prefix_tokens + [x], dim=1)
        else:
            if prefix_tokens:
                x = torch.cat(prefix_tokens + [x], dim=1)
            if pos_embed is not None:
                if pos_embed.shape[1] != x.shape[1]:
                    raise ValueError(f"pos_embed token count {pos_embed.shape[1]} != input tokens {x.shape[1]}")
                x = x + pos_embed

        if hasattr(self.model, "pos_drop") and self.model.pos_drop is not None:
            x = self.model.pos_drop(x)

        return x

    # ---- forward taps ----
    def _forward_taps(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], Tuple[int, int], int]:
        """
        Returns:
          tap_tokens: list of (B, T, C) tokens from selected blocks
          grid_hw: (grid_h, grid_w) patch grid
          num_prefix_tokens: number of prefix tokens we assume for token->map conversion
        """
        _, _, h, w = images.shape

        x = self.model.patch_embed(images)

        if x.ndim == 4:
            grid_h, grid_w = x.shape[-2:]
            x = x.flatten(2).transpose(1, 2)
        else:
            num_patches = x.shape[1]
            grid_h = h // self.patch_size
            grid_w = w // self.patch_size
            assert grid_h * grid_w == num_patches

        num_patches = grid_h * grid_w
        x = self._apply_pos_embed(x, (grid_h, grid_w), num_patches)

        tap_tokens: List[torch.Tensor] = []
        for idx, blk in enumerate(self.model.blocks):
            x = blk(x)
            if idx in self.tap_indices:
                tap_tokens.append(x)

        if (
            self.use_final_norm
            and hasattr(self.model, "norm")
            and self.model.norm is not None
            and self.tap_indices
            and self.tap_indices[-1] == len(self.model.blocks) - 1
        ):
            tap_tokens[-1] = self.model.norm(tap_tokens[-1])

        return tap_tokens, (grid_h, grid_w), self.num_prefix_tokens


class TimmViTPyramidBackbone(_TimmViTBase):
    """
    A Detectron2 Backbone wrapper that turns a flat ViT into pyramid features P2-P5.

    Key design:
    - We always produce outputs at strides 4/8/16/32 (by resizing), independent of patch size.
    - Intermediate taps are taken from transformer blocks; tokens are reshaped into grids.
    - Positional embeddings are applied with interpolation if needed (square-grid assumption).

    If your encoder already handles pos-embed interpolation (e.g., via model._pos_embed),
    we use that. Otherwise we implement a conservative interpolation path.
    """

    def __init__(
        self,
        model: nn.Module,
        fpn_dim: int = 256,
        tap_fractions: Sequence[float] = DEFAULT_TAP_FRACTIONS,
        out_features: Sequence[str] = DEFAULT_OUT_FEATURES,
        out_strides: Sequence[int] = DEFAULT_OUT_STRIDES,
        use_topdown_fpn: bool = False,
        use_smoothing: bool = True,
        use_gn: bool = True,
        use_final_norm: bool = True,
        patch_size: Optional[int] = None,
        embed_dim: Optional[int] = None,
        num_prefix_tokens_override: Optional[int] = None,
    ) -> None:
        super().__init__(
            model=model,
            patch_size=patch_size,
            embed_dim=embed_dim,
            use_final_norm=use_final_norm,
            num_prefix_tokens_override=num_prefix_tokens_override,
        )
        self.tap_indices = compute_tap_indices(self.num_layers, tap_fractions)

        self._out_features = list(out_features)
        self._out_feature_strides = {name: stride for name, stride in zip(out_features, out_strides)}
        self._out_feature_channels = {name: fpn_dim for name in out_features}
        self._size_divisibility = max(out_strides)

        self.neck = ViTPyramidNeck(
            in_dim=self.embed_dim,
            fpn_dim=fpn_dim,
            num_levels=len(self._out_features),
            use_topdown_fpn=use_topdown_fpn,
            use_smoothing=use_smoothing,
            use_gn=use_gn,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, _, h, w = x.shape
        tap_tokens, grid_hw, num_prefix = self._forward_taps(x)

        if len(tap_tokens) != len(self._out_features):
            raise ValueError(
                f"Expected {len(self._out_features)} tap outputs, got {len(tap_tokens)}. "
                f"(tap_indices={self.tap_indices})"
            )

        maps = [tokens_to_feature_map_strict(t, grid_hw, num_prefix) for t in tap_tokens]

        target_sizes = [
            (
                int(math.ceil(h / self._out_feature_strides[name])),
                int(math.ceil(w / self._out_feature_strides[name])),
            )
            for name in self._out_features
        ]

        pyramid = self.neck(maps, target_sizes)
        return {name: feat for name, feat in zip(self._out_features, pyramid)}


class TimmViTLastFeatBackbone(_TimmViTBase):
    def __init__(
        self,
        model: nn.Module,
        patch_size: Optional[int] = None,
        embed_dim: Optional[int] = None,
        use_final_norm: bool = True,
        num_prefix_tokens_override: Optional[int] = None,
        out_feature: str = "last_feat",
    ) -> None:
        super().__init__(
            model=model,
            patch_size=patch_size,
            embed_dim=embed_dim,
            use_final_norm=use_final_norm,
            num_prefix_tokens_override=num_prefix_tokens_override,
        )
        self.tap_indices = [self.num_layers - 1]
        self._out_features = [out_feature]
        self._out_feature_strides = {out_feature: self.patch_size}
        self._out_feature_channels = {out_feature: self.embed_dim}
        self._size_divisibility = self.patch_size

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        tap_tokens, grid_hw, num_prefix = self._forward_taps(x)
        if not tap_tokens:
            raise ValueError("No tap outputs available for last feature map.")
        last_map = tokens_to_feature_map_strict(tap_tokens[-1], grid_hw, num_prefix)
        return {self._out_features[0]: last_map}


def build_vitdet_sfp_backbone(
    model: nn.Module,
    fpn_dim: int,
    out_strides: Sequence[int],
    patch_size: int,
    embed_dim: int,
    use_final_norm: bool = True,
    num_prefix_tokens_override: Optional[int] = None,
    sfp_norm: str = "LN",
) -> Backbone:
    if SimpleFeaturePyramid is None:
        raise ImportError(
            "Detectron2 SimpleFeaturePyramid is required for neck_type=vitdet_sfp."
        )

    bottom_up = TimmViTLastFeatBackbone(
        model=model,
        patch_size=patch_size,
        embed_dim=embed_dim,
        use_final_norm=use_final_norm,
        num_prefix_tokens_override=num_prefix_tokens_override,
    )
    scale_factors = _infer_sfp_scales(patch_size, out_strides)
    return SimpleFeaturePyramid(
        net=bottom_up,
        in_feature="last_feat",
        out_channels=fpn_dim,
        scale_factors=scale_factors,
        norm=sfp_norm,
    )


if BACKBONE_REGISTRY is not None:

    @BACKBONE_REGISTRY.register()
    class TimmViTPyramidBackboneRegistered(TimmViTPyramidBackbone):
        pass

    @BACKBONE_REGISTRY.register()
    def build_vit_pyramid_backbone(cfg, input_shape):
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - optional dependency.
            raise ImportError("timm is required to build the ViT backbone.") from exc

        vit_cfg = cfg.MODEL.VIT
        kwargs = {"pretrained": vit_cfg.PRETRAINED}
        if vit_cfg.IMG_SIZE:
            kwargs["img_size"] = vit_cfg.IMG_SIZE
        model = timm.create_model(vit_cfg.MODEL_NAME, **kwargs)
        patch_embed = getattr(model, "patch_embed", None)
        if patch_embed is not None:
            if hasattr(patch_embed, "strict_img_size"):
                patch_embed.strict_img_size = False
            if hasattr(patch_embed, "dynamic_img_size"):
                patch_embed.dynamic_img_size = True
        if hasattr(model, "dynamic_img_size"):
            model.dynamic_img_size = True

        num_prefix_override = None
        if hasattr(vit_cfg, "NUM_PREFIX_TOKENS") and vit_cfg.NUM_PREFIX_TOKENS >= 0:
            num_prefix_override = int(vit_cfg.NUM_PREFIX_TOKENS)

        if vit_cfg.NECK_TYPE == "vitdet_sfp":
            return build_vitdet_sfp_backbone(
                model=model,
                fpn_dim=vit_cfg.FPN_DIM,
                out_strides=vit_cfg.OUT_STRIDES,
                patch_size=vit_cfg.PATCH_SIZE,
                embed_dim=vit_cfg.EMBED_DIM,
                use_final_norm=vit_cfg.USE_FINAL_NORM,
                num_prefix_tokens_override=num_prefix_override,
                sfp_norm=vit_cfg.SFP_NORM,
            )

        return TimmViTPyramidBackbone(
            model=model,
            fpn_dim=vit_cfg.FPN_DIM,
            tap_fractions=vit_cfg.TAP_FRACTIONS,
            out_features=DEFAULT_OUT_FEATURES,
            out_strides=vit_cfg.OUT_STRIDES,
            use_topdown_fpn=vit_cfg.USE_TOPDOWN_FPN,
            use_smoothing=vit_cfg.USE_SMOOTHING,
            use_gn=vit_cfg.USE_GN,
            use_final_norm=vit_cfg.USE_FINAL_NORM,
            patch_size=vit_cfg.PATCH_SIZE,
            embed_dim=vit_cfg.EMBED_DIM,
            num_prefix_tokens_override=num_prefix_override,
        )


def add_vit_pyramid_config(cfg):
    if CN is None:
        raise ImportError("Detectron2 is required to use add_vit_pyramid_config.")
    cfg.MODEL.VIT = CN()
    cfg.MODEL.VIT.MODEL_NAME = "vit_base_patch16_clip_224.openai"
    cfg.MODEL.VIT.PRETRAINED = True
    cfg.MODEL.VIT.IMG_SIZE = 0
    cfg.MODEL.VIT.PATCH_SIZE = 16
    cfg.MODEL.VIT.EMBED_DIM = 768
    cfg.MODEL.VIT.FPN_DIM = 256
    cfg.MODEL.VIT.TAP_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
    cfg.MODEL.VIT.OUT_STRIDES = [4, 8, 16, 32]
    cfg.MODEL.VIT.USE_TOPDOWN_FPN = False
    cfg.MODEL.VIT.USE_SMOOTHING = True
    cfg.MODEL.VIT.USE_GN = True
    cfg.MODEL.VIT.USE_FINAL_NORM = True
    cfg.MODEL.VIT.NUM_PREFIX_TOKENS = -1
    cfg.MODEL.VIT.NECK_TYPE = "vit_pyramid"
    cfg.MODEL.VIT.SFP_NORM = "LN"
