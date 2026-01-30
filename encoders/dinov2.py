from typing import Optional, Sequence

try:
    import timm
except Exception as exc:  # pragma: no cover - handled at runtime.
    timm = None

from .vit_pyramid import (
    DEFAULT_OUT_STRIDES,
    DEFAULT_TAP_FRACTIONS,
    TimmViTPyramidBackbone,
    build_vitdet_sfp_backbone,
)

DEFAULT_MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"


def build_dinov2_backbone(
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: bool = True,
    fpn_dim: int = 256,
    tap_fractions: Sequence[float] = DEFAULT_TAP_FRACTIONS,
    out_strides: Optional[Sequence[int]] = None,
    use_topdown_fpn: bool = False,
    use_smoothing: bool = True,
    use_final_norm: bool = True,
    img_size: Optional[int] = None,
    patch_size: Optional[int] = None,
    embed_dim: Optional[int] = None,
    neck_type: str = "vit_pyramid",
    sfp_norm: str = "LN",
) -> TimmViTPyramidBackbone:
    if timm is None:
        raise ImportError("timm is required to build the DINOv2 ViT backbone.")
    if patch_size is None:
        raise ValueError("patch_size must be provided in the encoder config.")
    if embed_dim is None:
        raise ValueError("embed_dim must be provided in the encoder config.")
    kwargs = {"pretrained": pretrained}
    if img_size is not None:
        kwargs["img_size"] = img_size
    model = timm.create_model(model_name, **kwargs)
    if neck_type == "vitdet_sfp":
        return build_vitdet_sfp_backbone(
            model=model,
            fpn_dim=fpn_dim,
            out_strides=out_strides or DEFAULT_OUT_STRIDES,
            patch_size=patch_size,
            embed_dim=embed_dim,
            use_final_norm=use_final_norm,
            sfp_norm=sfp_norm,
        )
    return TimmViTPyramidBackbone(
        model=model,
        fpn_dim=fpn_dim,
        tap_fractions=tap_fractions,
        out_strides=out_strides or DEFAULT_OUT_STRIDES,
        use_topdown_fpn=use_topdown_fpn,
        use_smoothing=use_smoothing,
        use_final_norm=use_final_norm,
        patch_size=patch_size,
        embed_dim=embed_dim,
    )
