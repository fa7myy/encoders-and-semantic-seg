import os
from typing import Any, Dict, Iterable, Sequence


_ALLOWED_ENCODERS = {"clip", "dinov2", "mae"}
_ALLOWED_NECK_TYPES = {"vit_pyramid", "vitdet_sfp"}
_SFP_SUPPORTED_SCALES = (4.0, 2.0, 1.0, 0.5)


def validate_existing_file(path: str, label: str) -> None:
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{label} must be a non-empty path.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _as_positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be an integer. Got: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0. Got: {parsed}")
    return parsed


def _as_fraction_list(value: Any, field_name: str) -> Sequence[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field_name} must be a non-empty list.")
    fractions = []
    for item in value:
        try:
            frac = float(item)
        except Exception as exc:
            raise ValueError(f"{field_name} contains non-numeric value: {item!r}") from exc
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"{field_name} values must be in (0, 1]. Got: {frac}")
        fractions.append(frac)
    return fractions


def _as_positive_int_list(
    value: Any,
    field_name: str,
    expected_len: int,
    *,
    strictly_increasing: bool = False,
) -> Sequence[int]:
    if not isinstance(value, (list, tuple)) or len(value) != expected_len:
        raise ValueError(f"{field_name} must be a list of length {expected_len}.")
    parsed = [_as_positive_int(item, field_name) for item in value]
    if strictly_increasing:
        for prev, curr in zip(parsed, parsed[1:]):
            if curr <= prev:
                raise ValueError(
                    f"{field_name} must be strictly increasing. Got: {parsed}"
                )
    return parsed


def _validate_sfp_compatibility(patch_size: int, out_strides: Iterable[int]) -> None:
    for stride in out_strides:
        scale = float(patch_size) / float(stride)
        if not any(abs(scale - candidate) < 1e-6 for candidate in _SFP_SUPPORTED_SCALES):
            raise ValueError(
                "encoder.neck_type=vitdet_sfp requires out_strides compatible with patch_size. "
                f"Got patch_size={patch_size}, out_stride={stride}, scale={scale}."
            )


def validate_encoder_and_data_config(
    encoder_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    *,
    require_model_name: bool = False,
) -> None:
    if not isinstance(encoder_cfg, dict):
        raise ValueError("encoder config must be a mapping.")
    if not isinstance(data_cfg, dict):
        raise ValueError("data config must be a mapping.")

    if require_model_name and not encoder_cfg.get("model_name"):
        raise ValueError("encoder.model_name is required for Mask2Former runs.")

    encoder_name = encoder_cfg.get("name", "clip")
    if encoder_name not in _ALLOWED_ENCODERS:
        raise ValueError(
            f"encoder.name must be one of {sorted(_ALLOWED_ENCODERS)}. Got: {encoder_name!r}"
        )

    patch_size = _as_positive_int(encoder_cfg.get("patch_size"), "encoder.patch_size")
    _as_positive_int(encoder_cfg.get("embed_dim"), "encoder.embed_dim")

    if encoder_cfg.get("fpn_dim") is not None:
        _as_positive_int(encoder_cfg.get("fpn_dim"), "encoder.fpn_dim")

    neck_type = encoder_cfg.get("neck_type", "vit_pyramid")
    if neck_type not in _ALLOWED_NECK_TYPES:
        raise ValueError(
            f"encoder.neck_type must be one of {sorted(_ALLOWED_NECK_TYPES)}. Got: {neck_type!r}"
        )

    tap_fractions = _as_fraction_list(
        encoder_cfg.get("tap_fractions", [0.25, 0.5, 0.75, 1.0]),
        "encoder.tap_fractions",
    )
    if len(tap_fractions) != 4:
        raise ValueError(
            "encoder.tap_fractions must contain exactly 4 values for P2/P3/P4/P5 taps."
        )

    out_strides = _as_positive_int_list(
        encoder_cfg.get("out_strides", [4, 8, 16, 32]),
        "encoder.out_strides",
        4,
        strictly_increasing=True,
    )
    if neck_type == "vitdet_sfp":
        _validate_sfp_compatibility(patch_size, out_strides)

    img_size = encoder_cfg.get("img_size")
    if img_size is not None:
        img_size = _as_positive_int(img_size, "encoder.img_size")
        if img_size % patch_size != 0:
            raise ValueError(
                "encoder.img_size must be divisible by encoder.patch_size. "
                f"Got img_size={img_size}, patch_size={patch_size}."
            )

    input_size = data_cfg.get("input_size")
    if input_size is not None:
        input_size = _as_positive_int(input_size, "data.input_size")
        if input_size % patch_size != 0:
            raise ValueError(
                "data.input_size must be divisible by encoder.patch_size. "
                f"Got input_size={input_size}, patch_size={patch_size}."
            )

    fraction = data_cfg.get("fraction")
    if fraction is not None:
        try:
            frac = float(fraction)
        except Exception as exc:
            raise ValueError("data.fraction must be numeric.") from exc
        if not (0.0 < frac <= 1.0):
            raise ValueError("data.fraction must be in (0, 1].")


def validate_mask2former_config(
    mask_cfg: Dict[str, Any],
    *,
    mode: str,
    config_required: bool,
) -> None:
    if not isinstance(mask_cfg, dict):
        raise ValueError("mask2former config must be a mapping.")
    if mode not in {"train", "eval"}:
        raise ValueError(f"mode must be 'train' or 'eval'. Got: {mode!r}")

    config_file = mask_cfg.get("config_file")
    if config_required and not config_file:
        raise ValueError("mask2former.config_file is required for this mode.")
    if config_file:
        validate_existing_file(str(config_file), "mask2former.config_file")

    max_epochs = mask_cfg.get("max_epochs")
    if max_epochs is not None:
        _as_positive_int(max_epochs, "mask2former.max_epochs")

    max_batches = mask_cfg.get("benchmark_max_batches")
    if max_batches is not None:
        _as_positive_int(max_batches, "mask2former.benchmark_max_batches")

    opts = mask_cfg.get("opts")
    if opts is not None and not isinstance(opts, (list, tuple)):
        raise ValueError("mask2former.opts must be a list of strings.")

    eval_output_file = mask_cfg.get("eval_output_file")
    if eval_output_file is not None and not isinstance(eval_output_file, str):
        raise ValueError("mask2former.eval_output_file must be a string path.")
