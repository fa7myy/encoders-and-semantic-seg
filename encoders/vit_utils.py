import math
from typing import List, Sequence, Tuple

import torch


def compute_tap_indices(num_layers: int, fractions: Sequence[float]) -> List[int]:
    """
    Converts fractions of depth into unique 0-based layer indices.
    Example: num_layers=12, fractions=(.25,.5,.75,1.0) -> [2,5,8,11]
    """
    raw: List[int] = []
    for frac in fractions:
        idx_1b = int(math.ceil(frac * num_layers))
        idx_1b = max(1, min(num_layers, idx_1b))
        raw.append(idx_1b - 1)

    indices: List[int] = []
    used = set()
    for idx in raw:
        if idx not in used:
            indices.append(idx)
            used.add(idx)
            continue

        offset = 1
        candidate = None
        while True:
            up = idx + offset
            down = idx - offset
            if up < num_layers and up not in used:
                candidate = up
                break
            if down >= 0 and down not in used:
                candidate = down
                break
            if up >= num_layers and down < 0:
                break
            offset += 1

        if candidate is None:
            raise ValueError("Unable to find unique tap index.")
        indices.append(candidate)
        used.add(candidate)

    return indices


def _get_num_prefix_tokens(model) -> int:
    """
    Conservative prefix token counting:
    - cls_token: (1, 1, C)
    - dist_token: (1, 1, C) (DeiT)
    We intentionally do NOT guess about register_tokens by default.
    If you have a model that uses them, pass num_prefix_tokens_override.
    """
    n = 0
    for name in ("cls_token", "dist_token"):
        tok = getattr(model, name, None)
        if tok is not None:
            if tok.ndim != 3:
                continue
            n += tok.shape[1]
    return int(n)


def tokens_to_feature_map_strict(
    tokens: torch.Tensor,
    grid_hw: Tuple[int, int],
    num_prefix_tokens: int,
) -> torch.Tensor:
    """
    Strict conversion:
    - expects total_tokens == num_prefix_tokens + num_patches OR == num_patches
    - will raise if mismatched (prevents silent misalignment)
    """
    grid_h, grid_w = grid_hw
    num_patches = grid_h * grid_w
    total_tokens = tokens.shape[1]

    if total_tokens == num_patches:
        patch_tokens = tokens
    elif total_tokens == num_prefix_tokens + num_patches:
        patch_tokens = tokens[:, num_prefix_tokens:, :]
    elif total_tokens == num_patches + 1 and num_prefix_tokens == 1:
        patch_tokens = tokens[:, 1:, :]
    else:
        raise ValueError(
            f"Token count mismatch: total={total_tokens}, "
            f"num_prefix={num_prefix_tokens}, num_patches={num_patches}. "
            "Provide correct num_prefix_tokens or wrap the encoder."
        )

    b, _, c = patch_tokens.shape
    return patch_tokens.transpose(1, 2).reshape(b, c, grid_h, grid_w)
