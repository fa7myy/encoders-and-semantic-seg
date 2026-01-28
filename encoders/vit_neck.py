from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, p: int, gn_groups: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(gn_groups, out_ch), num_channels=out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class ViTPyramidNeck(nn.Module):
    """
    Baseline:
      per-level 1x1 proj -> resize to target -> optional 3x3 smoothing

    Optional classic top-down fusion:
      P5 up + P4, etc.
    """

    def __init__(
        self,
        in_dim: int,
        fpn_dim: int,
        num_levels: int,
        use_topdown_fpn: bool,
        use_smoothing: bool,
        use_gn: bool = True,
    ) -> None:
        super().__init__()
        self.use_topdown_fpn = use_topdown_fpn
        self.use_smoothing = use_smoothing

        if use_gn:
            self.proj = nn.ModuleList(
                [ConvGNAct(in_dim, fpn_dim, k=1, p=0) for _ in range(num_levels)]
            )
        else:
            self.proj = nn.ModuleList(
                [nn.Conv2d(in_dim, fpn_dim, kernel_size=1) for _ in range(num_levels)]
            )

        self.smooth: Optional[nn.ModuleList] = None
        if use_smoothing:
            if use_gn:
                self.smooth = nn.ModuleList(
                    [ConvGNAct(fpn_dim, fpn_dim, k=3, p=1) for _ in range(num_levels)]
                )
            else:
                self.smooth = nn.ModuleList(
                    [
                        nn.Conv2d(
                            fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=True
                        )
                        for _ in range(num_levels)
                    ]
                )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        target_sizes: Sequence[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        assert len(features) == len(
            target_sizes
        ), "features and target_sizes must match length"
        resized: List[torch.Tensor] = []

        for idx, x in enumerate(features):
            x = self.proj[idx](x)
            x = F.interpolate(x, size=target_sizes[idx], mode="bilinear", align_corners=False)
            resized.append(x)

        if self.use_topdown_fpn:
            for idx in range(len(resized) - 1, 0, -1):
                up = F.interpolate(resized[idx], size=resized[idx - 1].shape[-2:], mode="nearest")
                resized[idx - 1] = resized[idx - 1] + up

        if self.use_smoothing and self.smooth is not None:
            resized = [self.smooth[i](x) for i, x in enumerate(resized)]

        return resized
