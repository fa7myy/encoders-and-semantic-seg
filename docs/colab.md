# Google Colab Guide

This guide shows how to run encoder sanity checks and full Mask2Former training
in Google Colab after cloning this repo.

## 0) Runtime

1) In Colab: `Runtime -> Change runtime type -> GPU` (T4/A100 is fine).
2) `Runtime -> Restart runtime` after installing packages.

## 1) Clone the repo

```bash
!git clone https://github.com/fa7myy/encoders-and-semantic-seg
%cd encoders-and-semantic-seg
```

## 2) Choose your path

### A) Encoder sanity checks (no Mask2Former)

These tests only require `torch`, `timm`, `PyYAML`, and `Pillow`.

```bash
!pip -q install timm==0.6.12 PyYAML==6.0 Pillow==9.2.0
```

Torch is usually preinstalled on Colab. If not:

```bash
!pip -q install torch torchvision
```

Run a backbone tap test:

```bash
!python experiments/test_backbone_taps.py --config configs/encoder_clip.yaml
```

Try other encoders:

```bash
!python experiments/test_backbone_taps.py --config configs/encoder_dinov2.yaml
!python experiments/test_backbone_taps.py --config configs/encoder_mae.yaml
```

### B) Full Mask2Former training (GPU, Python 3.8)

This path follows `docs/setup.md` and assumes Python 3.8 with CUDA 11.1,
which matches the pinned wheels in `pyproject.toml`.

Check your Python version:

```python
import sys
print(sys.version)
```

If you are not on Python 3.8, the pinned `torch==1.9.1+cu111` wheels will not
install. In that case, use a local Linux/WSL2 environment, or create a custom
Colab environment that runs Python 3.8.

Install the GPU stack (Python 3.8 only):

```bash
!pip -q install -e ".[gpu]"
```

Clone Mask2Former and add it to `PYTHONPATH`:

```bash
!git clone https://github.com/facebookresearch/Mask2Former.git
%cd Mask2Former
!git checkout 9b0651c
```

Build the MSDeformAttn CUDA op:

```bash
%cd mask2former/modeling/pixel_decoder/ops
!sh make.sh
```

In a Python cell, export `PYTHONPATH` so Colab sees Mask2Former:

```python
import os
import sys

mask2former_root = "/content/Mask2Former"
if mask2former_root not in sys.path:
    sys.path.insert(0, mask2former_root)
os.environ["PYTHONPATH"] = mask2former_root + ":" + os.environ.get("PYTHONPATH", "")
```

Run training from this repo (choose a Mask2Former config that matches your
dataset):

```bash
%cd /content/encoders-and-semantic-seg
!python experiments/train_mask2former.py \
  --config-file /content/Mask2Former/configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --encoder-config configs/encoder_clip.yaml
```

Swap encoder configs as needed:

```bash
!python experiments/train_mask2former.py \
  --config-file /content/Mask2Former/configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml \
  --encoder-config configs/encoder_dinov2.yaml
```

## 3) Datasets

- Mask2Former configs expect Detectron2-registered datasets (COCO, ADE20K, etc).
- If you want to train on a custom dataset (for example CamVid), you must
  register it in Detectron2 and update the Mask2Former config accordingly.
- The repo does not include dataset registration code.

## 4) Outputs

Training artifacts are saved under `outputs/` by default (configurable via
`configs/base.yaml`).
