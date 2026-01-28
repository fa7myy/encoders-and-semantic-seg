# Setup Guide

## Quickstart (Backbone Tests Only)

1) Install dependencies:

```
pip install -r requirements.txt
```

2) Run a tap test:

```
python experiments/test_backbone_taps.py --config configs/encoder_clip.yaml
```

3) Show heatmaps (tap + neck):

```
python experiments/test_backbone_taps.py --config configs/encoder_clip.yaml --show-heatmaps
```

---

## Mask2Former Training (Linux or WSL2)

Mask2Former depends on Detectron2 CUDA extensions, which are reliable on Linux/WSL2 but not on native Windows.

1) Install PyTorch with CUDA support.
2) Install Detectron2 (from source or a compatible wheel).
3) Install Mask2Former and add it to `PYTHONPATH`.
4) Run training:

```
python experiments/train_mask2former.py \
  --config-file path/to/mask2former_config.yaml \
  --encoder-config configs/encoder_clip.yaml
```

---

## Notes

- `configs/encoder_dinov2.yaml` uses `input_size: 518` to match patch size 14.
- Encoder configs must include `patch_size` and `embed_dim`.
- For DINOv2/CLIP/MAE, model weights are pulled via `timm` when `pretrained: true`.
