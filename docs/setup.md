# Setup Guide

## Quickstart (Backbone Tests Only)

1) Install dependencies:

```
pip install -e ".[cpu]"
```

For GPU, use:

```
pip install -e ".[gpu]"
```

If you need Detectron2 on CPU:

```
pip install -e ".[cpu-d2]"
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

1) Install the stack:

```
# CPU (WSL/testing)
pip install -e ".[cpu]"
pip install -e ".[cpu-d2]"

# GPU (T4/CUDA 11.1)
pip install -e ".[gpu]"
```

2) Clone Mask2Former and add it to `PYTHONPATH`:

```
git clone https://github.com/facebookresearch/Mask2Former.git
cd Mask2Former
git checkout 9b0651c
export PYTHONPATH="$PWD:$PYTHONPATH"
```

3) For GPU, build the MSDeformAttn CUDA op:

```
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

4) Run training:

```
python experiments/train_mask2former.py \
  --config-file path/to/mask2former_config.yaml \
  --encoder-config configs/encoder_clip.yaml
```

Optional training flags:

- `--freeze-backbone` to keep the encoder frozen.
- `--max-epochs N` to cap training by epoch count (default: 30). This is converted to `SOLVER.MAX_ITER`.
  If you want to set iterations directly, pass `SOLVER.MAX_ITER` on the command line.

Memory-friendly defaults applied by `train_mask2former.py` (override on the CLI if needed):

- `SOLVER.IMS_PER_BATCH = 2`
- `INPUT.MIN/MAX_SIZE_{TRAIN,TEST} = 384`, `INPUT.CROP.SIZE = [384, 384]`
- `INPUT.SIZE_DIVISIBILITY = 32`
- `MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 50`
- `MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 8192`

---

## Notes

- Extras are defined in `pyproject.toml`: use `.[cpu]` for WSL/CPU and `.[gpu]` for T4/CUDA.
- Install `.[cpu]` before `.[cpu-d2]` so Detectron2 can import torch during setup.
- Mask2Former is not pip-installable; it must be cloned and added to `PYTHONPATH`.
- Mask2Former’s MSDeformAttn op is CUDA-only; CPU installs are best for quick config/tests.
- `configs/encoder_dinov2.yaml` uses `input_size: 518` to match patch size 14.
- Encoder configs must include `patch_size` and `embed_dim`.
- For DINOv2/CLIP/MAE, model weights are pulled via `timm` when `pretrained: true`.
