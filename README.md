
# Vision Encoder Ablations for Segmentation

This repository contains the complete experimental codebase for studying the effect of pretrained vision encoders such as CLIP, DINOv2, and MAE on semantic segmentation performance using Mask2Former.

The code is designed to be environment agnostic and can be executed from Google Colab or any Linux machine once dependencies are installed.

---

## Research Goal

This project investigates the following question:

How much do pretrained vision encoders help segmentation?

The study measures the impact of different encoders on:

- Training speed and convergence
- Data efficiency in low data regimes
- Final segmentation accuracy
- Stability across random seeds
- Cross dataset generalization

The segmentation architecture Mask2Former is kept fixed. Only the encoder is varied.

---

## High Level Design

- Mask2Former is used as the segmentation framework
- The official Mask2Former repository is used without modification
- Encoder integrations are implemented in this repository
- Google Colab is used only as a runtime environment

---

## Repository Structure

```

.
├── experiments/
│   ├── run_experiment.py
│   ├── train.py
│   └── evaluate.py
│
├── encoders/
│   ├── clip.py
│   ├── dinov2.py
│   └── mae.py
│
├── configs/
│   ├── base.yaml
│   ├── encoder_clip.yaml
│   ├── encoder_dinov2.yaml
│   ├── encoder_mae.yaml
│   └── low_data/
│
├── utils/
│   ├── logging.py
│   ├── data.py
│   └── metrics.py
│
└── README.md

```

---

## Environment Setup

This repository assumes the environment has already been prepared with:

- Python 3.9
- CUDA enabled PyTorch
- Detectron2 built from source
- Mask2Former from the official repository

A Colab notebook is provided separately to handle environment installation.

Once the environment is ready:

```

git clone https://github.com/fa7myy/encoders-and-semantic-seg
cd encoders-and-semantic-seg

```

---

## Running Experiments

All experiments are launched through a single entry point.

Example running DINOv2 with 10 percent of the data:

```

python experiments/run_experiment.py 
--encoder dinov2 
--data-fraction 0.1 
--config configs/encoder_dinov2.yaml

```

Example running CLIP:

```

python experiments/run_experiment.py 
--encoder clip 
--config configs/encoder_clip.yaml

```

---

## Reproducibility

- All hyperparameters are stored in configuration files
- Random seeds are explicitly set
- Environment versions are logged at runtime
- Results are saved with encoder dataset and seed identifiers

This enables exact reproduction of all experiments.

---

## Evaluation Metrics

Models are evaluated across accuracy, efficiency, and learning behavior using the following metrics.

### Segmentation Quality
- **Mean Intersection over Union (mIoU)**  
  Measures overlap between predicted and ground truth masks. Primary accuracy metric.
- **Mean Class Accuracy**  
  Average per class accuracy. Evaluates performance balance across classes.

### Error Analysis
- **Confusion Matrix**  
  Shows class level prediction errors and label confusions.

### Efficiency
- **Inference Time**  
  Average time to process a single image.
- **Parameters and FLOPs**  
  Measures model size and computational cost.

### Training Dynamics
- **Epochs to Converge**  
  Number of epochs required to reach a target performance.
- **Loss Curves**  
  Tracks training stability and convergence behavior.

### Data Efficiency
- **mIoU at k Percent Data**  
  Performance when trained on k percent of the dataset. Evaluates low data learning ability.

All metrics are computed under identical training and evaluation settings for fair comparison.

---

## Dependencies

This project depends on the following external libraries:

- PyTorch
- Detectron2
- Mask2Former
- CLIP DINOv2 and MAE pretrained models

Dependencies are not vendored into this repository.

---

## Notes

- Mask2Former is treated as an external project and is added via PYTHONPATH
- This repository does not include copies of Detectron2 or Mask2Former code
- All encoder comparisons are conducted under identical training conditions

---

## Thesis Context

This repository supports the thesis titled:

How Much Do Vision Encoders Help Segmentation?

The codebase is designed to meet academic standards for controlled experimentation reproducibility and clarity of methodology.

