# MOVIE: Interpretable Synthesis of Late-Frame [¹¹C]-PiB PET from Early-Frame Counterparts

Official implementation of the paper:

> **A PET MOVIE for Interpretable Synthesis of Late-Frame [¹¹C]-PiB PET Images from Early-Frame Counterparts**  
> *Submitted to EJNMMI Physics (under review)*

---

## Overview

MOVIE synthesizes late-frame [¹¹C]-PiB PET images directly from early-frame acquisitions, reducing patient scan time while maintaining diagnostic quality. The model uses a **LKMUNet** backbone (Mamba-based U-Net) combined with either a **Stochastic Differential Equation (SDE)** or **Ordinary Differential Equation (ODE)** module conditioned via **FiLM** (Feature-wise Linear Modulation) on intermediate PET frames.

```
Early Frame  ──►  LKMUNet Encoder  ──►  SDE/ODE Block (FiLM)  ──►  Decoder  ──►  Synthesized Late Frame
                                              ▲
                                     Intermediate Frames
```

---

## Project Structure

```
MOVIE/
├── configs/
│   ├── stratified_5fold_all.json          # 5-fold cross-validation split
│   ├── subjects_data_with_abeta.json      # Subject metadata (diagnosis, Aβ status)
│   └── subject_stats_Early2Late_withLatent_FULL_0729.csv  # Normalization statistics
├── datasets/
│   └── early2late_dataset.py             # Dataset class
├── models/
│   ├── lkmunet.py                        # LKMUNet backbone
│   ├── sde_film.py                       # SDE-FiLM model
│   └── ode_film.py                       # ODE-FiLM model
├── losses/
│   └── hybrid_loss.py                    # MSE + SSIM + LPIPS loss
├── notebooks/
│   └── MOVIE.ipynb                       # Development notebook
├── train.py                              # Training script
├── inference.py                          # Inference & evaluation script
└── requirements.txt
```

---

## Requirements

### Environment

```bash
conda create -n movie python=3.10
conda activate movie
```

### Install dependencies

> ⚠️ `mamba-ssm` requires CUDA. Please install PyTorch with the appropriate CUDA version first.
> See: https://pytorch.org/get-started/locally/

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Note:** `mamba-ssm` and `nnunetv2` can be tricky to install. If you encounter issues:
```bash
pip install mamba-ssm --no-build-isolation
pip install nnunetv2
```

---

## Data Preparation

This project uses preprocessed [¹¹C]-PiB PET data. Due to patient privacy, the raw data cannot be shared. The expected directory structure is:

```
Preprocessed_Early2Late_withLatent/
├── group0/
│   ├── data/              # Early-frame .npy slices  (shape: H × W)
│   ├── ground_truth/      # Late-frame .npy slices   (shape: H × W)
│   └── latent_target/     # Intermediate-frame .npy  (shape: H × W × M)
├── group1/
│   └── ...
└── group4/
    └── ...
```

File naming convention: `{SubjectID}_{sliceIndex}.npy`  
Example: `Subject001_042.npy`

---

## Training

```bash
python train.py \
  --model sde \
  --data-root /path/to/Preprocessed_Early2Late \
  --save-dir ./runs \
  --epochs 150 \
  --seeds 42 \
  --folds 0 1 2 3 4
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model` | `sde` | Model variant: `sde` or `ode` |
| `--data-root` | required | Path to preprocessed dataset |
| `--save-dir` | `./runs` | Output directory for checkpoints |
| `--epochs` | `150` | Number of training epochs |
| `--seeds` | `42` | Random seed(s) |
| `--folds` | `0 1 2 3 4` | Fold(s) to train |
| `--lr` | `1e-4` | Learning rate |
| `--batch-size` | `16` | Batch size |
| `--baseline-csv` | *(optional)* | CSV with baseline metrics for early stopping |

---

## Inference

```bash
python inference.py \
  --model sde \
  --ckpt-dirs \
    ./runs/LKMUSDE_F0_XXXXXXXX \
    ./runs/LKMUSDE_F1_XXXXXXXX \
    ./runs/LKMUSDE_F2_XXXXXXXX \
    ./runs/LKMUSDE_F3_XXXXXXXX \
    ./runs/LKMUSDE_F4_XXXXXXXX \
  --data-root /path/to/Preprocessed_Early2Late \
  --output-prefix results_sde
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model` | `ode` | Model variant: `sde` or `ode` |
| `--ckpt-dirs` | required | 5 checkpoint folders (F0–F4) |
| `--data-root` | required | Path to preprocessed dataset |
| `--output-prefix` | `inference` | Output CSV filename prefix |

**Outputs:**
- `{prefix}_subject_metrics.csv` — per-subject PSNR / SSIM / LPIPS / MSE
- `{prefix}_overall_metrics.csv` — mean ± std across all subjects

---

## Model Architecture

### LKMUNet
A residual U-Net where each encoder stage is augmented with bidirectional Mamba layers (`BiPixelMambaLayer` + `BiWindowMambaLayer`) for efficient long-range dependency modeling.

### SDE-FiLM / ODE-FiLM
The bottleneck feature map is evolved through a stochastic (or deterministic) differential equation. FiLM conditioning injects information from intermediate PET frames into the drift field, enabling interpretable time-dependent synthesis.

### Loss Function
```
L = MSE + (1 - SSIM)/2 + LPIPS + λ_sm · ||drift||² + λ_c · (1 - cos_sim(early, mid))
```

---

## Citation

If you use this code, please cite our paper (citation will be updated upon acceptance):

```bibtex
@article{movie2025,
  title={A PET MOVIE for Interpretable Synthesis of Late-Frame [11C]-PiB PET Images from Early-Frame Counterparts},
  journal={EJNMMI Physics},
  year={2025},
  note={Under review}
}
```

---

## Contact

For questions, please open a GitHub issue.
