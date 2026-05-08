# Spectrum-SLM — Cognitive Radio Spectrum Sensing

> A production-grade Transformer-based Spectrum Language Model (SLM) for real-time cognitive radio spectrum sensing using Software-Defined Radio (SDR) data.

**Authors:** Anjani · Ashish Joshi · Mayank  
**Guide:** Dr. Abhinandan S.P.  
**Date:** May 2026

---

## Architecture

```
PSD (192,) → PatchEmbedding (patch_size=1) → 192 Tokens
+ CLS Token → 193 Tokens
→ FrequencyAwarePositionalEncoding (learned + sinusoidal blend)
→ TransformerEncoder (4 layers × 4 heads × d=128, Pre-LN)
→ CLS Feature (128-d) → Multi-task Heads:
    PU Detection  : 128→64→2       (Binary: PU present/absent)
    Modulation    : 128→64→5       (BPSK / QPSK / 8PSK / 16QAM / DQPSK)
    SNR Estimation: 128→64→1       (Regression, dB)
```

- **Parameters:** ~943K  
- **Input:** 192 frequency bins (real SDR PSD vectors)  
- **Token sequence:** 193 (192 bin-tokens + 1 CLS)

---

## Dataset

Real SDR recordings. **No synthetic data used for training.**

| Modulation | Samples |
|-----------|---------|
| BPSK      | 45,990  |
| QPSK      | 34,680  |
| 8PSK      | 13,745  |
| 16QAM     | 23,090  |
| DQPSK     | 14,735  |
| **Total** | **132,240** |

Data is loaded from two sources:
- `Secondary_User/` — binned format `.pth` files
- `files-20260414T094743Z-3-001/Symbol2/, Symbol3/` — log format `.pth` files

> ⚠️ Data files are NOT included in this repo (too large for GitHub).  
> Download them separately and place in the correct directories.

---

## Project Structure

```
SDR_Data/
├── config.py                    # Central config (N_BINS=192, all paths)
├── spectrum_slm_model.py        # Transformer model + loss functions
├── spectrum_slm_dataset.py      # Dataset pipeline (normalize, split, augment)
├── spectrum_slm_train.py        # Phase 1 & 2 training + evaluation
├── app_phase2.py                # Streamlit dashboard
├── verify_all.py                # End-to-end pipeline verification
├── dataset_report.json          # Auto-generated dataset statistics
├── dataset_statistics.csv       # Per-modulation statistics
├── dataset_structure.txt        # Dataset structure summary
├── dataset/
│   ├── loader.py                # Unified PTH loader (binned + log formats)
│   └── analysis.py              # Dataset analysis script
└── training/
    ├── train_phase1.py          # Phase 1: Masked Spectrum Modelling
    ├── train_phase2.py          # Phase 2: Supervised Multi-task
    ├── run_3_phases.py          # Run all phases in sequence
    └── export_onnx.py           # ONNX export + benchmark
```

---

## Three-Phase Training

| Phase | Method | Purpose |
|-------|--------|---------|
| **Phase 1** | Masked Spectrum Modelling (MSM) | Self-supervised pre-training on real spectra |
| **Phase 2** | Supervised Multi-task | PU + Modulation + SNR joint learning |
| **Phase 3** | *Skipped* | No real temporal sequence data exists |

### Loss Functions
- **PU:** Focal Loss (γ=2) + Kendall uncertainty weighting  
- **Mod:** CrossEntropy  
- **SNR:** HuberLoss (robust to outliers)

---

## Quick Start

### Install
```bash
pip install torch numpy pandas scikit-learn streamlit plotly scipy tqdm
```

### Verify Pipeline
```bash
python verify_all.py
```

### Run Training
```bash
# Phase 1 (MSM pre-training)
python training/train_phase1.py --epochs 30

# Phase 2 (supervised fine-tuning)
python training/train_phase2.py --epochs 50

# Or run both in sequence
python training/run_3_phases.py
```

### Export ONNX
```bash
python training/export_onnx.py
```

### Run Dashboard
```bash
streamlit run app_phase2.py
```

---

## Running on Kaggle

1. Upload `Secondary_User/` and `files-20260414T094743Z-3-001/` as a Kaggle Dataset
2. Upload all `.py` files as a second Kaggle Dataset
3. Create a GPU notebook and follow the guide in `KAGGLE_GUIDE.md`

**Estimated GPU time:** ~85 minutes (T4) · ~58 minutes (P100)

---

## Evaluation Metrics

- **PU Detection:** Accuracy, F1, ROC-AUC, PR-AUC, per-SNR-bin breakdown  
- **Modulation:** Accuracy, macro-F1, per-class F1  
- **SNR:** MAE (dB), RMSE, R²  
- **Low-SNR Robustness:** Accuracy and F1 for SNR < 8 dB

---

## Deployment

The model exports to **ONNX** for edge deployment on SDR hardware:
```python
from training.export_onnx import export_onnx
export_onnx("checkpoints/phase2/slm_phase2_best.pt", "spectrum_slm.onnx")
```
Typical latency: **< 5ms per sample** on CPU.
