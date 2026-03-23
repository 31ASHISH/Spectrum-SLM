# Spectrum-SLM 🧠📡
> **A Small Language Model for Cognitive Radio Spectrum Sensing**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Authors:** Anjani · Ashish Joshi · Mayank | **Guide:** Dr. Abhinandan S.P. | **IIT Palakkad** | March 2026

---

## 📌 Overview

Spectrum-SLM is a GPT-like **Small Language Model (~1M parameters)** that treats RF Power Spectral Density (PSD) vectors like sequences of tokens. It is trained on **152K+ real-world measurements** from an **ADALM-Pluto SDR** at 2.4 GHz and performs **four tasks simultaneously**:

| Task | Output | Target Performance |
|------|--------|--------------------|
| PU Detection | Binary (0/1) | **96–98% accuracy** |
| Low-SNR Detection | Binary @ <8 dB | **90–94% accuracy** |
| Modulation Classification | BPSK/QPSK/8PSK/16QAM | **92–95% accuracy** |
| SNR Estimation | Regression (dB) | **MAE < 1.5 dB** |

---

## 🧠 Architecture

```
PSD Vector (176 bins)
      ↓
Patch Embedding: 176 → 22 spectral tokens (patch size = 8)
      ↓
Frequency-Aware Positional Encoding (learnable + sinusoidal blend)
      ↓
Transformer Encoder: 4 layers, 4 heads, d_model=128, d_ff=512
      ↓ [CLS token]
┌────────────────────────────────────────────────┐
│  PU Head  │  Mod Head  │  SNR Head  │  Gen Head │
│ (Binary)  │  (4-class) │ (Regress.) │ (176-dim) │
└────────────────────────────────────────────────┘
```

**~1M parameters** — edge-deployable with ONNX export.

---

## 📁 Project Structure

```
SDR_Data/
├── spectrum_slm_model.py       # PyTorch model (PatchEmbed, Transformer, all heads, losses)
├── spectrum_slm_dataset.py     # Data pipeline (PTH/CSV loaders, augmentation, DataLoaders)
├── spectrum_slm_train.py       # 3-phase training loop + evaluation + ONNX export
├── app.py                      # Streamlit interactive web demo
├── requirements.txt            # Python dependencies
│
├── Primary_User/               # GNU Radio transmitter setup (SDR hardware)
├── Secondary_User/             # SDR receiver PSD measurements
│   ├── Symbol1_Modulation/     # Main dataset (76,560 rows, Output.csv)
│   ├── Symbol2_Results/        # 44,824 rows
│   ├── Symbol3_Results/        # 31,594 rows
│   └── psd_binned_by_snr_*.pth # Full 176-bin PSD vectors per modulation
└── GeneratedDatasets_realistic/
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit web app (works without real SDR data!)
```bash
streamlit run app.py
```

### 3. Train the model (demo mode — synthetic data, no SDR files needed)
```bash
python spectrum_slm_train.py --synthetic --phase 2 --epochs_p1 10 --epochs_p2 20
```

### 4. Train on real SDR data
```bash
python spectrum_slm_train.py \
  --data_dir . \
  --phase 2 \
  --epochs_p1 30 \
  --epochs_p2 50 \
  --batch_size 64
```

### 5. Export to ONNX (edge deployment)
```bash
python spectrum_slm_train.py --synthetic --phase 2 --export_onnx
```

---

## 📊 Three-Phase Training

| Phase | Method | Loss | Data |
|-------|--------|------|------|
| **1** Pre-training | Masked Spectrum Modelling (MSM) | MSE (masked patches) | All (no labels) |
| **2** Fine-tuning | Supervised Multi-task | α·FocalLoss + β·CE + γ·MSE | Labelled |
| **3** Generative | Autoregressive Next-PSD | MSE | Sequences |

---

## 🔬 Research Novelty

1. **First SLM for Spectrum Sensing** — no prior GPT-style model on PSD data
2. **Masked Spectrum Modelling** — novel self-supervised RF pre-training
3. **Multi-task Spectrum Intelligence** — detection + classification + estimation in one model
4. **Generative PSD Forecasting** — autoregressive spectrum occupancy prediction
5. **Real Hardware Dataset** — ADALM-Pluto SDR, not simulation

**Target venues:** IEEE TCCN · IEEE DySPAN · IEEE GLOBECOM/ICC · IEEE WCL

---

## 💬 Streamlit Demo Features

- **AI Chat Tab** — Ask questions about spectrum sensing, run live scans via natural language
- **Single Scan Tab** — Upload CSV or generate synthetic PSD, see visualised predictions
- **Batch Analysis Tab** — Run 100–2000 synthetic samples, see per-SNR-bin accuracy chart
- **Research Tab** — Ablation study table and comparison vs traditional ML

---

## 📡 Hardware Setup

| Parameter | Value |
|-----------|-------|
| SDR Device | ADALM-Pluto |
| Frequency | 2.4 GHz (ISM band) |
| Sample Rate | 1.024 MHz |
| FFT Size | 1024-point (Blackman-Harris) |
| PSD Bins | 176 frequency bins per snapshot |
| Modulations | BPSK, QPSK, 8PSK, 16QAM |
| SNR Range | 3–20 dB |

---

## 📈 Expected Results vs Baseline

| Task | VotingClassifier | Spectrum-SLM | Improvement |
|------|------------------|-----------:|:-----------:|
| PU Detection | 92–95% | **96–98%** | +3–6% |
| Low-SNR (<8 dB) | 80–85% | **90–94%** | +8–12% |
| Modulation | 85–88% | **92–95%** | +7–10% |
| SNR MAE | N/A | **< 1.5 dB** | New capability |
| Generative | ❌ | ✅ | New capability |

---

## 📜 License

MIT License — see [LICENSE](LICENSE)
