# 📦 Model Checkpoints

The trained model checkpoint files (`.pt`) are **not stored in this repository** due to GitHub's 100MB file size limit. They are available for download from Kaggle.

## Download Instructions

### Option 1 — Run the Kaggle Notebook (Recommended)
1. Open [`spectrum_slm_kaggle.ipynb`](../spectrum_slm_kaggle.ipynb) in Kaggle
2. Enable **GPU accelerator** (T4 or P100)
3. Run all cells — checkpoints are auto-saved to `/kaggle/working/checkpoints/`
4. Download the output folder or push to this repo via PAT

### Option 2 — Manual Download from Kaggle Output
After running the notebook, go to **Output → Download** on the Kaggle run page.

## Expected Checkpoint Files

| File | Phase | Description | Size |
|------|-------|-------------|------|
| `phase1/slm_phase1_best.pt` | 1 | MSM pre-trained backbone | ~3.6 MB |
| `phase2/slm_phase2_best.pt` | 2 | Multi-task model (93.32% PU accuracy) | ~3.6 MB |
| `phase2/normalizer.pkl` | 2 | Fitted StandardScaler for inference | ~1 KB |
| `phase3/slm_phase3_best.pt` | 3 | Generative head (Val MSE 0.1054) | ~3.7 MB |

## Checkpoint Schema

```python
# Phase 2 / Phase 3 checkpoint structure:
{
    "model":     OrderedDict,  # model.state_dict()
    "epoch":     int,          # best epoch number
    "val_loss":  float,        # validation loss at best epoch
    "phase":     int,          # 1, 2, or 3
}
```

## Place Files At

```
checkpoints/
├── phase1/
│   └── slm_phase1_best.pt
├── phase2/
│   ├── slm_phase2_best.pt
│   └── normalizer.pkl
└── phase3/
    └── slm_phase3_best.pt
```

Then run `streamlit run app_phase2.py` — the app auto-detects all checkpoints.
