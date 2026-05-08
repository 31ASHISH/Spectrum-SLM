"""
Generates spectrum_slm_kaggle.ipynb programmatically.
This guarantees 100% valid JSON that Kaggle can import.
"""
import json

def cell(source_lines, cell_type="code"):
    src = "\n".join(source_lines)
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": src}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": src,
    }

cells = [

cell([
    "# Spectrum-SLM — Cognitive Radio Spectrum Sensing",
    "### 192-bin Transformer | 5-class Modulation | PU Detection | SNR Estimation",
    "",
    "**Authors:** Anjani, Ashish Joshi, Mayank | **Guide:** Dr. Abhinandan S.P.",
    "",
    "**Before running:**",
    "1. Settings → Accelerator → **GPU T4 x2**",
    "2. Settings → Internet → **On**",
    "3. Settings → Persistence → **Files**",
    "4. Add your SDR dataset via **Add Input** on the right",
], "markdown"),

cell([
    "# Cell 1 — Clone code from GitHub",
    "import os",
    "os.makedirs('/kaggle/working', exist_ok=True)",
    "if not os.path.isdir('/kaggle/working/Spectrum-SLM'):",
    "    import subprocess",
    "    result = subprocess.run(",
    "        ['git', 'clone', 'https://github.com/31ASHISH/Spectrum-SLM.git',",
    "         '/kaggle/working/Spectrum-SLM'],",
    "        capture_output=True, text=True",
    "    )",
    "    print(result.stdout)",
    "    print(result.stderr)",
    "else:",
    "    print('Repo already cloned.')",
    "print('Done.')",
]),

cell([
    "# Cell 2 — Change into project directory + add to Python path",
    "import sys, os",
    "REPO = '/kaggle/working/Spectrum-SLM'",
    "os.chdir(REPO)",
    "sys.path.insert(0, REPO)",
    "print('CWD:', os.getcwd())",
    "print('Contents:', os.listdir('.'))",
]),

cell([
    "# Cell 3 — Install missing packages",
    "import subprocess",
    "subprocess.run(['pip', 'install', '-q', 'scikit-learn', 'pandas', 'tqdm'],",
    "               capture_output=True)",
    "import torch",
    "print(f'PyTorch : {torch.__version__}')",
    "print(f'CUDA    : {torch.cuda.is_available()}')",
    "if torch.cuda.is_available():",
    "    print(f'GPU     : {torch.cuda.get_device_name(0)}')",
]),

cell([
    "# Cell 4 — Configure Kaggle paths",
    "# EDIT DATA_ROOT below if your dataset folder name differs",
    "import os",
    "",
    "# List all datasets available",
    "print('Available datasets in /kaggle/input/:')",
    "if os.path.isdir('/kaggle/input'):",
    "    for d in os.listdir('/kaggle/input'):",
    "        print(f'  /kaggle/input/{d}/')",
    "",
    "# ---- EDIT THIS LINE if your dataset name is different ----",
    "DATA_ROOT = '/kaggle/input/spectrum-slm-sdr-data'",
    "# ----------------------------------------------------------",
    "",
    "SU_DIR  = f'{DATA_ROOT}/Secondary_User'",
    "NEW_DIR = f'{DATA_ROOT}/files-20260414T094743Z-3-001'",
    "OUT_DIR = '/kaggle/working/checkpoints'",
    "os.makedirs(f'{OUT_DIR}/phase1', exist_ok=True)",
    "os.makedirs(f'{OUT_DIR}/phase2', exist_ok=True)",
    "",
    "print()",
    "print('=== Path Check ===')",
    "for name, path in [('SU_DIR', SU_DIR), ('NEW_DIR', NEW_DIR)]:",
    "    ok = os.path.isdir(path)",
    "    print(f'  {name}: {path}  [{\"OK\" if ok else \"NOT FOUND\"}]')",
]),

cell([
    "# Cell 5 — Override config for Kaggle",
    "import config",
    "config.SECONDARY_USER_DIR = SU_DIR",
    "config.NEW_DATASET_DIR    = NEW_DIR",
    "config.PHASE1_DATA_FILE   = f'{SU_DIR}/psd_binned_by_snr_.pth'",
    "config.CKPT_ROOT          = OUT_DIR",
    "config.CKPT_PHASE1        = f'{OUT_DIR}/phase1'",
    "config.CKPT_PHASE2        = f'{OUT_DIR}/phase2'",
    "config.PHASE1_DATA_DIR    = SU_DIR",
    "config.PHASE2_DATA_DIR    = NEW_DIR",
    "print(f'N_BINS={config.N_BINS}  SEQ_LEN={config.SEQ_LEN}')",
    "print(f'Modulations: {config.MOD_NAMES_V2}')",
]),

cell([
    "# Cell 6 — Verify dataset",
    "import numpy as np",
    "from dataset.loader import load_all_real_data",
    "",
    "psds, pu, mod, snr = load_all_real_data(SU_DIR, NEW_DIR)",
    "names = {0:'BPSK', 1:'QPSK', 2:'8PSK', 3:'16QAM', 4:'DQPSK'}",
    "print(f'Total: {len(psds):,}  shape={psds.shape}  NaN={np.isnan(psds).sum()}')",
    "print(f'PU=1: {pu.sum():,} ({pu.mean()*100:.1f}%)')",
    "print(f'SNR : {snr.min():.1f} - {snr.max():.1f} dB')",
    "for m in range(5):",
    "    print(f'  {names[m]}: {(mod==m).sum():,}')",
    "assert psds.shape[1] == 192",
    "print('Dataset: OK')",
]),

cell([
    "# Cell 7 — Model sanity check",
    "import torch",
    "from spectrum_slm_model import SpectrumSLM",
    "",
    "model = SpectrumSLM(n_bins=192, patch_size=1, d_model=128,",
    "                    nhead=4, num_layers=4, dim_feedforward=512,",
    "                    dropout=0.1, n_mod_classes=5)",
    "print(f'Parameters: {model.count_parameters():,}')",
    "psd_t = torch.randn(4, 192)",
    "out   = model(psd_t, return_msm=True)",
    "for k, v in out.items():",
    "    print(f'  {k}: {v.shape}')",
    "assert out['pu_logits'].shape  == (4, 2)",
    "assert out['mod_logits'].shape == (4, 5)",
    "assert out['snr_pred'].shape   == (4,)",
    "print('Model: OK')",
]),

cell([
    "# Cell 8 — Generate dataset reports",
    "from dataset.analysis import run_analysis",
    "run_analysis(out_dir='/kaggle/working')",
    "print('Reports saved to /kaggle/working/')",
]),

cell([
    "# ================================================================",
    "# Cell 9 — PHASE 1: Masked Spectrum Modelling (~25 min on T4)",
    "# ================================================================",
    "from training.train_phase1 import run_phase1",
    "",
    "history_p1 = run_phase1(",
    "    data_dir   = SU_DIR,",
    "    save_dir   = f'{OUT_DIR}/phase1',",
    "    epochs     = 30,",
    "    lr         = 3e-4,",
    "    batch_size = 64,",
    "    patience   = 5,",
    "    resume     = True,",
    ")",
    "",
    "if history_p1:",
    "    print(f'Best Val MSM: {min(h[\"val_msm\"] for h in history_p1):.4f}')",
]),

cell([
    "# Cell 10 — Phase 1 training curve",
    "import matplotlib.pyplot as plt",
    "",
    "if history_p1:",
    "    ep = [h['epoch']     for h in history_p1]",
    "    tr = [h['train_msm'] for h in history_p1]",
    "    vl = [h['val_msm']   for h in history_p1]",
    "    plt.figure(figsize=(9,4))",
    "    plt.plot(ep, tr, label='Train', color='#58a6ff', lw=2)",
    "    plt.plot(ep, vl, label='Val',   color='#f78166', lw=2, ls='--')",
    "    plt.xlabel('Epoch'); plt.ylabel('MSM Loss')",
    "    plt.title('Phase 1 — Masked Spectrum Modelling')",
    "    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()",
    "    plt.savefig('/kaggle/working/phase1_curve.png', dpi=150); plt.show()",
]),

cell([
    "# ================================================================",
    "# Cell 11 — PHASE 2: Supervised Multi-task Fine-tuning (~60 min)",
    "# ================================================================",
    "from training.train_phase2 import run_phase2",
    "",
    "metrics = run_phase2(",
    "    secondary_user_dir = SU_DIR,",
    "    new_dataset_dir    = NEW_DIR,",
    "    save_dir           = f'{OUT_DIR}/phase2',",
    "    epochs             = 50,",
    "    batch_size         = 64,",
    "    lr                 = 1e-4,",
    "    patience           = 8,",
    "    learn_weights      = True,",
    "    resume             = True,",
    "    resume_phase1      = True,",
    ")",
]),

cell([
    "# Cell 12 — Final results",
    "print('=' * 58)",
    "print('  SPECTRUM-SLM FINAL RESULTS')",
    "print('=' * 58)",
    "print(f'  PU  Accuracy : {metrics[\"pu_accuracy\"]*100:.2f}%')",
    "print(f'  PU  F1       : {metrics[\"pu_f1\"]:.4f}')",
    "print(f'  PU  AUC      : {metrics[\"pu_auc\"]:.4f}')",
    "print(f'  PU  PR-AUC   : {metrics[\"pu_pr_auc\"]:.4f}')",
    "print(f'  LowSNR Acc   : {metrics[\"low_snr_pu_acc\"]*100:.2f}%  (<8dB)')",
    "print(f'  Mod Accuracy : {metrics[\"mod_accuracy\"]*100:.2f}%')",
    "print(f'  Mod macro-F1 : {metrics[\"mod_f1_macro\"]:.4f}')",
    "print(f'  SNR MAE      : {metrics[\"snr_mae_db\"]:.3f} dB')",
    "print(f'  SNR RMSE     : {metrics[\"snr_rmse_db\"]:.3f} dB')",
    "print(f'  SNR R2       : {metrics[\"snr_r2\"]:.4f}')",
    "print('=' * 58)",
    "print('  Per-SNR Breakdown:')",
    "for b, m in sorted(metrics['per_snr_metrics'].items(), key=lambda x: int(x[0])):",
    "    print(f'  {b:>3}dB  PU={m[\"pu_acc\"]*100:.1f}%  F1={m[\"pu_f1\"]:.3f}  SNR_MAE={m[\"snr_mae\"]:.2f}dB  n={m[\"n\"]}')",
]),

cell([
    "# Cell 13 — Training curves (Phase 2)",
    "import json, matplotlib.pyplot as plt",
    "hist_path = f'{OUT_DIR}/phase2/training_history_phase2.json'",
    "if os.path.exists(hist_path):",
    "    with open(hist_path) as f: h2 = json.load(f)",
    "    ep = [h['epoch']       for h in h2]",
    "    fig, ax = plt.subplots(1, 2, figsize=(13, 4))",
    "    ax[0].plot(ep, [h['train_total'] for h in h2], label='Train', color='#58a6ff', lw=2)",
    "    ax[0].plot(ep, [h['val_total']   for h in h2], label='Val',   color='#f78166', lw=2, ls='--')",
    "    ax[0].set_title('Total Loss'); ax[0].legend(); ax[0].grid(alpha=0.3)",
    "    ax[1].plot(ep, [h['train_pu']  for h in h2], label='PU',  color='#3fb950', lw=2)",
    "    ax[1].plot(ep, [h['train_mod'] for h in h2], label='Mod', color='#ffa657', lw=2)",
    "    ax[1].plot(ep, [h['train_snr'] for h in h2], label='SNR', color='#d2a8ff', lw=2)",
    "    ax[1].set_title('Per-Task Train Loss'); ax[1].legend(); ax[1].grid(alpha=0.3)",
    "    plt.tight_layout()",
    "    plt.savefig('/kaggle/working/phase2_curves.png', dpi=150); plt.show()",
]),

cell([
    "# Cell 14 — Confusion matrix + per-SNR bar chart",
    "import numpy as np, matplotlib.pyplot as plt",
    "cm = np.array(metrics['pu_confusion'])",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))",
    "",
    "# Confusion matrix",
    "axes[0].imshow(cm, cmap='Blues')",
    "for i in range(2):",
    "    for j in range(2):",
    "        axes[0].text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=14,",
    "                     color='white' if cm[i,j] > cm.max()/2 else 'black')",
    "axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])",
    "axes[0].set_xticklabels(['Pred:Idle','Pred:Active'])",
    "axes[0].set_yticklabels(['True:Idle','True:Active'])",
    "axes[0].set_title('PU Confusion Matrix')",
    "",
    "# Per-SNR accuracy",
    "snr_bins = sorted(metrics['per_snr_metrics'].keys(), key=int)",
    "accs = [metrics['per_snr_metrics'][b]['pu_acc']*100 for b in snr_bins]",
    "axes[1].bar(snr_bins, accs, color='#58a6ff')",
    "axes[1].axhline(90, color='red', ls='--', label='90% target')",
    "axes[1].set_xlabel('SNR (dB)'); axes[1].set_ylabel('PU Accuracy (%)')",
    "axes[1].set_title('PU Accuracy vs SNR'); axes[1].legend()",
    "",
    "plt.tight_layout()",
    "plt.savefig('/kaggle/working/results.png', dpi=150); plt.show()",
]),

cell([
    "# Cell 15 — Export ONNX",
    "from training.export_onnx import export_onnx",
    "export_onnx(",
    "    ckpt_path = f'{OUT_DIR}/phase2/slm_phase2_best.pt',",
    "    save_path = '/kaggle/working/spectrum_slm.onnx',",
    ")",
]),

cell([
    "# Cell 16 — List all output files (download from Output tab)",
    "import os",
    "print('Files ready to download:')",
    "for root, dirs, files in os.walk('/kaggle/working'):",
    "    dirs[:] = [d for d in dirs if not d.startswith('.')]",
    "    for f in sorted(files):",
    "        fpath = os.path.join(root, f)",
    "        size  = os.path.getsize(fpath) / 1e6",
    "        print(f'  {fpath.replace(\"/kaggle/working/\",\"\"):<50} {size:>7.2f} MB')",
]),

]

notebook = {
    "metadata": {
        "kernelspec": {
            "language": "python",
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.11",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py"
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True
        }
    },
    "nbformat_minor": 4,
    "nbformat": 4,
    "cells": cells
}

out_path = "spectrum_slm_kaggle.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=True)

# Validate it
with open(out_path, "r", encoding="utf-8") as f:
    loaded = json.load(f)

print(f"Generated: {out_path}")
print(f"Cells    : {len(loaded['cells'])}")
print(f"File size: {os.path.getsize(out_path)/1e3:.1f} KB")
print("JSON validation: PASSED")
