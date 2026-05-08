"""
spectrum_slm_train.py
=====================
Three-phase training pipeline for Spectrum-SLM.

Phase 1 — Masked Spectrum Modelling (self-supervised pre-training)
Phase 2 — Supervised Multi-task Fine-tuning (PU + Mod + SNR)
Phase 3 — SKIPPED (no real temporal data exists)

Authors : Anjani, Ashish Joshi, Mayank
Guide   : Dr. Abhinandan S.P.
Dated   : May 2026
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, mean_absolute_error, precision_recall_curve,
    auc as sklearn_auc,
)
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

from spectrum_slm_model   import SpectrumSLM, MultiTaskLoss, MSMLoss
from spectrum_slm_dataset import N_BINS, N_PATCHES, SNR_BINS

MOD_NAMES = {0: 'BPSK', 1: 'QPSK', 2: '8PSK', 3: '16QAM', 4: 'DQPSK'}


# ─── Device / checkpoint helpers ──────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        if torch.backends.mps.is_available():
            return torch.device('mps')
    except AttributeError:
        pass
    return torch.device('cpu')


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'val_loss': val_loss}, path)
    print(f"  ✓ Checkpoint → {path}")


def load_checkpoint(model, path, optimizer=None, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    print(f"  ✓ Loaded {path}  (epoch {ckpt.get('epoch','?')})")
    return ckpt.get('epoch', 0)


# ─── Phase 1: Masked Spectrum Modelling ───────────────────────────────────────

def pretrain_msm(model: SpectrumSLM, train_loader: DataLoader,
                 val_loader: DataLoader, n_epochs=30, lr=3e-4,
                 device=None, save_dir='.', patience=5) -> List[dict]:
    """
    Phase 1: Self-supervised Masked Spectrum Modelling.
    Masks 20% of 192 bin-tokens and reconstructs them.
    Uses ONLY real data from psd_binned_by_snr_.pth.
    """
    if device is None:
        device = get_device()
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    criterion = MSMLoss()

    best_val, patience_left, history = float('inf'), patience, []

    print(f"\n{'='*60}")
    print(f"  PHASE 1 — Masked Spectrum Modelling  ({n_epochs} epochs)")
    print(f"  Device: {device}  |  LR: {lr}  |  N_PATCHES: {N_PATCHES}")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0, tr_losses = time.time(), []

        for batch in train_loader:
            psd, mask = batch               # (B,192), (B,192) bool
            psd, mask = psd.to(device), mask.to(device)

            # Ground truth: (B, 192, 1) — patch_size=1
            true_patches = psd.unsqueeze(-1)   # (B, 192, 1)

            optimizer.zero_grad()
            out  = model(psd, mask=mask, return_msm=True)
            loss = criterion(out['msm_pred'], true_patches, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        vl_losses = []
        with torch.no_grad():
            for batch in val_loader:
                psd, mask = batch
                psd, mask = psd.to(device), mask.to(device)
                true_patches = psd.unsqueeze(-1)
                out  = model(psd, mask=mask, return_msm=True)
                loss = criterion(out['msm_pred'], true_patches, mask)
                vl_losses.append(loss.item())

        scheduler.step()
        tr_l = np.mean(tr_losses)
        vl_l = np.mean(vl_losses)
        history.append({'epoch': epoch, 'train_msm': tr_l, 'val_msm': vl_l})
        print(f"  Ep {epoch:3d}/{n_epochs}  Train:{tr_l:.4f}  Val:{vl_l:.4f}  "
              f"({time.time()-t0:.1f}s)")

        if vl_l < best_val:
            best_val = vl_l
            patience_left = patience
            save_checkpoint(model, optimizer, epoch, vl_l,
                            os.path.join(save_dir, 'slm_phase1_best.pt'))
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stop at epoch {epoch}")
                break

    print(f"\n  ✓ Phase 1 done. Best Val MSM: {best_val:.4f}")
    return history


# ─── Phase 2: Supervised Multi-task Fine-tuning ───────────────────────────────

def finetune_supervised(model: SpectrumSLM, train_loader: DataLoader,
                        val_loader: DataLoader,
                        pu_class_weight: Optional[torch.Tensor] = None,
                        n_epochs=50, lr=1e-4, device=None, save_dir='.',
                        patience=8, alpha=1.0, beta=0.5, gamma=0.3,
                        learn_weights=True) -> List[dict]:
    """
    Phase 2: Supervised Multi-task Fine-tuning.
    Loss = α·Focal(PU) + β·CE(Mod) + γ·Huber(SNR)
    """
    if device is None:
        device = get_device()
    model = model.to(device)

    if pu_class_weight is not None:
        pu_class_weight = pu_class_weight.to(device)

    criterion = MultiTaskLoss(alpha=alpha, beta=beta, gamma=gamma,
                              pu_class_weight=pu_class_weight,
                              focal_gamma=2.0, learn_weights=learn_weights).to(device)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=lr*10, epochs=n_epochs,
                           steps_per_epoch=len(train_loader),
                           pct_start=0.1, anneal_strategy='cos')

    best_val, patience_left, history = float('inf'), patience, []
    start_epoch = 0

    # Auto-resume
    ckpt_path = os.path.join(save_dir, 'slm_phase2_best.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        best_val    = ckpt.get('val_loss', float('inf'))
        start_epoch = ckpt.get('epoch', 0)
        print(f"  [RESUME] From epoch {start_epoch}, val_loss={best_val:.4f}")

    print(f"\n{'='*60}")
    print(f"  PHASE 2 — Supervised Multi-task  ({n_epochs} epochs)")
    print(f"  Device:{device}  LR:{lr}  learn_weights:{learn_weights}")
    print(f"{'='*60}")

    for epoch in range(start_epoch + 1, n_epochs + 1):
        model.train()
        t0 = time.time()
        tr_tot, tr_pu, tr_mod, tr_snr = [], [], [], []

        for psd, pu_lab, mod_lab, snr_lab in train_loader:
            psd, pu_lab = psd.to(device), pu_lab.to(device)
            mod_lab, snr_lab = mod_lab.to(device), snr_lab.to(device)

            valid = mod_lab >= 0
            if valid.sum() == 0:
                continue

            optimizer.zero_grad()
            out = model(psd)
            total, bd = criterion(
                out['pu_logits'][valid],  pu_lab[valid],
                out['mod_logits'][valid], mod_lab[valid],
                out['snr_pred'][valid],   snr_lab[valid])
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_tot.append(bd['total']); tr_pu.append(bd['pu'])
            tr_mod.append(bd['mod']);   tr_snr.append(bd['snr'])

        model.eval()
        vl_tot = []
        with torch.no_grad():
            for psd, pu_lab, mod_lab, snr_lab in val_loader:
                psd, pu_lab = psd.to(device), pu_lab.to(device)
                mod_lab, snr_lab = mod_lab.to(device), snr_lab.to(device)
                valid = mod_lab >= 0
                if valid.sum() == 0:
                    continue
                out = model(psd)
                loss, _ = criterion(
                    out['pu_logits'][valid],  pu_lab[valid],
                    out['mod_logits'][valid], mod_lab[valid],
                    out['snr_pred'][valid],   snr_lab[valid])
                vl_tot.append(loss.item())

        vl_mean = np.mean(vl_tot) if vl_tot else float('inf')
        history.append({
            'epoch': epoch, 'train_total': np.mean(tr_tot),
            'train_pu': np.mean(tr_pu), 'train_mod': np.mean(tr_mod),
            'train_snr': np.mean(tr_snr), 'val_total': vl_mean,
        })
        print(f"  Ep {epoch:3d}/{n_epochs}  "
              f"Train:{np.mean(tr_tot):.4f} "
              f"(PU={np.mean(tr_pu):.3f} Mod={np.mean(tr_mod):.3f} "
              f"SNR={np.mean(tr_snr):.3f})  Val:{vl_mean:.4f}  "
              f"({time.time()-t0:.1f}s)")

        if vl_mean < best_val:
            best_val = vl_mean
            patience_left = patience
            save_checkpoint(model, optimizer, epoch, vl_mean,
                            os.path.join(save_dir, 'slm_phase2_best.pt'))
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stop at epoch {epoch}")
                break

    print(f"\n  ✓ Phase 2 done. Best Val: {best_val:.4f}")
    return history


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model: SpectrumSLM, test_loader: DataLoader,
                   device=None, snr_bins=None) -> dict:
    """
    Full evaluation: accuracy, F1, AUC, PR-AUC, MAE + per-SNR-bin breakdown.
    Operates ONLY on real test split — no synthetic data.
    """
    if device is None:
        device = get_device()
    if snr_bins is None:
        snr_bins = SNR_BINS
    model.eval().to(device)

    pu_pred_l, pu_true_l  = [], []
    pu_prob_l             = []
    mod_pred_l, mod_true_l = [], []
    snr_pred_l, snr_true_l = [], []

    with torch.no_grad():
        for psd, pu_lab, mod_lab, snr_lab in test_loader:
            out      = model(psd.to(device))
            pu_prob  = torch.softmax(out['pu_logits'], dim=1)[:, 1].cpu().numpy()
            pu_pred  = (pu_prob > 0.5).astype(int)
            mod_pred = out['mod_logits'].argmax(1).cpu().numpy()
            snr_pred = out['snr_pred'].cpu().numpy()

            pu_pred_l.extend(pu_pred);    pu_true_l.extend(pu_lab.numpy())
            pu_prob_l.extend(pu_prob)
            mod_pred_l.extend(mod_pred);  mod_true_l.extend(mod_lab.numpy())
            snr_pred_l.extend(snr_pred);  snr_true_l.extend(snr_lab.numpy())

    pu_pred_a  = np.array(pu_pred_l)
    pu_true_a  = np.array(pu_true_l)
    pu_prob_a  = np.array(pu_prob_l)
    mod_pred_a = np.array(mod_pred_l)
    mod_true_a = np.array(mod_true_l)
    snr_pred_a = np.array(snr_pred_l)
    snr_true_a = np.array(snr_true_l)

    # PU metrics
    pu_acc  = accuracy_score(pu_true_a, pu_pred_a)
    pu_f1   = f1_score(pu_true_a, pu_pred_a, average='binary', zero_division=0)
    pu_prec = f1_score(pu_true_a, pu_pred_a, average='binary',
                       zero_division=0, pos_label=1)
    try:
        pu_auc = roc_auc_score(pu_true_a, pu_prob_a)
    except ValueError:
        pu_auc = float('nan')
    try:
        prec_c, rec_c, _ = precision_recall_curve(pu_true_a, pu_prob_a)
        pu_pr_auc = sklearn_auc(rec_c, prec_c)
    except Exception:
        pu_pr_auc = float('nan')
    pu_cm = confusion_matrix(pu_true_a, pu_pred_a).tolist()

    # Low-SNR PU (<8 dB)
    low_mask = snr_true_a < 8.0
    if low_mask.sum() > 0:
        low_acc = accuracy_score(pu_true_a[low_mask], pu_pred_a[low_mask])
        low_f1  = f1_score(pu_true_a[low_mask], pu_pred_a[low_mask],
                           average='binary', zero_division=0)
    else:
        low_acc = low_f1 = float('nan')

    # Modulation metrics
    valid_mod = mod_true_a >= 0
    if valid_mod.sum() > 0:
        mod_acc = accuracy_score(mod_true_a[valid_mod], mod_pred_a[valid_mod])
        mod_f1  = f1_score(mod_true_a[valid_mod], mod_pred_a[valid_mod],
                           average='macro', zero_division=0)
        mod_report = classification_report(
            mod_true_a[valid_mod], mod_pred_a[valid_mod],
            target_names=['BPSK','QPSK','8PSK','16QAM','DQPSK'],
            output_dict=True, zero_division=0)
    else:
        mod_acc = mod_f1 = float('nan')
        mod_report = {}

    # SNR metrics
    snr_mae  = float(mean_absolute_error(snr_true_a, snr_pred_a))
    snr_rmse = float(np.sqrt(np.mean((snr_pred_a - snr_true_a)**2)))
    ss_res   = np.sum((snr_pred_a - snr_true_a)**2)
    ss_tot   = np.sum((snr_true_a - snr_true_a.mean())**2)
    snr_r2   = float(1 - ss_res / (ss_tot + 1e-8))

    # Per-SNR-bin metrics
    per_snr = {}
    for sbin in snr_bins:
        m = (snr_true_a >= sbin - 1) & (snr_true_a < sbin + 1)
        if m.sum() > 5:
            per_snr[str(sbin)] = {
                'pu_acc': float(accuracy_score(pu_true_a[m], pu_pred_a[m])),
                'pu_f1' : float(f1_score(pu_true_a[m], pu_pred_a[m],
                                         average='binary', zero_division=0)),
                'snr_mae': float(mean_absolute_error(snr_true_a[m], snr_pred_a[m])),
                'n'      : int(m.sum()),
            }

    metrics = {
        'pu_accuracy'    : float(pu_acc),
        'pu_f1'          : float(pu_f1),
        'pu_auc'         : float(pu_auc),
        'pu_pr_auc'      : float(pu_pr_auc),
        'pu_confusion'   : pu_cm,
        'low_snr_pu_acc' : float(low_acc),
        'low_snr_pu_f1'  : float(low_f1),
        'mod_accuracy'   : float(mod_acc),
        'mod_f1_macro'   : float(mod_f1),
        'mod_report'     : mod_report,
        'snr_mae_db'     : snr_mae,
        'snr_rmse_db'    : snr_rmse,
        'snr_r2'         : snr_r2,
        'per_snr_metrics': per_snr,
        'n_samples'      : len(pu_true_a),
    }

    print(f"\n{'='*55}")
    print("  EVALUATION RESULTS (Real Test Set)")
    print(f"{'='*55}")
    print(f"  PU   Acc:{pu_acc*100:.2f}%  F1:{pu_f1:.4f}  AUC:{pu_auc:.4f}  PR-AUC:{pu_pr_auc:.4f}")
    print(f"  LowSNR(<8dB)  Acc:{low_acc*100:.2f}%  F1:{low_f1:.4f}")
    print(f"  Mod  Acc:{mod_acc*100:.2f}%  macro-F1:{mod_f1:.4f}")
    print(f"  SNR  MAE:{snr_mae:.3f}dB  RMSE:{snr_rmse:.3f}dB  R²:{snr_r2:.4f}")
    print(f"{'='*55}")
    return metrics


# ─── Single-sample inference ──────────────────────────────────────────────────

def predict_single(model: SpectrumSLM, psd_vector: np.ndarray,
                   normalizer, device=None) -> dict:
    if device is None:
        device = get_device()
    model.eval().to(device)

    psd_n = normalizer.transform(psd_vector.reshape(1, -1))
    t     = torch.tensor(psd_n, dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(t)

    pu_prob  = torch.softmax(out['pu_logits'], dim=1)[0, 1].item()
    mod_prob = torch.softmax(out['mod_logits'], dim=1)[0].cpu().numpy()
    mod_pred = int(np.argmax(mod_prob))
    snr_pred = float(out['snr_pred'][0].item())

    return {
        'pu_present'      : bool(pu_prob > 0.5),
        'pu_confidence'   : float(pu_prob),
        'modulation'      : MOD_NAMES.get(mod_pred, 'Unknown'),
        'mod_confidence'  : float(mod_prob[mod_pred]),
        'mod_probabilities': {MOD_NAMES[i]: float(mod_prob[i])
                              for i in range(len(mod_prob))},
        'snr_estimated_db': snr_pred,
    }
