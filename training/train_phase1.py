"""
training/train_phase1.py
========================
Standalone Phase 1 trainer — Masked Spectrum Modelling.

Uses ONLY psd_binned_by_snr_.pth (real mixed data, no synthetic).

CLI:
    python training/train_phase1.py
    python training/train_phase1.py --epochs 20 --lr 3e-4

Authors : Anjani, Ashish Joshi, Mayank
Guide   : Dr. Abhinandan S.P.
Dated   : May 2026
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import (
    SECONDARY_USER_DIR, CKPT_PHASE1,
    N_BINS, D_MODEL, N_HEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
    PHASE1_LR, PHASE1_EPOCHS, PHASE1_BATCH_SIZE, PHASE1_PATIENCE,
    PHASE1_MASK_RATIO, N_MOD_CLASSES_V2, ensure_dirs,
)
from dataset.loader import load_secondary_user
from spectrum_slm_dataset import SpectrumNormalizer, SpectrumDataset, SpectrumAugmenter
from spectrum_slm_model   import SpectrumSLM
from spectrum_slm_train   import pretrain_msm, get_device, load_checkpoint


def run_phase1(
    data_dir:   str   = SECONDARY_USER_DIR,
    save_dir:   str   = CKPT_PHASE1,
    epochs:     int   = PHASE1_EPOCHS,
    lr:         float = PHASE1_LR,
    batch_size: int   = PHASE1_BATCH_SIZE,
    patience:   int   = PHASE1_PATIENCE,
    mask_ratio: float = PHASE1_MASK_RATIO,
    resume:     bool  = True,
):
    ensure_dirs()
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Spectrum-SLM — Phase 1: Masked Spectrum Modelling")
    print(f"  Data dir : {data_dir}")
    print(f"  Save dir : {save_dir}")
    print(f"  Device   : {device}")
    print(f"{'='*60}")

    # Load Phase 1 data (psd_binned_by_snr_.pth only)
    print("\n[PHASE1] Loading real SDR data (psd_binned_by_snr_.pth)...")
    psds, pu, mod, snr = load_secondary_user(data_dir, for_phase1=True)

    if len(psds) == 0:
        raise RuntimeError(f"No Phase 1 data found in {data_dir}")

    # Normalize — fit on train split only
    idx = np.arange(len(psds))
    idx_train, idx_val = train_test_split(idx, test_size=0.15,
                                          stratify=pu, random_state=42)
    normalizer = SpectrumNormalizer()
    psds_train = normalizer.fit_transform(psds[idx_train])
    psds_val   = normalizer.transform(psds[idx_val])

    augmenter = SpectrumAugmenter(noise_std=0.01, max_shift=2,
                                  scale_range=(0.98, 1.02))

    train_ds = SpectrumDataset(psds_train, pu[idx_train], mod[idx_train],
                               snr[idx_train], phase=1, mask_ratio=mask_ratio,
                               augmenter=augmenter, training=True)
    val_ds   = SpectrumDataset(psds_val,   pu[idx_val],   mod[idx_val],
                               snr[idx_val],   phase=1, mask_ratio=mask_ratio,
                               augmenter=None,    training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    # Model
    model = SpectrumSLM(n_bins=N_BINS, patch_size=1, d_model=D_MODEL,
                        nhead=N_HEAD, num_layers=NUM_LAYERS,
                        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
                        n_mod_classes=N_MOD_CLASSES_V2)
    print(f"  Parameters: {model.count_parameters():,}")

    # Resume
    ckpt_path = os.path.join(save_dir, 'slm_phase1_best.pt')
    if resume and os.path.exists(ckpt_path):
        load_checkpoint(model, ckpt_path, device=device)

    # Train
    history = pretrain_msm(model, train_loader, val_loader,
                           n_epochs=epochs, lr=lr, device=device,
                           save_dir=save_dir, patience=patience)

    # Save history
    hist_path = os.path.join(save_dir, 'phase1_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History → {hist_path}")
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spectrum-SLM Phase 1 Trainer')
    parser.add_argument('--data-dir',   default=SECONDARY_USER_DIR)
    parser.add_argument('--save-dir',   default=CKPT_PHASE1)
    parser.add_argument('--epochs',     type=int,   default=PHASE1_EPOCHS)
    parser.add_argument('--lr',         type=float, default=PHASE1_LR)
    parser.add_argument('--batch-size', type=int,   default=PHASE1_BATCH_SIZE)
    parser.add_argument('--patience',   type=int,   default=PHASE1_PATIENCE)
    parser.add_argument('--no-resume',  action='store_true')
    args = parser.parse_args()

    run_phase1(
        data_dir=args.data_dir, save_dir=args.save_dir,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        patience=args.patience, resume=not args.no_resume,
    )
