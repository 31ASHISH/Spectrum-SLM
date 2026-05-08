"""
training/train_phase2.py
========================
Standalone Phase 2 trainer — Supervised Multi-task Fine-tuning.

Loads all real labeled data from both sources (Secondary_User + new dataset).
Trains PU detection, Modulation classification (5-class), SNR estimation.

CLI:
    python training/train_phase2.py
    python training/train_phase2.py --epochs 50 --dry-run

Authors : Anjani, Ashish Joshi, Mayank
Guide   : Dr. Abhinandan S.P.
Dated   : May 2026
"""

import os, sys, json, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import (
    SECONDARY_USER_DIR, NEW_DATASET_DIR, CKPT_PHASE1, CKPT_PHASE2,
    N_BINS, D_MODEL, N_HEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
    N_MOD_CLASSES_V2, MOD_NAMES_V2,
    PHASE2_LR, PHASE2_EPOCHS, PHASE2_BATCH_SIZE, PHASE2_PATIENCE,
    PHASE2_NUM_WORKERS, PHASE2_LEARN_WEIGHTS,
    LOSS_ALPHA, LOSS_BETA, LOSS_GAMMA,
    CKPT_PHASE2_BEST, NORMALIZER_FILE, METRICS_FILE,
    PREDICTIONS_FILE, HISTORY_FILE, ensure_dirs,
)
from spectrum_slm_dataset import build_dataloaders
from spectrum_slm_model   import SpectrumSLM
from spectrum_slm_train   import (
    get_device, load_checkpoint, finetune_supervised, evaluate_model,
)


def export_predictions(model, test_loader, save_path, device):
    """Save test-set predictions to CSV."""
    model.eval().to(device)
    rows = []
    with torch.no_grad():
        for psd, pu_lab, mod_lab, snr_lab in test_loader:
            out      = model(psd.to(device))
            pu_probs = torch.softmax(out['pu_logits'],  dim=1).cpu().numpy()
            mp       = torch.softmax(out['mod_logits'], dim=1).cpu().numpy()
            snr_pred = out['snr_pred'].cpu().numpy()
            for i in range(len(pu_lab)):
                row = {
                    'true_pu':     int(pu_lab[i]),
                    'pred_pu':     int(pu_probs[i, 1] > 0.5),
                    'pu_conf':     round(float(pu_probs[i, 1]), 4),
                    'true_mod':    MOD_NAMES_V2[int(mod_lab[i])] if int(mod_lab[i]) >= 0 else 'UNK',
                    'pred_mod':    MOD_NAMES_V2[int(np.argmax(mp[i]))],
                    'mod_conf':    round(float(np.max(mp[i])), 4),
                    'true_snr_db': round(float(snr_lab[i]), 2),
                    'pred_snr_db': round(float(snr_pred[i]), 2),
                }
                for j, nm in enumerate(MOD_NAMES_V2):
                    row[f'prob_{nm}'] = round(float(mp[i, j]), 4)
                rows.append(row)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"  Predictions → {save_path}  ({len(rows):,} rows)")


def run_phase2(
    secondary_user_dir: str   = SECONDARY_USER_DIR,
    new_dataset_dir:    str   = NEW_DATASET_DIR,
    save_dir:           str   = CKPT_PHASE2,
    epochs:             int   = PHASE2_EPOCHS,
    batch_size:         int   = PHASE2_BATCH_SIZE,
    lr:                 float = PHASE2_LR,
    patience:           int   = PHASE2_PATIENCE,
    learn_weights:      bool  = PHASE2_LEARN_WEIGHTS,
    resume:             bool  = True,
    dry_run:            bool  = False,
    resume_phase1:      bool  = True,
) -> dict:
    """End-to-end Phase 2 training."""
    ensure_dirs()
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Spectrum-SLM — Phase 2: Supervised Multi-task")
    print(f"  Device  : {device}")
    print(f"  Classes : {N_MOD_CLASSES_V2}  {MOD_NAMES_V2}")
    print(f"{'='*60}")

    # Build dataloaders
    norm_path = os.path.join(save_dir, NORMALIZER_FILE)
    train_loader, val_loader, test_loader, normalizer, meta = build_dataloaders(
        secondary_user_dir   = secondary_user_dir,
        new_dataset_dir      = new_dataset_dir,
        phase                = 2,
        batch_size           = batch_size,
        num_workers          = PHASE2_NUM_WORKERS,
        normalizer_save_path = norm_path,
    )

    if dry_run:
        print("\n[DRY RUN] Verifying one batch...")
        batch = next(iter(train_loader))
        p, pu, mod, snr = batch
        print(f"  PSD:{p.shape}  PU:{pu.shape}  Mod:{mod.shape}  SNR:{snr.shape}")
        print(f"  Mod IDs: {mod.unique().tolist()}")
        print("  ✓ Dry run passed\n")
        return {}

    # Model
    model = SpectrumSLM(n_bins=N_BINS, patch_size=1, d_model=D_MODEL,
                        nhead=N_HEAD, num_layers=NUM_LAYERS,
                        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
                        n_mod_classes=N_MOD_CLASSES_V2)
    print(f"\n  Parameters: {model.count_parameters():,}")

    # Load Phase 1 weights if available
    p1_ckpt = os.path.join(CKPT_PHASE1, 'slm_phase1_best.pt')
    if resume_phase1 and os.path.exists(p1_ckpt):
        print(f"\n  [INIT] Loading Phase 1 weights: {p1_ckpt}")
        try:
            load_checkpoint(model, p1_ckpt, device=device)
        except Exception as e:
            print(f"  [WARN] Could not load Phase 1: {e}")

    # Train
    history = finetune_supervised(
        model=model, train_loader=train_loader, val_loader=val_loader,
        pu_class_weight=meta['pu_weights'],
        n_epochs=epochs, lr=lr, device=device, save_dir=save_dir,
        patience=patience, alpha=LOSS_ALPHA, beta=LOSS_BETA, gamma=LOSS_GAMMA,
        learn_weights=learn_weights,
    )

    # Save history
    hist_path = os.path.join(save_dir, HISTORY_FILE)
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  History → {hist_path}")

    # Save last checkpoint
    torch.save(model.state_dict(), os.path.join(save_dir, CKPT_PHASE2_LAST))

    # Load best and evaluate
    best_ckpt = os.path.join(save_dir, CKPT_PHASE2_BEST)
    if os.path.exists(best_ckpt):
        load_checkpoint(model, best_ckpt, device=device)

    metrics = evaluate_model(model, test_loader, device=device)
    metrics.update({
        'dataset': 'real_combined', 'n_mod_classes': N_MOD_CLASSES_V2,
        'mod_names': MOD_NAMES_V2, 'n_epochs_run': len(history),
        **{k: meta[k] for k in ['n_train','n_val','n_test']},
    })

    met_path = os.path.join(save_dir, METRICS_FILE)
    with open(met_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics → {met_path}")

    pred_path = os.path.join(save_dir, PREDICTIONS_FILE)
    export_predictions(model, test_loader, pred_path, device)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spectrum-SLM Phase 2 Trainer')
    parser.add_argument('--su-dir',     default=SECONDARY_USER_DIR)
    parser.add_argument('--nd-dir',     default=NEW_DATASET_DIR)
    parser.add_argument('--save-dir',   default=CKPT_PHASE2)
    parser.add_argument('--epochs',     type=int,   default=PHASE2_EPOCHS)
    parser.add_argument('--batch-size', type=int,   default=PHASE2_BATCH_SIZE)
    parser.add_argument('--lr',         type=float, default=PHASE2_LR)
    parser.add_argument('--patience',   type=int,   default=PHASE2_PATIENCE)
    parser.add_argument('--dry-run',    action='store_true')
    parser.add_argument('--no-resume',  action='store_true')
    parser.add_argument('--no-phase1',  action='store_true')
    args = parser.parse_args()

    run_phase2(
        secondary_user_dir=args.su_dir,
        new_dataset_dir=args.nd_dir,
        save_dir=args.save_dir,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        resume_phase1=not args.no_phase1,
    )
