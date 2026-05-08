"""
training/run_3_phases.py
========================
Orchestrates all training phases.

Phase 3 is SKIPPED — no real temporal sequence data exists.

Authors : Anjani, Ashish Joshi, Mayank
Dated   : May 2026
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SECONDARY_USER_DIR, NEW_DATASET_DIR, CKPT_PHASE1, CKPT_PHASE2,
    PHASE2_BATCH_SIZE, PHASE2_NUM_WORKERS, N_BINS, N_MOD_CLASSES_V2,
    PHASE1_LR, PHASE1_EPOCHS, PHASE1_PATIENCE,
    PHASE2_LR, PHASE2_EPOCHS, PHASE2_PATIENCE,
    D_MODEL, N_HEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
    NORMALIZER_FILE, ensure_dirs,
)
from spectrum_slm_model   import SpectrumSLM
from spectrum_slm_dataset import build_dataloaders
from spectrum_slm_train   import get_device, load_checkpoint
from training.train_phase1 import run_phase1
from training.train_phase2 import run_phase2


def run_all_phases(
    epochs_p1:   int = PHASE1_EPOCHS,
    epochs_p2:   int = PHASE2_EPOCHS,
    batch_size:  int = PHASE2_BATCH_SIZE,
    skip_phase1: bool = False,
):
    ensure_dirs()
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  Spectrum-SLM — 3-Phase Training  (device={device})")
    print(f"  Phase 3: SKIPPED (no real temporal data)")
    print(f"{'='*60}")

    # ── Phase 1: Masked Spectrum Modelling ─────────────────────────────────
    p1_ckpt = os.path.join(CKPT_PHASE1, 'slm_phase1_best.pt')
    if skip_phase1 and os.path.exists(p1_ckpt):
        print(f"\n[PHASE 1] Checkpoint exists — skipping training")
    else:
        print(f"\n[PHASE 1] Starting Masked Spectrum Modelling...")
        run_phase1(
            data_dir=SECONDARY_USER_DIR, save_dir=CKPT_PHASE1,
            epochs=epochs_p1, lr=PHASE1_LR, batch_size=batch_size,
            patience=PHASE1_PATIENCE,
        )

    # ── Phase 2: Supervised Multi-task ─────────────────────────────────────
    print(f"\n[PHASE 2] Starting Supervised Multi-task Fine-tuning...")
    metrics = run_phase2(
        secondary_user_dir=SECONDARY_USER_DIR,
        new_dataset_dir=NEW_DATASET_DIR,
        save_dir=CKPT_PHASE2,
        epochs=epochs_p2, lr=PHASE2_LR,
        batch_size=batch_size, patience=PHASE2_PATIENCE,
    )

    # ── Phase 3: SKIPPED ───────────────────────────────────────────────────
    print(f"\n[PHASE 3] SKIPPED — No real temporal sequence data in dataset.")
    print(f"  Creating Phase 3 checkpoint alias from Phase 2 best...")
    import shutil
    p2_best = os.path.join(CKPT_PHASE2, 'slm_phase2_best.pt')
    p3_dir  = os.path.join(os.path.dirname(CKPT_PHASE2), 'phase3')
    os.makedirs(p3_dir, exist_ok=True)
    p3_out = os.path.join(p3_dir, 'slm_phase3_best.pt')
    if os.path.exists(p2_best) and not os.path.exists(p3_out):
        shutil.copy2(p2_best, p3_out)
        print(f"  Copied Phase 2 → {p3_out}")

    print(f"\n{'='*60}")
    print("  ALL PHASES COMPLETE")
    if metrics:
        print(f"  PU Accuracy  : {metrics.get('pu_accuracy', 0)*100:.2f}%")
        print(f"  Mod Accuracy : {metrics.get('mod_accuracy', 0)*100:.2f}%")
        print(f"  SNR MAE      : {metrics.get('snr_mae_db', 0):.3f} dB")
    print(f"{'='*60}")
    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs-p1',  type=int, default=PHASE1_EPOCHS)
    parser.add_argument('--epochs-p2',  type=int, default=PHASE2_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=PHASE2_BATCH_SIZE)
    parser.add_argument('--skip-p1',    action='store_true')
    args = parser.parse_args()
    run_all_phases(args.epochs_p1, args.epochs_p2, args.batch_size, args.skip_p1)
