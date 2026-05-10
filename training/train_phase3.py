"""
training/train_phase3.py
========================
Phase 3: Generative AutoEncoder Fine-tuning.

SAFETY GUARANTEE:
  - Loads Phase 2 checkpoint as starting point (read-only).
  - Freezes ALL weights EXCEPT gen_head.
  - Saves ONLY to checkpoints/phase3/ — Phase 1 and Phase 2 are NEVER touched.
  - Classification accuracy (93%) is preserved because backbone weights are frozen.

Goal:
  Teach gen_head to reconstruct the input PSD cleanly from the CLS token,
  acting as a signal envelope predictor / denoiser.

CLI:
    python training/train_phase3.py
    python training/train_phase3.py --epochs 20 --lr 0.005

Authors : Anjani, Ashish Joshi, Mayank
Guide   : Dr. Abhinandan S.P.
Dated   : May 2026
"""

import os, sys, json, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    SECONDARY_USER_DIR, NEW_DATASET_DIR,
    CKPT_PHASE2, CKPT_PHASE3, CKPT_PHASE2_BEST, NORMALIZER_FILE,
    N_BINS, D_MODEL, N_HEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
    N_MOD_CLASSES_V2, ensure_dirs,
)
from spectrum_slm_model import SpectrumSLM
from spectrum_slm_train import get_device


# ── Phase 3 hyperparameters ────────────────────────────────────────────────────
PHASE3_LR        = 5e-3
PHASE3_EPOCHS    = 20
PHASE3_BATCH     = 256
PHASE3_PATIENCE  = 5
PHASE3_BEST_FILE = "slm_phase3_best.pt"


def _collect_psds(su_dir: str, nd_dir: str) -> np.ndarray:
    """
    Collect raw PSD arrays (float32, shape N×192) from all available .pth files.
    Labels are NOT needed — Phase 3 is purely unsupervised reconstruction.
    """
    psds = []

    def _from_binned(path: str):
        if not os.path.exists(path):
            return
        data = torch.load(path, map_location="cpu", weights_only=False)
        pairs = data.get("pairs_by_bin", {})
        for bin_key in pairs:
            for entry in pairs[bin_key]:
                raw = np.array(entry[0], dtype=np.float32).flatten()
                if len(raw) >= N_BINS:
                    psds.append(raw[:N_BINS])
                else:
                    psds.append(np.pad(raw, (0, N_BINS - len(raw))))

    def _from_log(path: str):
        if not os.path.exists(path):
            return
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, (list, tuple)):
            for item in data:
                raw = np.array(item, dtype=np.float32).flatten()
                if len(raw) >= N_BINS:
                    psds.append(raw[:N_BINS])
        elif isinstance(data, dict):
            for v in data.values():
                try:
                    raw = np.array(v, dtype=np.float32).flatten()
                    if len(raw) >= N_BINS:
                        psds.append(raw[:N_BINS])
                except Exception:
                    pass

    # Secondary_User/ — binned files
    if os.path.isdir(su_dir):
        for f in os.listdir(su_dir):
            if f.endswith(".pth"):
                _from_binned(os.path.join(su_dir, f))

    # New dataset (log files)
    if os.path.isdir(nd_dir):
        for root, dirs, files in os.walk(nd_dir):
            for f in files:
                if f.endswith(".pth"):
                    fp = os.path.join(root, f)
                    _from_binned(fp)   # try binned first
                    _from_log(fp)      # also try log

    return np.array(psds, dtype=np.float32)


def run_phase3(
    secondary_user_dir: str  = SECONDARY_USER_DIR,
    new_dataset_dir:    str  = NEW_DATASET_DIR,
    p2_ckpt_dir:        str  = CKPT_PHASE2,
    save_dir:           str  = CKPT_PHASE3,
    epochs:             int  = PHASE3_EPOCHS,
    lr:                 float= PHASE3_LR,
    batch_size:         int  = PHASE3_BATCH,
    patience:           int  = PHASE3_PATIENCE,
) -> float:
    """
    Train gen_head as an AutoEncoder on top of the frozen Phase 2 backbone.

    Returns:
        best_val_loss (float)
    """
    ensure_dirs()
    os.makedirs(save_dir, exist_ok=True)
    device = get_device()

    print(f"\n{'='*60}")
    print("  Spectrum-SLM — Phase 3: Generative AutoEncoder")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {epochs}   LR: {lr}   Batch: {batch_size}")
    print(f"  Save to : {save_dir}")
    print(f"  SAFETY  : Phase 1 & Phase 2 checkpoints are NOT modified.")
    print(f"{'='*60}")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n[P3] Collecting PSD samples...")
    psds_raw = _collect_psds(secondary_user_dir, new_dataset_dir)
    print(f"  Total raw PSD samples : {len(psds_raw):,}")

    if len(psds_raw) == 0:
        raise RuntimeError("No PSD data found. Check SECONDARY_USER_DIR / NEW_DATASET_DIR.")

    # Normalise using Phase 2 scaler
    norm_path = os.path.join(p2_ckpt_dir, NORMALIZER_FILE)
    if os.path.exists(norm_path):
        with open(norm_path, "rb") as f:
            scaler = pickle.load(f)
        n_feats = getattr(scaler, "n_features_in_", None)
        if n_feats == N_BINS:
            psds_norm = scaler.transform(psds_raw).astype(np.float32)
            print(f"  Normalizer loaded from {norm_path}")
        else:
            print(f"  [WARN] Scaler expects {n_feats} features, got {N_BINS} — using z-score")
            psds_norm = ((psds_raw - psds_raw.mean(axis=1, keepdims=True))
                         / (psds_raw.std(axis=1, keepdims=True) + 1e-8))
    else:
        print("  [WARN] normalizer.pkl not found — using per-sample z-score")
        psds_norm = ((psds_raw - psds_raw.mean(axis=1, keepdims=True))
                     / (psds_raw.std(axis=1, keepdims=True) + 1e-8))

    # Train / Val split (85 / 15)
    n_total = len(psds_norm)
    n_val   = max(1, int(0.15 * n_total))
    idx     = np.random.permutation(n_total)
    X_train = torch.tensor(psds_norm[idx[n_val:]])
    X_val   = torch.tensor(psds_norm[idx[:n_val]])
    print(f"  Train : {len(X_train):,}   Val : {len(X_val):,}")

    train_loader = DataLoader(TensorDataset(X_train, X_train),
                              batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(TensorDataset(X_val,   X_val),
                              batch_size=batch_size, shuffle=False, drop_last=False)

    # ── 2. Load Phase 2 model ────────────────────────────────────────────────
    model = SpectrumSLM(
        n_bins=N_BINS, patch_size=1, d_model=D_MODEL,
        nhead=N_HEAD, num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
        n_mod_classes=N_MOD_CLASSES_V2,
    )
    p2_path = os.path.join(p2_ckpt_dir, CKPT_PHASE2_BEST)
    if not os.path.exists(p2_path):
        raise FileNotFoundError(f"Phase 2 checkpoint not found: {p2_path}")

    ck = torch.load(p2_path, map_location="cpu", weights_only=False)
    model_state  = model.state_dict()
    loaded_state = ck.get("model", ck)
    # Filter mismatched keys (safety)
    compatible = {k: v for k, v in loaded_state.items()
                  if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(compatible, strict=False)
    print(f"\n  Loaded Phase 2 checkpoint (epoch {ck.get('epoch','?')}).")
    print(f"  SAFETY: {len(compatible)}/{len(model_state)} keys loaded successfully.")

    # ── 3. FREEZE everything except gen_head ─────────────────────────────────
    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        if "gen_head" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()

    print(f"  Frozen  : {frozen:,} params  (backbone + classification heads)")
    print(f"  Trainable: {trainable:,} params  (gen_head only)")
    model.to(device)

    # ── 4. Train ─────────────────────────────────────────────────────────────
    optimizer  = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=2)
    criterion  = nn.MSELoss()

    best_val   = float("inf")
    no_improve = 0

    print(f"\n{'─'*60}")
    for ep in range(1, epochs + 1):
        # Train
        model.train()
        tr_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out  = model(bx)
            loss = criterion(out["gen_pred"], by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            optimizer.step()
            tr_losses.append(loss.item())

        # Validate
        model.eval()
        vl_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                loss   = criterion(model(bx)["gen_pred"], by)
                vl_losses.append(loss.item())

        tr_l = float(np.mean(tr_losses))
        vl_l = float(np.mean(vl_losses))
        scheduler.step(vl_l)
        print(f"  Epoch {ep:3d}/{epochs} │ train_mse={tr_l:.4f} │ val_mse={vl_l:.4f}", end="")

        if vl_l < best_val:
            best_val = vl_l
            no_improve = 0
            # Save Phase 3 checkpoint — Phase 2 is NEVER touched
            ck_save = {
                "model"    : model.state_dict(),
                "epoch"    : ep,
                "val_loss" : best_val,
                "phase"    : 3,
                "source_p2": p2_path,
            }
            out_path = os.path.join(save_dir, PHASE3_BEST_FILE)
            torch.save(ck_save, out_path)
            print("  ← BEST saved", end="")
        else:
            no_improve += 1
        print()

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {ep} (no improvement for {patience} epochs)")
            break

    print(f"\n{'─'*60}")
    print(f"  Phase 3 done.  Best val MSE: {best_val:.4f}")
    print(f"  Saved → {os.path.join(save_dir, PHASE3_BEST_FILE)}")
    print(f"  Phase 1 & Phase 2 checkpoints were NOT modified.")
    return best_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectrum-SLM Phase 3 — GenHead AutoEncoder")
    parser.add_argument("--su-dir",     default=SECONDARY_USER_DIR)
    parser.add_argument("--nd-dir",     default=NEW_DATASET_DIR)
    parser.add_argument("--p2-dir",     default=CKPT_PHASE2)
    parser.add_argument("--save-dir",   default=CKPT_PHASE3)
    parser.add_argument("--epochs",     type=int,   default=PHASE3_EPOCHS)
    parser.add_argument("--lr",         type=float, default=PHASE3_LR)
    parser.add_argument("--batch-size", type=int,   default=PHASE3_BATCH)
    parser.add_argument("--patience",   type=int,   default=PHASE3_PATIENCE)
    args = parser.parse_args()

    run_phase3(
        secondary_user_dir=args.su_dir,
        new_dataset_dir=args.nd_dir,
        p2_ckpt_dir=args.p2_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )
