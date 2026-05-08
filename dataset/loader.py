"""
dataset/loader.py
=================
Unified PTH loader for Spectrum-SLM.

Handles two real data sources:
  1. Secondary_User/          — Binned format: {bins, pairs_by_bin}  → psd (192,)
  2. files-20260414T094743Z-3-001/ — Log format: {psds:(N,192,1), snrs, pu_flags} → psd (192,)

Symbol1 (1024-bin files) are automatically skipped.
Corrupted .pth files are skipped gracefully with a warning.

Returns tuples: (psd [192,], pu_label, mod_label, snr_value)

Authors : Anjani, Ashish Joshi, Mayank
Guide   : Dr. Abhinandan S.P.
Dated   : May 2026
"""

import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
N_BINS = 192   # Confirmed from real data inspection

MOD_MAP = {"bpsk": 0, "qpsk": 1, "8psk": 2, "16qam": 3, "dqpsk": 4}

# Folder → mod ID for new dataset
_FOLDER_MOD_MAP = {"bpsk": 0, "qpsk": 1, "8psk": 2, "16qam": 3, "dqpsk": 4}

# Secondary_User files: filename → (format, mod_id)
_SU_FILES = {
    "psd_binned_by_snr_bpsk.pth": ("binned", 0),
    "psd_binned_by_snr_qpsk.pth": ("binned", 1),
    "psd_log_8psk.pth":           ("log",    2),
    "psd_log_16qam.pth":          ("log",    3),
    # psd_binned_by_snr_.pth → Phase 1 only (no mod label)
    # psd_binned_by_snr_16qam.pth → CORRUPTED, skipped automatically
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (np.empty((0, N_BINS), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32))


def _concat(plist, pulist, modlist, snrlist):
    return (np.concatenate(plist,   axis=0),
            np.concatenate(pulist,  axis=0),
            np.concatenate(modlist, axis=0),
            np.concatenate(snrlist, axis=0))


def _safe_load(path: str) -> Optional[object]:
    """Load a .pth file, return None if corrupted."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [SKIP] Corrupted: {os.path.basename(path)} — {e}")
        return None


def _to_192(psd_raw) -> Optional[np.ndarray]:
    """Convert any PSD to (192,) float32; return None if incompatible."""
    arr = np.array(psd_raw, dtype=np.float32).flatten()
    if len(arr) == N_BINS:
        return arr
    if len(arr) == 1024:
        return None   # Wrong resolution — skip
    if len(arr) < N_BINS:
        return None   # Too short
    return arr[:N_BINS]  # Crop


# ---------------------------------------------------------------------------
# Loader A: Binned format  {bins:[...], pairs_by_bin:{snr:[(psd,label)]}}
# ---------------------------------------------------------------------------

def load_binned_pth(path: str, mod_id: int = -1):
    data = _safe_load(path)
    if data is None:
        return _empty()
    if not isinstance(data, dict) or "pairs_by_bin" not in data:
        print(f"  [SKIP] Not binned format: {os.path.basename(path)}")
        return _empty()

    bins  = data.get("bins", [])
    pairs = data.get("pairs_by_bin", {})
    psds, pu_l, mod_l, snr_l = [], [], [], []

    for snr_bin in bins:
        if snr_bin not in pairs:
            continue
        for psd_raw, label in pairs[snr_bin]:
            arr = _to_192(psd_raw)
            if arr is None:
                continue
            psds.append(arr)
            pu_l.append(int(label))
            mod_l.append(mod_id)
            snr_l.append(float(snr_bin))

    if not psds:
        return _empty()
    return (np.stack(psds),
            np.array(pu_l,  dtype=np.int64),
            np.array(mod_l, dtype=np.int64),
            np.array(snr_l, dtype=np.float32))


# ---------------------------------------------------------------------------
# Loader B: Log format  {psds:(N,192,1), snrs:(N,), pu_flags:(N,)}
# ---------------------------------------------------------------------------

def load_log_pth(path: str, mod_id: int = -1):
    data = _safe_load(path)
    if data is None:
        return _empty()
    if not isinstance(data, dict) or "psds" not in data:
        print(f"  [SKIP] Not log format: {os.path.basename(path)}")
        return _empty()

    psds_raw = np.array(data["psds"], dtype=np.float32)
    snrs_raw = np.array(data["snrs"], dtype=np.float32)
    pu_key   = "pu_flags" if "pu_flags" in data else "pu_labels"
    pu_raw   = np.array(data[pu_key], dtype=np.int64)

    # (N, 192, 1) → (N, 192)
    if psds_raw.ndim == 3 and psds_raw.shape[2] == 1:
        psds_raw = psds_raw.squeeze(-1)

    if psds_raw.ndim == 2 and psds_raw.shape[1] != N_BINS:
        print(f"  [SKIP] Wrong bins {psds_raw.shape[1]}: {os.path.basename(path)}")
        return _empty()

    n = len(psds_raw)
    return (psds_raw,
            pu_raw,
            np.full(n, mod_id, dtype=np.int64),
            snrs_raw)


# ---------------------------------------------------------------------------
# Source 1: Secondary_User/
# ---------------------------------------------------------------------------

def load_secondary_user(data_dir: str, for_phase1: bool = False):
    """
    Load Secondary_User/ dataset.

    for_phase1=True  → only psd_binned_by_snr_.pth (mixed, no mod label)
    for_phase1=False → all labeled modulation files
    """
    all_p, all_pu, all_mod, all_snr = [], [], [], []

    if for_phase1:
        fpath = os.path.join(data_dir, "psd_binned_by_snr_.pth")
        if os.path.exists(fpath):
            p, pu, mod, snr = load_binned_pth(fpath, mod_id=-1)
            if len(p) > 0:
                all_p.append(p); all_pu.append(pu)
                all_mod.append(mod); all_snr.append(snr)
                print(f"  Phase1 data: {len(p):,} samples from psd_binned_by_snr_.pth")
    else:
        for fname, (fmt, mod_id) in _SU_FILES.items():
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                continue
            fn = load_binned_pth if fmt == "binned" else load_log_pth
            p, pu, mod, snr = fn(fpath, mod_id)
            if len(p) > 0:
                all_p.append(p); all_pu.append(pu)
                all_mod.append(mod); all_snr.append(snr)
                print(f"  Secondary_User/{fname}: {len(p):,} samples (mod={mod_id})")

    if not all_p:
        return _empty()
    return _concat(all_p, all_pu, all_mod, all_snr)


# ---------------------------------------------------------------------------
# Source 2: files-20260414T094743Z-3-001/
# ---------------------------------------------------------------------------

def load_new_dataset(data_dir: str, symbol_dirs: Optional[List[str]] = None):
    """
    Load 192-bin dataset.pth files from Symbol2/ and Symbol3/.
    Symbol1 is excluded (1024-bin, incompatible resolution).
    """
    if symbol_dirs is None:
        symbol_dirs = ["Symbol2", "Symbol3"]

    all_p, all_pu, all_mod, all_snr = [], [], [], []

    for sym in symbol_dirs:
        if sym == "Symbol1":
            print(f"  [SKIP] Symbol1 — 1024-bin files")
            continue
        sym_dir = os.path.join(data_dir, sym)
        if not os.path.isdir(sym_dir):
            continue

        for mod_folder in sorted(os.listdir(sym_dir)):
            mod_id = _FOLDER_MOD_MAP.get(mod_folder.lower(), -1)
            if mod_id < 0:
                continue
            dset_path = os.path.join(sym_dir, mod_folder, "dataset.pth")
            if not os.path.exists(dset_path):
                continue
            p, pu, mod, snr = load_log_pth(dset_path, mod_id)
            if len(p) > 0:
                all_p.append(p); all_pu.append(pu)
                all_mod.append(mod); all_snr.append(snr)
                print(f"  {sym}/{mod_folder}/dataset.pth: {len(p):,} samples (mod={mod_id})")

    if not all_p:
        return _empty()
    return _concat(all_p, all_pu, all_mod, all_snr)


# ---------------------------------------------------------------------------
# Master loader: combine both sources
# ---------------------------------------------------------------------------

def load_all_real_data(
    secondary_user_dir: str,
    new_dataset_dir:    str,
    for_phase1:         bool = False,
    symbol_dirs:        Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all real SDR data from both sources.

    Returns:
        psds       : (N, 192)  float32
        pu_labels  : (N,)      int64
        mod_labels : (N,)      int64   (-1 = unknown/Phase1)
        snr_labels : (N,)      float32
    """
    all_p, all_pu, all_mod, all_snr = [], [], [], []

    print("\n[LOADER] Loading Secondary_User/ ...")
    p, pu, mod, snr = load_secondary_user(secondary_user_dir, for_phase1=for_phase1)
    if len(p) > 0:
        all_p.append(p); all_pu.append(pu)
        all_mod.append(mod); all_snr.append(snr)

    if not for_phase1:
        print("\n[LOADER] Loading new dataset (Symbol2/Symbol3) ...")
        p, pu, mod, snr = load_new_dataset(new_dataset_dir, symbol_dirs)
        if len(p) > 0:
            all_p.append(p); all_pu.append(pu)
            all_mod.append(mod); all_snr.append(snr)

    if not all_p:
        raise RuntimeError("No real data found! Check data paths.")

    psds       = np.concatenate(all_p,   axis=0)
    pu_labels  = np.concatenate(all_pu,  axis=0)
    mod_labels = np.concatenate(all_mod, axis=0)
    snr_labels = np.concatenate(all_snr, axis=0)

    print(f"\n[LOADER] Total: {len(psds):,} samples")
    print(f"  PSD shape   : {psds.shape}")
    print(f"  PU=1        : {pu_labels.sum():,} ({100*pu_labels.mean():.1f}%)")
    print(f"  SNR range   : {snr_labels.min():.1f} – {snr_labels.max():.1f} dB")
    unique_mods, counts = np.unique(mod_labels[mod_labels >= 0], return_counts=True)
    mod_names = {0:"BPSK",1:"QPSK",2:"8PSK",3:"16QAM",4:"DQPSK"}
    for m, c in zip(unique_mods, counts):
        print(f"  {mod_names.get(m,'?'):<8}: {c:,}")

    return psds, pu_labels, mod_labels, snr_labels


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    su_dir  = os.path.join(root, "Secondary_User")
    new_dir = os.path.join(root, "files-20260414T094743Z-3-001")

    psds, pu, mod, snr = load_all_real_data(su_dir, new_dir)
    print(f"\nSanity check — NaN in PSDs: {np.isnan(psds).sum()}")
    print(f"PSD min={psds.min():.3f}  max={psds.max():.3f}")
    print("Loader OK ✓")
