"""
config.py
=========
Central configuration for Spectrum-SLM.
All paths and hyperparameters live here — zero hardcoding elsewhere.

KEY FACT: Real PSD shape = (192,) — confirmed by inspecting all .pth files.

Authors : Anjani, Ashish Joshi, Mayank
Guide   : Dr. Abhinandan S.P.
Dated   : May 2026
"""

import os

# ─── Root ─────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Dataset paths ────────────────────────────────────────────────────────────
# Source 1: Secondary_User/ (binned .pth, 192-bin)
SECONDARY_USER_DIR  = os.path.join(ROOT_DIR, "Secondary_User")
# Source 2: files-20260414T094743Z-3-001/ (log .pth, 192-bin, includes DQPSK)
NEW_DATASET_DIR     = os.path.join(ROOT_DIR, "files-20260414T094743Z-3-001")
# Phase 1 MSM file (mixed modulations, no mod label)
PHASE1_DATA_FILE    = os.path.join(SECONDARY_USER_DIR, "psd_binned_by_snr_.pth")

# Legacy aliases
PHASE1_DATA_DIR     = SECONDARY_USER_DIR
PHASE2_DATA_DIR     = NEW_DATASET_DIR
PHASE2_SYMBOL_DIRS  = ["Symbol2", "Symbol3"]   # 192-bin only
PHASE2_MODULATIONS  = ["bpsk", "qpsk", "8psk", "16qam", "dqpsk"]

# ─── Checkpoints ──────────────────────────────────────────────────────────────
CKPT_ROOT   = os.path.join(ROOT_DIR, "checkpoints")
CKPT_PHASE1 = os.path.join(CKPT_ROOT, "phase1")
CKPT_PHASE2 = os.path.join(CKPT_ROOT, "phase2")
CKPT_PHASE3 = os.path.join(CKPT_ROOT, "phase3")
LEGACY_CKPT_DIR = os.path.join(ROOT_DIR, "slm_checkpoints")

# ─── Model architecture ───────────────────────────────────────────────────────
N_BINS          = 192          # ← Real PSD size (confirmed from all .pth files)
PATCH_SIZE      = 1            # Each frequency bin = 1 token
N_PATCHES       = N_BINS       # 192 tokens
SEQ_LEN         = N_PATCHES + 1  # 193 (192 patches + 1 CLS)
D_MODEL         = 128
N_HEAD          = 4
NUM_LAYERS      = 4
DIM_FEEDFORWARD = 512
DROPOUT         = 0.1

# ─── Modulation classes (5-class, all confirmed in real data) ─────────────────
MOD_MAP_V1       = {"BPSK": 0, "QPSK": 1, "8PSK": 2, "16QAM": 3}          # legacy
N_MOD_CLASSES_V1 = 4

MOD_MAP_V2       = {"BPSK": 0, "QPSK": 1, "8PSK": 2, "16QAM": 3, "DQPSK": 4}
MOD_MAP_V2_INV   = {v: k for k, v in MOD_MAP_V2.items()}
N_MOD_CLASSES_V2 = 5
MOD_NAMES_V2     = ["BPSK", "QPSK", "8PSK", "16QAM", "DQPSK"]
MOD_COLORS_V2    = ["#58a6ff", "#3fb950", "#f78166", "#ffa657", "#d2a8ff"]

# ─── Training hyperparameters ─────────────────────────────────────────────────
PHASE1_LR         = 3e-4
PHASE1_EPOCHS     = 30
PHASE1_BATCH_SIZE = 64
PHASE1_PATIENCE   = 5
PHASE1_MASK_RATIO = 0.20

PHASE2_BATCH_SIZE    = 64
PHASE2_LR            = 1e-4
PHASE2_EPOCHS        = 50
PHASE2_PATIENCE      = 8
PHASE2_VAL_RATIO     = 0.15
PHASE2_TEST_RATIO    = 0.15
PHASE2_MASK_RATIO    = 0.20
PHASE2_RANDOM_STATE  = 42
PHASE2_NUM_WORKERS   = 0
PHASE2_AUGMENT       = True
PHASE2_LEARN_WEIGHTS = True    # Kendall uncertainty weighting

# Multi-task loss weights
LOSS_ALPHA = 1.0   # PU detection (Focal)
LOSS_BETA  = 0.5   # Modulation (CE)
LOSS_GAMMA = 0.3   # SNR (Huber)

# ─── SNR evaluation bins ──────────────────────────────────────────────────────
SNR_BINS          = [4, 6, 8, 10, 12, 14, 16, 18, 20]
LOW_SNR_THRESHOLD = 8.0   # dB

# ─── Artifact filenames ───────────────────────────────────────────────────────
CKPT_PHASE2_BEST  = "slm_phase2_best.pt"
CKPT_PHASE2_LAST  = "slm_phase2_last.pt"
NORMALIZER_FILE   = "normalizer.pkl"
METRICS_FILE      = "metrics_phase2.json"
PREDICTIONS_FILE  = "predictions_phase2.csv"
HISTORY_FILE      = "training_history_phase2.json"


# ─── Utilities ────────────────────────────────────────────────────────────────
def ensure_dirs():
    for d in [CKPT_ROOT, CKPT_PHASE1, CKPT_PHASE2, CKPT_PHASE3]:
        os.makedirs(d, exist_ok=True)


def get_phase2_ckpt_path(filename: str) -> str:
    ensure_dirs()
    return os.path.join(CKPT_PHASE2, filename)


def kaggle_override(kaggle_dataset_path: str):
    global SECONDARY_USER_DIR, NEW_DATASET_DIR, CKPT_PHASE2
    SECONDARY_USER_DIR = kaggle_dataset_path
    NEW_DATASET_DIR    = kaggle_dataset_path
    CKPT_PHASE2        = "/kaggle/working/checkpoints/phase2"
    os.makedirs(CKPT_PHASE2, exist_ok=True)
    print(f"[CONFIG] Kaggle mode — data: {kaggle_dataset_path}")
    print(f"[CONFIG] Kaggle mode — ckpt: {CKPT_PHASE2}")


if __name__ == "__main__":
    ensure_dirs()
    print("=== Spectrum-SLM Config ===")
    print(f"  ROOT_DIR           : {ROOT_DIR}")
    print(f"  SECONDARY_USER_DIR : {SECONDARY_USER_DIR}  (exists={os.path.isdir(SECONDARY_USER_DIR)})")
    print(f"  NEW_DATASET_DIR    : {NEW_DATASET_DIR}  (exists={os.path.isdir(NEW_DATASET_DIR)})")
    print(f"  N_BINS={N_BINS}  PATCH_SIZE={PATCH_SIZE}  SEQ_LEN={SEQ_LEN}")
    print(f"  N_MOD_CLASSES_V2={N_MOD_CLASSES_V2}  {MOD_NAMES_V2}")
    print("  Checkpoint dirs ✓")
