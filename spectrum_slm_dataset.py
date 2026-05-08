"""
spectrum_slm_dataset.py
=======================
PyTorch Dataset/DataLoader pipeline for Spectrum-SLM.

Uses dataset/loader.py to read real SDR .pth files (both sources).
NO synthetic data generation used for training or evaluation.

Authors : Anjani, Ashish Joshi, Mayank
Guide   : Dr. Abhinandan S.P.
Dated   : May 2026
"""

import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, List

# Allow import from parent dir
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from dataset.loader import load_all_real_data

# ─── Constants ────────────────────────────────────────────────────────────────
N_BINS   = 192
N_PATCHES = 192     # patch_size=1 → each bin is its own token
SNR_BINS = [4, 6, 8, 10, 12, 14, 16, 18, 20]
MOD_MAP  = {"BPSK": 0, "QPSK": 1, "8PSK": 2, "16QAM": 3, "DQPSK": 4}
MOD_MAP_INV = {v: k for k, v in MOD_MAP.items()}


# ─── Normalizer ───────────────────────────────────────────────────────────────

class SpectrumNormalizer:
    """Per-bin StandardScaler. Fit on train only; apply to val/test."""

    def __init__(self):
        self.scaler   = StandardScaler()
        self._fitted  = False

    def fit(self, psds: np.ndarray) -> 'SpectrumNormalizer':
        self.scaler.fit(psds)
        self._fitted = True
        return self

    def transform(self, psds: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() first"
        return self.scaler.transform(psds).astype(np.float32)

    def fit_transform(self, psds: np.ndarray) -> np.ndarray:
        return self.fit(psds).transform(psds)

    def inverse_transform(self, psds: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(psds)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  Normalizer saved → {path}")

    def load(self, path: str) -> 'SpectrumNormalizer':
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        self._fitted = True
        return self


def save_normalizer(norm: SpectrumNormalizer, path: str):
    norm.save(path)

def load_normalizer(path: str) -> SpectrumNormalizer:
    return SpectrumNormalizer().load(path)


# ─── Augmentation (TRAIN ONLY, real data only) ────────────────────────────────

class SpectrumAugmenter:
    """
    Allowed augmentations (applied to real PSD vectors during training):
      1. Tiny Gaussian noise injection
      2. Tiny amplitude scaling
      3. Tiny spectral shift (circular roll)
    """

    def __init__(self, noise_std=0.02, max_shift=3,
                 scale_range=(0.95, 1.05),
                 p_noise=0.5, p_shift=0.3, p_scale=0.3):
        self.noise_std   = noise_std
        self.max_shift   = max_shift
        self.scale_range = scale_range
        self.p_noise     = p_noise
        self.p_shift     = p_shift
        self.p_scale     = p_scale

    def augment(self, psd: np.ndarray) -> np.ndarray:
        """psd: (192,) float32"""
        if np.random.rand() < self.p_noise:
            psd = psd + np.random.randn(N_BINS).astype(np.float32) * self.noise_std
        if np.random.rand() < self.p_shift:
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            psd   = np.roll(psd, shift)
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(*self.scale_range)
            psd   = psd * scale
        return psd


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class SpectrumDataset(Dataset):
    """
    Phase 1 (MSM)  : returns (psd, mask)
    Phase 2 (SFT)  : returns (psd, pu_label, mod_label, snr_label)
    """

    def __init__(self, psds: np.ndarray, pu_labels: np.ndarray,
                 mod_labels: np.ndarray, snr_labels: np.ndarray,
                 phase: int = 2, mask_ratio: float = 0.20,
                 augmenter: Optional[SpectrumAugmenter] = None,
                 training: bool = True):
        self.psds       = psds
        self.pu_labels  = pu_labels
        self.mod_labels = mod_labels
        self.snr_labels = snr_labels
        self.phase      = phase
        self.mask_ratio = mask_ratio
        self.augmenter  = augmenter
        self.training   = training

    def __len__(self):
        return len(self.psds)

    def _random_mask(self) -> torch.Tensor:
        """Bool mask over 192 bin-tokens."""
        n_mask = max(1, int(N_PATCHES * self.mask_ratio))
        idx    = np.random.choice(N_PATCHES, n_mask, replace=False)
        mask   = torch.zeros(N_PATCHES, dtype=torch.bool)
        mask[idx] = True
        return mask

    def __getitem__(self, idx: int):
        psd = self.psds[idx].copy()
        if self.training and self.augmenter:
            psd = self.augmenter.augment(psd)

        if self.phase == 1:
            mask = self._random_mask()
            return torch.tensor(psd, dtype=torch.float32), mask
        else:
            return (
                torch.tensor(psd, dtype=torch.float32),
                torch.tensor(self.pu_labels[idx],  dtype=torch.long),
                torch.tensor(self.mod_labels[idx], dtype=torch.long),
                torch.tensor(self.snr_labels[idx], dtype=torch.float32),
            )


# ─── Full DataLoader builder ──────────────────────────────────────────────────

def build_dataloaders(
    secondary_user_dir: str,
    new_dataset_dir:    str,
    phase:              int   = 2,
    batch_size:         int   = 64,
    val_ratio:          float = 0.15,
    test_ratio:         float = 0.15,
    mask_ratio:         float = 0.20,
    num_workers:        int   = 0,
    use_weighted_sampler: bool = True,
    augment_train:      bool  = True,
    random_state:       int   = 42,
    normalizer_save_path: Optional[str] = None,
    symbol_dirs:        Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, SpectrumNormalizer, dict]:
    """
    Full pipeline: load real data → split → normalize → augment → DataLoaders.

    Returns: train_loader, val_loader, test_loader, normalizer, meta
    """
    for_phase1 = (phase == 1)

    psds, pu_labels, mod_labels, snr_labels = load_all_real_data(
        secondary_user_dir = secondary_user_dir,
        new_dataset_dir    = new_dataset_dir,
        for_phase1         = for_phase1,
        symbol_dirs        = symbol_dirs,
    )

    # For Phase 1: only use samples without mod label (mod=-1)
    # For Phase 2: only use samples with known modulation (mod>=0)
    if phase == 2:
        valid = mod_labels >= 0
        psds       = psds[valid]
        pu_labels  = pu_labels[valid]
        mod_labels = mod_labels[valid]
        snr_labels = snr_labels[valid]
        print(f"  Phase 2: {len(psds):,} labeled samples")

    # Stratified 70 / 15 / 15 split
    idx = np.arange(len(psds))
    strat = pu_labels  # stratify on PU label
    idx_train, idx_tmp = train_test_split(
        idx, test_size=(val_ratio + test_ratio),
        stratify=strat, random_state=random_state)
    vt_frac = test_ratio / (val_ratio + test_ratio)
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=vt_frac,
        stratify=strat[idx_tmp], random_state=random_state)

    # Normalize — fit on train ONLY
    normalizer   = SpectrumNormalizer()
    psds_train   = normalizer.fit_transform(psds[idx_train])
    psds_val     = normalizer.transform(psds[idx_val])
    psds_test    = normalizer.transform(psds[idx_test])

    if normalizer_save_path:
        normalizer.save(normalizer_save_path)

    augmenter = SpectrumAugmenter() if augment_train else None

    train_ds = SpectrumDataset(psds_train, pu_labels[idx_train],
                               mod_labels[idx_train], snr_labels[idx_train],
                               phase=phase, mask_ratio=mask_ratio,
                               augmenter=augmenter, training=True)
    val_ds   = SpectrumDataset(psds_val,   pu_labels[idx_val],
                               mod_labels[idx_val],   snr_labels[idx_val],
                               phase=phase, mask_ratio=mask_ratio,
                               augmenter=None, training=False)
    test_ds  = SpectrumDataset(psds_test,  pu_labels[idx_test],
                               mod_labels[idx_test],  snr_labels[idx_test],
                               phase=phase, mask_ratio=mask_ratio,
                               augmenter=None, training=False)

    # Weighted sampler for class balance (Phase 2)
    sampler = None
    if phase == 2 and use_weighted_sampler:
        pu_tr   = pu_labels[idx_train]
        counts  = np.bincount(pu_tr, minlength=2)
        weights = 1.0 / np.maximum(counts, 1)
        s_w     = weights[pu_tr]
        sampler = WeightedRandomSampler(
            torch.DoubleTensor(s_w), len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, shuffle=(sampler is None),
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    pu_counts = np.bincount(pu_labels[idx_train], minlength=2)
    pu_w      = torch.tensor(len(idx_train) / (2 * np.maximum(pu_counts, 1)),
                              dtype=torch.float32)

    meta = {
        'n_train'   : len(idx_train),
        'n_val'     : len(idx_val),
        'n_test'    : len(idx_test),
        'pu_weights': pu_w,
        'snr_mean'  : float(snr_labels[idx_train].mean()),
        'snr_std'   : float(snr_labels[idx_train].std()),
        'idx_train' : idx_train,
        'idx_val'   : idx_val,
        'idx_test'  : idx_test,
    }
    print(f"\nDataLoaders (phase={phase}): "
          f"train={meta['n_train']:,}  val={meta['n_val']:,}  "
          f"test={meta['n_test']:,}  batch={batch_size}")
    return train_loader, val_loader, test_loader, normalizer, meta


# ─── Backward-compat aliases ──────────────────────────────────────────────────

def build_dataloaders_v2(data_dir: str, batch_size: int = 64,
                          num_workers: int = 0,
                          normalizer_save_path: str = None,
                          symbol_dirs=None, **kwargs):
    """Backward-compatible wrapper used by training/phase2_trainer.py."""
    import sys, os
    root = os.path.dirname(os.path.abspath(__file__))
    su_dir  = os.path.join(root, "Secondary_User")
    new_dir = data_dir  # point to new dataset dir
    return build_dataloaders(
        secondary_user_dir   = su_dir,
        new_dataset_dir      = new_dir,
        phase                = 2,
        batch_size           = batch_size,
        num_workers          = num_workers,
        normalizer_save_path = normalizer_save_path,
        symbol_dirs          = symbol_dirs,
        **kwargs,
    )


if __name__ == '__main__':
    import sys
    root   = os.path.dirname(os.path.abspath(__file__))
    su_dir = os.path.join(root, "Secondary_User")
    nd_dir = os.path.join(root, "files-20260414T094743Z-3-001")

    tr, vl, te, norm, meta = build_dataloaders(
        secondary_user_dir=su_dir, new_dataset_dir=nd_dir,
        phase=2, batch_size=32)

    batch = next(iter(tr))
    psd_b, pu_b, mod_b, snr_b = batch
    print(f"Batch — PSD:{psd_b.shape} PU:{pu_b.shape} "
          f"Mod:{mod_b.shape} SNR:{snr_b.shape}")
    print("Pipeline OK ✓")
