import sys, os
sys.path.insert(0, '.')
import numpy as np

# 1. Full loader
print("=== Full Real Data Load ===")
from dataset.loader import load_all_real_data
p, pu, mod, snr = load_all_real_data(
    secondary_user_dir="Secondary_User",
    new_dataset_dir="files-20260414T094743Z-3-001"
)
print(f"Total: {len(p):,}  shape={p.shape}  NaN={np.isnan(p).sum()}")
print(f"PU=1: {pu.sum():,} ({pu.mean()*100:.1f}%)")
names = {0:"BPSK",1:"QPSK",2:"8PSK",3:"16QAM",4:"DQPSK"}
for m in range(5):
    print(f"  {names[m]}: {(mod==m).sum():,}")
print(f"SNR: {snr.min():.1f} - {snr.max():.1f} dB")
assert p.shape[1] == 192, "Wrong bin count!"
assert len(np.unique(mod[mod>=0])) == 5, "Missing modulation classes!"
print("Full loader: PASS\n")

# 2. Dataset analysis
print("=== Dataset Analysis ===")
from dataset.analysis import run_analysis
report = run_analysis(out_dir=".")
assert os.path.exists("dataset_report.json"), "Missing report"
assert os.path.exists("dataset_statistics.csv"), "Missing stats"
assert os.path.exists("dataset_structure.txt"), "Missing structure"
print("Analysis reports: PASS\n")

# 3. DataLoaders dry-run
print("=== DataLoader Dry-Run ===")
from spectrum_slm_dataset import build_dataloaders
tr, vl, te, norm, meta = build_dataloaders(
    secondary_user_dir="Secondary_User",
    new_dataset_dir="files-20260414T094743Z-3-001",
    phase=2, batch_size=32,
)
batch = next(iter(tr))
psd_b, pu_b, mod_b, snr_b = batch
print(f"Batch: PSD={psd_b.shape} PU={pu_b.shape} Mod={mod_b.shape} SNR={snr_b.shape}")
assert psd_b.shape[1] == 192
assert mod_b.unique().min() >= 0
print(f"Train={meta['n_train']:,} Val={meta['n_val']:,} Test={meta['n_test']:,}")
print("DataLoaders: PASS\n")

print("ALL VERIFICATIONS PASSED")
