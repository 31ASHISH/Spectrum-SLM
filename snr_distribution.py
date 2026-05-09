import sys, os
sys.path.insert(0, '.')

import numpy as np
from dataset.loader import load_all_real_data

SU_DIR  = r"C:\Users\ASUS Vivo book\Desktop\Complete-Data-Science-With-Machine-Learning-And-NLP-2024-main\SDR_Data\Secondary_User"
NEW_DIR = r"C:\Users\ASUS Vivo book\Desktop\Complete-Data-Science-With-Machine-Learning-And-NLP-2024-main\SDR_Data\files-20260414T094743Z-3-001"

print("Loading data...")
psds, pu, mod, snr = load_all_real_data(SU_DIR, NEW_DIR)

# Define SNR bins
snr_bins = sorted(set(np.round(snr).astype(int)))
mod_names = {0:'BPSK', 1:'QPSK', 2:'8PSK', 3:'16QAM', 4:'DQPSK'}

print(f"\nTotal samples: {len(psds):,}")
print(f"SNR range: {snr.min():.1f} - {snr.max():.1f} dB")
print()

# Per-SNR bin breakdown
print(f"{'SNR':>5} | {'Total':>7} | {'PU=0 (Idle)':>12} | {'PU=1 (Active)':>14} | {'%PU=1':>6} | Modulation counts")
print("-" * 110)

# Group into meaningful bins
bin_edges = [0,2,4,6,8,10,12,14,16,18,20,25,30,40,50,60,70,80,90]
for i in range(len(bin_edges)-1):
    lo, hi = bin_edges[i], bin_edges[i+1]
    mask = (snr >= lo) & (snr < hi)
    n = mask.sum()
    if n == 0:
        continue
    pu0 = ((pu[mask] == 0)).sum()
    pu1 = ((pu[mask] == 1)).sum()
    pct = 100*pu1/n
    # Mod counts
    mod_counts = []
    for m in range(5):
        cnt = ((mod[mask] == m)).sum()
        if cnt > 0:
            mod_counts.append(f"{mod_names[m]}={cnt:,}")
    mod_str = "  ".join(mod_counts)
    print(f"{lo:>2}-{hi:<2}dB | {n:>7,} | {pu0:>8,} ({100*pu0/n:4.1f}%) | {pu1:>10,} ({100*pu1/n:4.1f}%) | {pct:>5.1f}% | {mod_str}")

print()
print("Overall PU distribution:")
print(f"  PU=0 (Idle)  : {(pu==0).sum():,} ({100*(pu==0).mean():.1f}%)")
print(f"  PU=1 (Active): {(pu==1).sum():,} ({100*(pu==1).mean():.1f}%)")
