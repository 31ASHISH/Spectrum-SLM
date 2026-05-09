import sys, os
sys.path.insert(0, '.')
import numpy as np
from dataset.loader import (
    load_binned_pth, load_log_pth, load_secondary_user, load_new_dataset
)

SU  = r"C:\Users\ASUS Vivo book\Desktop\Complete-Data-Science-With-Machine-Learning-And-NLP-2024-main\SDR_Data\Secondary_User"
NEW = r"C:\Users\ASUS Vivo book\Desktop\Complete-Data-Science-With-Machine-Learning-And-NLP-2024-main\SDR_Data\files-20260414T094743Z-3-001"

MOD_NAMES = {0:'BPSK', 1:'QPSK', 2:'8PSK', 3:'16QAM', 4:'DQPSK', -1:'Mixed'}

rows = []

# ── Source 1: Secondary_User/ ─────────────────────────────────────────────────
su_files = [
    ("psd_binned_by_snr_.pth",       "binned", -1, "Phase-1 only (mixed)"),
    ("psd_binned_by_snr_bpsk.pth",   "binned",  0, "Phase-2 training"),
    ("psd_binned_by_snr_qpsk.pth",   "binned",  1, "Phase-2 training"),
    ("psd_binned_by_snr_16qam.pth",  "binned",  3, "CORRUPTED - skipped"),
    ("psd_log_8psk.pth",             "log",     2, "Phase-2 training"),
    ("psd_log_16qam.pth",            "log",     3, "Phase-2 training"),
]

for fname, fmt, mod_id, note in su_files:
    fpath = os.path.join(SU, fname)
    size_mb = os.path.getsize(fpath) / 1e6 if os.path.exists(fpath) else 0

    if "CORRUPTED" in note or not os.path.exists(fpath):
        rows.append({
            'source': 'Secondary_User',
            'file': fname,
            'format': fmt,
            'modulation': MOD_NAMES.get(mod_id, '?'),
            'samples': 0,
            'pu0': 0, 'pu1': 0,
            'snr_min': '-', 'snr_max': '-',
            'size_mb': size_mb,
            'status': note,
        })
        continue

    if fmt == 'binned':
        p, pu, m, snr = load_binned_pth(fpath, mod_id)
    else:
        p, pu, m, snr = load_log_pth(fpath, mod_id)

    n = len(p)
    rows.append({
        'source': 'Secondary_User',
        'file': fname,
        'format': fmt,
        'modulation': MOD_NAMES.get(mod_id, '?'),
        'samples': n,
        'pu0': int((pu==0).sum()),
        'pu1': int((pu==1).sum()),
        'snr_min': f"{snr.min():.1f}" if n > 0 else '-',
        'snr_max': f"{snr.max():.1f}" if n > 0 else '-',
        'size_mb': size_mb,
        'status': note,
    })

# ── Source 2: Symbol2/ + Symbol3/ ────────────────────────────────────────────
sym_files = [
    ("Symbol2", "bpsk",  0, "dataset.pth"),
    ("Symbol2", "qpsk",  1, "dataset.pth"),
    ("Symbol2", "8psk",  2, "dataset.pth"),
    ("Symbol2", "16qam", 3, "dataset.pth"),
    ("Symbol2", "dqpsk", 4, "dataset.pth"),
    ("Symbol3", "bpsk",  0, "dataset.pth"),
    ("Symbol3", "qpsk",  1, "dataset.pth"),
    ("Symbol3", "8psk",  2, "dataset.pth"),  # no file
    ("Symbol3", "16qam", 3, "dataset.pth"),
    ("Symbol3", "dqpsk", 4, "dataset.pth"),
]

for sym, mod_folder, mod_id, fname in sym_files:
    fpath = os.path.join(NEW, sym, mod_folder, fname)
    size_mb = os.path.getsize(fpath) / 1e6 if os.path.exists(fpath) else 0

    if not os.path.exists(fpath):
        rows.append({
            'source': sym,
            'file': f"{sym}/{mod_folder}/dataset.pth",
            'format': 'log',
            'modulation': MOD_NAMES[mod_id],
            'samples': 0,
            'pu0': 0, 'pu1': 0,
            'snr_min': '-', 'snr_max': '-',
            'size_mb': 0,
            'status': 'FILE NOT FOUND',
        })
        continue

    p, pu, m, snr = load_log_pth(fpath, mod_id)
    n = len(p)
    if n == 0:
        status = 'SKIPPED (wrong format/bins)'
    else:
        status = 'Phase-2 training'

    rows.append({
        'source': sym,
        'file': f"{sym}/{mod_folder}/dataset.pth",
        'format': 'log',
        'modulation': MOD_NAMES[mod_id],
        'samples': n,
        'pu0': int((pu==0).sum()) if n > 0 else 0,
        'pu1': int((pu==1).sum()) if n > 0 else 0,
        'snr_min': f"{snr.min():.1f}" if n > 0 else '-',
        'snr_max': f"{snr.max():.1f}" if n > 0 else '-',
        'size_mb': size_mb,
        'status': status,
    })

# ── Print master table ────────────────────────────────────────────────────────
print("=" * 130)
print("  SPECTRUM-SLM — MASTER DATASET TABLE")
print("=" * 130)
print(f"{'#':<3} {'Source':<16} {'File':<42} {'Mod':<8} {'Samples':>9} {'PU=0':>8} {'PU=1':>8} {'SNR Min':>8} {'SNR Max':>8} {'MB':>7}  Status")
print("-" * 130)

total_used = 0
total_all  = 0
for i, r in enumerate(rows):
    flag = '' if 'training' in r['status'] or 'only' in r['status'] else '[!]'
    print(f"{i+1:<3} {r['source']:<16} {r['file']:<42} {r['modulation']:<8} "
          f"{r['samples']:>9,} {r['pu0']:>8,} {r['pu1']:>8,} "
          f"{r['snr_min']:>8} {r['snr_max']:>8} {r['size_mb']:>7.1f}  {flag}{r['status']}")
    total_all += r['samples']
    if 'training' in r['status'] or 'only' in r['status']:
        total_used += r['samples']

print("-" * 130)
print(f"{'TOTAL USABLE':>73} {total_used:>9,}")
print(f"{'TOTAL ALL FILES':>73} {total_all:>9,}")
print("=" * 130)

# Per-modulation summary
print()
print("=" * 70)
print("  PER-MODULATION SUMMARY (Phase-2 Training Data)")
print("=" * 70)
mod_totals = {0:0, 1:0, 2:0, 3:0, 4:0}
for r in rows:
    if 'training' in r['status']:
        for mid, mname in MOD_NAMES.items():
            if mid >= 0 and mname == r['modulation']:
                mod_totals[mid] += r['samples']

print(f"  {'Modulation':<10} {'Samples':>10}  {'Share':>7}  Sources")
print("-" * 70)
grand = sum(mod_totals.values())
for mid, cnt in mod_totals.items():
    pct = 100*cnt/grand if grand > 0 else 0
    srcs = [r['file'].replace('dataset.pth','').rstrip('/') for r in rows
            if 'training' in r['status'] and MOD_NAMES.get(mid) == r['modulation']]
    print(f"  {MOD_NAMES[mid]:<10} {cnt:>10,}  {pct:>6.1f}%  {', '.join(srcs)}")
print(f"  {'TOTAL':<10} {grand:>10,}  100.0%")
print("=" * 70)

# Known results from Kaggle run
print()
print("=" * 70)
print("  FINAL MODEL PERFORMANCE (from Kaggle run)")
print("=" * 70)
results = [
    ("PU Detection Accuracy",  "93.32%", "Excellent"),
    ("PU F1 Score (binary)",   "0.9470", "Excellent"),
    ("PU ROC-AUC",             "0.9809", "Near-perfect"),
    ("PU PR-AUC",              "0.9912", "Excellent"),
    ("Low-SNR Acc (<8dB)",     "91.15%", "Strong"),
    ("Low-SNR F1 (<8dB)",      "0.7655", "Good"),
    ("Modulation Accuracy",    "75.13%", "Moderate (3 files missing)"),
    ("Modulation Macro-F1",    "0.7230", "Moderate"),
    ("SNR MAE",                "0.371 dB","Research-grade"),
    ("SNR RMSE",               "0.636 dB","Very good"),
    ("SNR R-squared",          "0.9321", "Excellent"),
]
print(f"  {'Metric':<30} {'Value':>12}  Verdict")
print("-" * 70)
for m, v, verdict in results:
    print(f"  {m:<30} {v:>12}  {verdict}")
print("=" * 70)
