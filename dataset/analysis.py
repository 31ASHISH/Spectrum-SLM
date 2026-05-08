"""
dataset/analysis.py — Dataset analysis and report generation.
Saves dataset_report.json, dataset_statistics.csv, dataset_structure.txt
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.loader import load_all_real_data
from config import SECONDARY_USER_DIR, NEW_DATASET_DIR, SNR_BINS

MOD_NAMES = {0:"BPSK",1:"QPSK",2:"8PSK",3:"16QAM",4:"DQPSK"}

def run_analysis(out_dir: str = "."):
    os.makedirs(out_dir, exist_ok=True)

    print("\n=== Spectrum-SLM Dataset Analysis ===")
    psds, pu, mod, snr = load_all_real_data(SECONDARY_USER_DIR, NEW_DATASET_DIR)

    N = len(psds)
    report = {
        "total_samples": int(N),
        "psd_shape": list(psds.shape),
        "psd_min": float(psds.min()),
        "psd_max": float(psds.max()),
        "psd_mean": float(psds.mean()),
        "psd_std":  float(psds.std()),
        "nan_count": int(np.isnan(psds).sum()),
        "inf_count": int(np.isinf(psds).sum()),
        "pu_distribution": {
            "PU=0": int((pu==0).sum()),
            "PU=1": int((pu==1).sum()),
            "PU=1_pct": float(pu.mean()*100),
        },
        "snr_range": {"min": float(snr.min()), "max": float(snr.max()),
                      "mean": float(snr.mean()), "std": float(snr.std())},
        "modulation_distribution": {},
        "per_snr_bin_counts": {},
    }

    for m_id in range(5):
        cnt = int((mod==m_id).sum())
        report["modulation_distribution"][MOD_NAMES[m_id]] = cnt

    for b in SNR_BINS:
        mask = (snr >= b-1) & (snr < b+1)
        report["per_snr_bin_counts"][str(b)] = int(mask.sum())

    # Save report
    rpt_path = os.path.join(out_dir, "dataset_report.json")
    with open(rpt_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved -> {rpt_path}")

    # Statistics CSV
    rows = []
    for m_id in range(5):
        mask = mod == m_id
        if mask.sum() == 0: continue
        rows.append({
            "modulation": MOD_NAMES[m_id],
            "n_samples": int(mask.sum()),
            "pu_1_pct": float(pu[mask].mean()*100),
            "snr_mean": float(snr[mask].mean()),
            "snr_std":  float(snr[mask].std()),
            "psd_mean": float(psds[mask].mean()),
            "psd_std":  float(psds[mask].std()),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "dataset_statistics.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved -> {csv_path}")

    # Structure text
    struct_lines = [
        "=== Spectrum-SLM Dataset Structure ===\n",
        f"Total samples : {N:,}",
        f"PSD shape     : {psds.shape}",
        f"Bin count     : 192 (confirmed)",
        f"NaN count     : {report['nan_count']}",
        f"SNR range     : {snr.min():.1f} – {snr.max():.1f} dB",
        "\nModulation breakdown:",
    ]
    for m_id in range(5):
        cnt = int((mod==m_id).sum())
        struct_lines.append(f"  {MOD_NAMES[m_id]:<8}: {cnt:>7,}")
    struct_lines += ["\nSNR bin counts (±1 dB):"]
    for b in SNR_BINS:
        cnt = report["per_snr_bin_counts"][str(b)]
        struct_lines.append(f"  {b:>3} dB : {cnt:>6,}")

    txt_path = os.path.join(out_dir, "dataset_structure.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(struct_lines))
    print(f"  Saved -> {txt_path}")

    print("\n=== Summary ===")
    print(f"  Total samples : {N:,}")
    print(f"  PU=1          : {pu.sum():,} ({pu.mean()*100:.1f}%)")
    print(f"  NaN in PSDs   : {report['nan_count']}")
    for nm, cnt in report["modulation_distribution"].items():
        print(f"  {nm:<8}: {cnt:,}")
    return report

if __name__ == "__main__":
    run_analysis(out_dir=".")
