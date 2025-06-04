#!/usr/bin/env python3
# plot_losses_psnr.py ---------------------------------------------------------
"""
Plot raw PSNR-vs-time up to 10 000 s and export summary metrics.

Usage
-----
python plot_losses_psnr.py \
  --pkl /abs/path/to/loss_vs_time.pkl
"""

import pickle, argparse, pathlib, csv
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------#
def main(pkl: str):
    pkl_path = pathlib.Path(pkl).expanduser()
    d = pickle.load(open(pkl_path, "rb"))

    t   = np.asarray(d["time"], dtype=np.float32)      # seconds
    psn = np.asarray(d["psnr"], dtype=np.float32)      # dB

    # ── keep only the first 10 000 s ───────────────────────────────────────────
    keep = t <= 10_000
    if not np.any(keep):
        raise RuntimeError("No data points ≤ 10 000 s!")
    t, psn = t[keep], psn[keep]

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, psn, lw=1.5, color="orange", label="Few-shot Suites")
    ax.set_xlabel("seconds")
    ax.set_ylabel("PSNR (dB)")
    ax.set_xlim(0, 10_000)
    ax.set_title("PSNR vs time")
    ax.legend()
    fig.tight_layout()
    fig.savefig("psnr_curve_0-10k_s.png", dpi=200)
    print("✓ Saved plot  → psnr_curve_0-10k_s.png")

    # ── metrics ───────────────────────────────────────────────────────────────
    metrics = {
        "final_psnr":      float(psn[-1]),
        "max_psnr":        float(psn.max()),
        "time_of_max":     float(t[psn.argmax()]),
        "mean_psnr_0-10k": float(psn.mean()),
    }

    print("\nPSNR summary (0–10 000 s):")
    for k, v in metrics.items():
        print(f"  {k:16s}: {v:6.2f}")

    # txt
    with open("psnr_metrics_0-10k_suites.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k:16s}: {v:6.2f}\n")

    # csv
    with open("psnr_metrics_0-10k_suites.csv", "w", newline="") as f:
        csv.writer(f).writerow(["metric", "value"])
        for k, v in metrics.items():
            csv.writer(f).writerow([k, v])

    print("✓ Saved metrics→ psnr_metrics_0-10k_suites.txt / psnr_metrics_0-10k_suites.csv")


# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True,
                    help="absolute path to loss_vs_time.pkl")
    main(**vars(ap.parse_args()))
