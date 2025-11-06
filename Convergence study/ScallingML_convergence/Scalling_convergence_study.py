#!/usr/bin/env python
import os
import sys
import csv
from math import log10
from decimal import Decimal, getcontext

import torch
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

from model_scalling_fnn import load_ff_pipelines_model

# ---------------- Decimal precision ----------------
getcontext().prec = 70   # enough for 64 decimal digits output

# ---------------- Paths ----------------
# Text data + summary for the ML scaling case
data_path  = r"C:\Git\Algoim_mimic\Convergence study\ScallingML_convergence\text_data\cut_subcell_polynomials_p1_data.txt"
summary_path = r"C:\Git\Algoim_mimic\Convergence study\ScallingML_convergence\text_data\subcell_classification_summary.txt"

# Model paths
model_dir = r"C:\Git\Algoim_mimic\ML Models\Scalling_FNN\Model"
weights_path = os.path.join(model_dir, "fnn_model_weights_v6.pth")

# Base folder for outputs (same folder as ScallingML_convergence)
base_folder = r"C:\Git\Algoim_mimic\Convergence study\ScallingML_convergence"
os.makedirs(base_folder, exist_ok=True)

out_area_path = os.path.join(base_folder, "subcell_areas_ml.txt")
conv_csv_path = os.path.join(base_folder, "convergence_rel_error_vs_h_ml.csv")
fig_path      = os.path.join(base_folder, "convergence_rel_error_vs_h_ml.png")

# ---------------- Quadrature base weights ----------------
weightRefs = np.array([
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.3626837833783620,
    0.3626837833783620,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763
], dtype=np.float64)

# 2D tensor–product weights on [-1, 1]^2
w = np.outer(weightRefs, weightRefs)   # shape (8, 8)

# ---------------- Torch setup ----------------
torch.set_default_dtype(torch.float64)
device = torch.device("cpu")


def load_ml_weights_map():
    """
    Run the scaling FNN on all entries in cut_subcell_polynomials_p1_data.txt
    and return a dict: id_str -> Decimal(sum_of_scaled_weights).
    """
    if not os.path.isfile(data_path):
        print(f"[error] data file not found: {data_path}")
        sys.exit(1)

    # Load model
    model = load_ff_pipelines_model()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=torch.float64)
    model.eval()

    ids = []
    exp_x_all = []
    exp_y_all = []
    coeffs_all = []

    # Read polynomial data 
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) != 5:
                print(f"[warn] skipping malformed line: {line}")
                continue

            _, id_str, expx_str, expy_str, coeff_str = parts

            exp_x = [int(v) for v in expx_str.split(",") if v.strip()]
            exp_y = [int(v) for v in expy_str.split(",") if v.strip()]
            coeffs = [Decimal(v) for v in coeff_str.split(",") if v.strip()]

            ids.append(id_str)
            exp_x_all.append(exp_x)
            exp_y_all.append(exp_y)
            coeffs_all.append(coeffs)

    print(f"Total records in data file: {len(ids)}")

    weights_map = {}  # id_str 

    with torch.no_grad():
        for i, id_str in enumerate(ids):
            exp_x = exp_x_all[i]
            exp_y = exp_y_all[i]
            coeffs = coeffs_all[i]

            coeffs_float = [float(c) for c in coeffs]

            exp_x_t = torch.tensor([exp_x], dtype=torch.float64, device=device)
            exp_y_t = torch.tensor([exp_y], dtype=torch.float64, device=device)
            coeff_t = torch.tensor([coeffs_float], dtype=torch.float64, device=device)

            pred_scale_x, pred_scale_y = model(exp_x_t, exp_y_t, coeff_t)

            pred_x = pred_scale_x.squeeze(0).cpu().numpy()  # shape (8,)
            pred_y = pred_scale_y.squeeze(0).cpu().numpy()  # shape (8,)

            scale_mat = pred_x[:, None] * pred_y[None, :]   # shape (8, 8)
            scaled = scale_mat * w                           # scaled weights

            # sum of all 64 scaled weights for this subcell
            sum_scaled = float(scaled.sum())
            # convert via str to preserve decimal precision reasonably well
            weights_map[id_str] = Decimal(str(sum_scaled))

    print(f"Built ML weights map for {len(weights_map)} ids.")
    return weights_map


def parse_summary(summary_file):
    """
    Generator over the summary file:
    yields (n, subno, id_str, status, is_count_line)
    """
    with open(summary_file, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            parts = s.split(";")
            if parts[0].lower() == "n":
                continue
            # count lines: e.g. "2;counts;inside=2;outside=0;partial=2"
            if len(parts) >= 2 and parts[1] == "counts":
                yield (int(parts[0]), None, None, None, True)
                continue
            if len(parts) < 4:
                continue
            n = int(parts[0])
            subno = int(parts[1])
            id_str = parts[2]
            status = parts[3]
            yield (n, subno, id_str, status, False)


def true_area():
    """
    Same analytic true area as in your Algoim convergence script.
    """
    a = mp.mpf("-0.018251")
    b = mp.mpf("-0.56895")
    c = mp.mpf("-0.11769")
    d = mp.mpf("0.0072423")

    one = mp.mpf("1")
    two = mp.mpf("2")
    neg_one = mp.mpf("-1")

    def y_thresh(x):
        # Solve a*x + b*y + c + d*x*y = 0 for y
        return -(a*x + c) / (b + d*x)

    def length_at_x(x):
        denom = b + d*x
        # If the line is horizontal (denom == 0): either full height or zero height
        if denom == 0:
            return two if (a*x + c) <= 0 else mp.mpf("0")
        yt = y_thresh(x)
        if denom > 0:
            lower, upper = neg_one, (yt if yt < one else one)
        else:
            lower, upper = (yt if yt > neg_one else neg_one), one
        length = upper - lower
        if length < 0:
            length = mp.mpf("0")
        if length > two:
            length = two
        return length

    mp.mp.dps = 80  # high precision for integration
    x0 = -b / d
    if -1 < x0 < 1:
        pieces = [-1, x0, 1]
    else:
        pieces = [-1, 1]

    A = mp.mpf("0")
    for s, t in zip(pieces[:-1], pieces[1:]):
        A += mp.quad(lambda x: length_at_x(x), [s, t])

    return Decimal(str(A))


def main():
    if not os.path.isfile(summary_path):
        print(f"[error] summary file not found: {summary_path}")
        sys.exit(1)

    # 1) Run ML model and build id -> sum_of_scaled_weights map
    weights_map = load_ml_weights_map()

    # 2) Use summary to accumulate areas for each n
    rows_out = []          # (n, subno, id, status, area: Decimal)
    per_n_total = {}       # n -> total area (Decimal)
    per_n_counts = {}      # n -> dict(status->count used)

    for n, subno, id_str, status, is_count in parse_summary(summary_path):
        if is_count:
            # We recompute totals ourselves, so ignore these lines
            continue

        status_l = status.lower()
        if status_l == "outside":
            # ignore completely
            continue

        nn = Decimal(n) * Decimal(n)  # n^2 as Decimal

        if status_l == "inside":
            # full subcell area in [-1,1]^2 is 4 / n^2
            area = Decimal(4) / nn

        elif status_l == "partial":
            if id_str not in weights_map:
                print(f"[warn] missing ML weights for partial id={id_str} (n={n}, subno={subno}); setting area to 0.0")
                area = Decimal(0)
            else:
                # weights_map[id_str] ~ integral over ref [-1,1]^2
                # physical subcell area contribution = weights_sum / n^2
                area = weights_map[id_str] / nn

        else:
            print(f"missing data")
            continue

        rows_out.append((n, subno, id_str, status_l, area))
        per_n_total[n] = per_n_total.get(n, Decimal(0)) + area
        per_n_counts.setdefault(n, {"inside": 0, "partial": 0})
        per_n_counts[n][status_l] = per_n_counts[n].get(status_l, 0) + 1

    # 3) Write detailed per-subcell results
    with open(out_area_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["n", "subno", "id", "status", "area"])
        # sort by n then subno for readability
        for row in sorted(rows_out, key=lambda t: (t[0], t[1])):
            n, subno, id_str, status_l, area = row
            writer.writerow([n, subno, id_str, status_l, f"{area:.64f}"])
        # add per-n totals at the end
        writer.writerow([])
        writer.writerow(["n", "total_area", "inside_used", "partial_used"])
        for n in sorted(per_n_total.keys()):
            c = per_n_counts.get(n, {})
            writer.writerow([n, f"{per_n_total[n]:.64f}", c.get("inside", 0), c.get("partial", 0)])

    print(f"\nWrote ML-based subcell areas to:\n  {out_area_path}")
    for n in sorted(per_n_total.keys()):
        c = per_n_counts.get(n, {})
        print(f"n={n}: total_area={per_n_total[n]:.64f}  (inside_used={c.get('inside',0)}, partial_used={c.get('partial',0)})")

    # 4) Convergence vs true area
    A_true = true_area()
    if A_true == 0:
        print("[warn] True area is zero; relative error undefined. Skipping error plot.")
        return

    ns = sorted(per_n_total.keys())
    totals = [per_n_total[n] for n in ns]

    # Element size (cell width) in [-1,1]: h = 2/n
    hs = [2.0 / float(n) for n in ns]

    # Relative error: |A_n - A_true| / |A_true|
    rel_err_dec = [abs(T - A_true) / abs(A_true) for T in totals]  # Decimal
    rel_err = [float(e) for e in rel_err_dec]                      # float for plotting

    # ---- Print per-n relative errors ----
    print("\nPer-n convergence data (ML):")
    print(" n   h=2/n           total_area (Decimal, 30dp)           rel_error (float)")
    for n, h, T, e_dec, e_float in zip(ns, hs, totals, rel_err_dec, rel_err):
        print(
            f"{n:2d}  {h: .6e}  "
            f"{T:.30f}  "
            f"{e_float: .6e}"
        )

    # 5) Save CSV with h and errors
    with open(conv_csv_path, "w", encoding="utf-8", newline="") as fc:
        wcsv = csv.writer(fc, delimiter=";")
        wcsv.writerow(["n", "h=2/n", "total_area", "true_area", "relative_error"])
        for n, h, T, e_dec in zip(ns, hs, totals, rel_err_dec):
            wcsv.writerow([n,
                           f"{h:.16e}",
                           f"{T:.64f}",
                           f"{A_true:.64f}",
                           f"{float(e_dec):.16e}"])

    # 6) Log–log convergence plot (same style as Algoim one)
    plt.figure()
    plt.loglog(hs, rel_err, marker="o")
    plt.xlabel("Element size h = 2/n")
    plt.ylabel("Relative error")
    plt.title("Convergence of area: relative error vs element size h")
    plt.grid(True, which="both")

    # slope/order estimate using first and last points
    if len(hs) >= 2 and rel_err[0] > 0 and rel_err[-1] > 0:
        s = (log10(rel_err[-1]) - log10(rel_err[0])) / (log10(hs[-1]) - log10(hs[0]))
        print(f"\nEstimated order (log10 error vs log10 h): {s:.3f}")

        xg = np.array([hs[-1], hs[0]], dtype=float)  # from large to small h for visibility
        C = rel_err[-1] / (hs[-1] ** s)
        yg = C * (xg ** s)
        plt.loglog(xg, yg, linestyle="--", label=f"slope ≈ {s:.2f}")
        plt.legend()

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # Final info
    print(f"\nTrue area (Decimal, 64dp): {A_true:.64f}")
    print(f"Wrote ML convergence CSV:\n  {conv_csv_path}")
    print(f"Wrote ML convergence plot:\n  {fig_path}")


if __name__ == "__main__":
    main()
