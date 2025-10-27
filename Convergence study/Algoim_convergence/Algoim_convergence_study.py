#!/usr/bin/env python
import os
import sys
import csv
from decimal import Decimal, getcontext
import mpmath as mp
import matplotlib.pyplot as plt
from math import log10


getcontext().prec = 70  


base_folder  = r"C:\Git\Algoim_mimic\Convergence study\Algoim_convergence"
summary_path = os.path.join(base_folder, "subcell_classification_summary.txt")


weights_files = [
    os.path.join(base_folder, "cut_subcell_polynomials_p1_output_8.txt"),
]

out_path = os.path.join(base_folder, "subcell_areas.txt")


def read_weights_map(paths):
    wmap = {}
    for p in paths:
        if not os.path.isfile(p):
            print(f"[warn] weights file not found: {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            header_checked = False
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split(";")
                # header row?
                if not header_checked:
                    header_checked = True
                    if parts[0].lower() in ("number", "num", "idx"):
                        # skip header
                        continue
                if len(parts) < 5:
                    # some files may have fewer fields
                    continue
                _, id_str, *rest = parts
                weights_csv = rest[-1]  # last field is weights
                try:
                    weights = [Decimal(x) for x in weights_csv.split(",") if x.strip() != ""]
                    wmap[id_str] = sum(weights, start=Decimal(0))
                except Exception as e:
                    print(f"[warn] failed parsing weights for id={id_str} in {p}: {e}")
    return wmap


def parse_summary(summary_file):

    with open(summary_file, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            parts = s.split(";")
            if parts[0].lower() == "n":
                continue
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
    # Potential discontinuity (denominator zero) at x0 = -b/d
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

    weights_map = read_weights_map(weights_files)

    rows_out = []  # (n, subno, id, status, area: Decimal)
    per_n_total = {}  # n -> total area (Decimal sum of included cells)
    per_n_counts = {} # n -> dict(status->count used)

    for n, subno, id_str, status, is_count in parse_summary(summary_path):
        if is_count:
            continue  # ignore the count lines; we’ll recompute totals

        if status == "outside":
            # skip entirely
            continue

        nn = Decimal(n) * Decimal(n)  # n^2 as Decimal

        # inside or partial
        if status == "inside":
            # full subcell area = 4 / n^2 (domain [-1,1]^2)
            area = Decimal(4) / nn
        elif status == "partial":
            if id_str not in weights_map:
                print(f"[warn] missing weights for partial id={id_str} (n={n}, subno={subno}); setting area to 0.0")
                area = Decimal(0)
            else:
                # weights_map[id_str] is already Decimal
                area = weights_map[id_str] / nn
        else:
            # unknown status; ignore for safety
            continue

        rows_out.append((n, subno, id_str, status, area))
        per_n_total[n] = per_n_total.get(n, Decimal(0)) + area
        per_n_counts.setdefault(n, {"inside":0, "partial":0})
        per_n_counts[n][status] = per_n_counts[n].get(status, 0) + 1

    # write detailed per-subcell results
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["n", "subno", "id", "status", "area"])
        # sort by n then subno for readability
        for row in sorted(rows_out, key=lambda t: (t[0], t[1])):
            n, subno, id_str, status, area = row
            writer.writerow([n, subno, id_str, status, f"{area:.64f}"])
        # add per-n totals at the end
        writer.writerow([])
        writer.writerow(["n", "total_area", "inside_used", "partial_used"])
        for n in sorted(per_n_total.keys()):
            c = per_n_counts.get(n, {})
            writer.writerow([n, f"{per_n_total[n]:.64f}", c.get("inside",0), c.get("partial",0)])

    print(f"Wrote subcell areas to:\n  {out_path}")
    for n in sorted(per_n_total.keys()):
        c = per_n_counts.get(n, {})
        print(f"n={n}: total_area={per_n_total[n]:.64f}  (inside_used={c.get('inside',0)}, partial_used={c.get('partial',0)})")

    A_true = true_area()
    if A_true == 0:
        print("[warn] True area is zero; relative error would be undefined. Skipping error plot.")
        return

    # Build arrays sorted by n
    ns = sorted(per_n_total.keys())
    totals = [per_n_total[n] for n in ns]

    # Element size (actual subcell width on [-1,1]): h = 2/n
    hs = [2.0 / float(n) for n in ns]

    # Relative error: |A_n - A_true| / |A_true|
    rel_err_dec = [abs(T - A_true) / abs(A_true) for T in totals]
    rel_err = [float(e) for e in rel_err_dec]  # for plotting

    # Save a small CSV for inspection (now includes h)
    conv_csv = os.path.join(base_folder, "convergence_rel_error_vs_h.csv")
    with open(conv_csv, "w", encoding="utf-8", newline="") as fc:
        w = csv.writer(fc, delimiter=";")
        w.writerow(["n", "h=2/n", "total_area", "true_area", "relative_error"])
        for n, h, T, e in zip(ns, hs, totals, rel_err_dec):
            w.writerow([n, f"{h:.16e}", f"{T:.64f}", f"{A_true:.64f}", f"{e:.16e}"])

    # Single plot: log–log of relative error vs element size h
    plt.figure()
    plt.loglog(hs, rel_err, marker="o")
    plt.xlabel("Element size h = 2/n")
    plt.ylabel("Relative error")
    plt.title("Convergence of area: relative error vs element size h")
    plt.grid(True, which="both")

    # Optional: slope/order estimate on error vs h (should be ~ +p)
    if len(hs) >= 2 and rel_err[0] > 0 and rel_err[-1] > 0:
        # Use first and last points
        s = (log10(rel_err[-1]) - log10(rel_err[0])) / (log10(hs[-1]) - log10(hs[0]))
        print(f"Estimated order (log10 error vs log10 h): {s:.3f}")
        import numpy as np
        xg = np.array([hs[-1], hs[0]], dtype=float)  # from small to large h for visibility
        C = rel_err[-1] / (hs[-1] ** s)
        yg = C * (xg ** s)
        plt.loglog(xg, yg, linestyle="--", label=f"slope ≈ {s:.2f}")

    plt.legend()
    fig_path = os.path.join(base_folder, "convergence_rel_error_vs_h.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"\nTrue area (Decimal, 64dp): {A_true:.64f}")
    print(f"Wrote convergence CSV:\n  {conv_csv}")
    print(f"Wrote plot:\n  {fig_path}")


if __name__ == "__main__":
    main()
