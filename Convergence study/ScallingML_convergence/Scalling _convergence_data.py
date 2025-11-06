import os
import uuid
from math import comb
import numpy as np

output_folder = r"C:\Git\Algoim_mimic\Convergence study\ScallingML_convergence\text_data"
os.makedirs(output_folder, exist_ok=True)
out_path_polys   = os.path.join(output_folder, "cut_subcell_polynomials_p1_data.txt")
out_path_summary = os.path.join(output_folder, "subcell_classification_summary.txt")

poly_line = "0a96c8d6dc0ed2d3d4032f3df7e72eef;0,1,0,1;0,0,1,1;-0.11769,-0.018251,-0.56895,0.0072423"

def _parse_csv_int(s):   
    return np.array([int(v) for v in s.split(',') if v.strip()!=''], dtype=int)

def _parse_csv_float(s):
    return np.array([float(v) for v in s.split(',') if v.strip()!=''], dtype=np.float64)

def parse_poly_line(line: str):
    parts = [p.strip() for p in line.split(';')]
    if len(parts) != 4:
        raise ValueError(f"Bad POLY_LINE format: {line}")
    exps_x = _parse_csv_int(parts[1])
    exps_y = _parse_csv_int(parts[2])
    coeffs = _parse_csv_float(parts[3])
    if not (len(exps_x) == len(exps_y) == len(coeffs)):
        raise ValueError("exps_x, exps_y, coeffs must have equal length")
    return exps_x, exps_y, coeffs

EXPS_X, EXPS_Y, COEFFS = parse_poly_line(poly_line)

def phi_eval(x, y, exps_x=EXPS_X, exps_y=EXPS_Y, coeffs=COEFFS):
    val = 0.0
    X = np.asarray(x); Y = np.asarray(y)
    for a, b, c in zip(exps_x, exps_y, coeffs):
        val += c * (X ** a) * (Y ** b)
    return val

def classify_cell(ox, oy, h):
    corners = [
        (ox - h, oy - h), (ox - h, oy + h),
        (ox + h, oy - h), (ox + h, oy + h)
    ]
    phi_c = [phi_eval(x, y) for (x, y) in corners]

    if all(p < 0.0 for p in phi_c):
        return 'inside'

    probes = [(ox, oy), (ox - h, oy), (ox + h, oy), (ox, oy - h), (ox, oy + h)]
    phi_p = [phi_eval(x, y) for (x, y) in probes]
    if all(p > 0.0 for p in phi_c) and all(p > 0.0 for p in phi_p):
        return 'outside'

    return 'partial'

def expand_to_subcell_monomials(exps_x, exps_y, coeffs, ox, oy, h):
    terms = {}
    for a, b, c in zip(exps_x, exps_y, coeffs):
        for k in range(a + 1):
            for l in range(b + 1):
                coeff_kl = (
                    c
                    * comb(a, k) * (ox ** (a - k)) * (h ** k)
                    * comb(b, l) * (oy ** (b - l)) * (h ** l)
                )
                terms[(k, l)] = terms.get((k, l), 0.0) + coeff_kl

    items = sorted(terms.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    k_list, l_list, c_list = [], [], []
    for (k, l), val in items:
        if val != 0.0:
            k_list.append(k); l_list.append(l); c_list.append(val)
    return k_list, l_list, c_list

def list_to_csv(vals, fmt="{:.17g}"):
    if not vals:
        return ""
    if isinstance(vals[0], (int, np.integer)):
        return ",".join(str(int(v)) for v in vals)
    return ",".join(fmt.format(v) for v in vals)

def random_hex_id():
    return uuid.uuid4().hex[:32]

def cut_subcell_polynomials():
    levels = [1, 2, 4, 8]
    poly_lines = []
    summary_lines = []
    count = 0

    for n in levels:
        h = 1.0 / n
        centers = np.linspace(-1 + h, 1 - h, n)

        inside_cnt = outside_cnt = partial_cnt = 0

        for r, oy in enumerate(centers):
            for c, ox in enumerate(centers):
                subno = r * n + c + 1  # 1..n^2
                cell_id = random_hex_id()
                status = classify_cell(ox, oy, h)

                summary_lines.append(f"{n};{subno};{cell_id};{status}")

                if status == 'inside':
                    inside_cnt += 1
                elif status == 'outside':
                    outside_cnt += 1
                else:
                    partial_cnt += 1
                    k_list, l_list, coeff_list = expand_to_subcell_monomials(
                        EXPS_X, EXPS_Y, COEFFS, ox, oy, h
                    )
                    count += 1
                    exp_x_csv = list_to_csv(k_list)
                    exp_y_csv = list_to_csv(l_list)
                    coeff_csv = list_to_csv(coeff_list)

                    poly_lines.append(f"{count};{cell_id};{exp_x_csv};{exp_y_csv};{coeff_csv}")

        summary_lines.append(f"{n};counts;inside={inside_cnt};outside={outside_cnt};partial={partial_cnt}")

    # Write files
    with open(out_path_polys, "w", encoding="utf-8") as f:
        for line in poly_lines:
            f.write(line + "\n")

    with open(out_path_summary, "w", encoding="utf-8") as f:
        f.write("n;subno;id;status\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"Saved {len(poly_lines)} cut subcell polynomials to:\n  {out_path_polys}")
    print(f"Saved subcell classification summary (with random 32-char IDs) to:\n  {out_path_summary}")

if __name__ == "__main__":
    cut_subcell_polynomials()
