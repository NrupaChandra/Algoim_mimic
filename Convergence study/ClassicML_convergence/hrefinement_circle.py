#!/usr/bin/env python
# Convergence study for a specific polynomial level set (no circle).
# Calls FNN only on partial cells; fully inside/outside cells are integrated analytically.
# After model inference, nodes are sorted into a consistent 8x8 raster and
# weights are scaled from Gauss–Legendre tensor-product references.

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from math import comb

from model_fnn import load_ff_pipelines_model
import utilities  # must provide utilities.compute_integration(x, y, w, f)

# ============================================================
# 0) Config: model + output
# ============================================================
device = torch.device('cpu')

model_path    = r"C:\Git\Algoim_mimic\ML Models\Classic_FNN\Model\fnn_model_weights_v6.pth"
output_folder = r"C:\Git\Algoim_mimic\Convergence study"
os.makedirs(output_folder, exist_ok=True)

model = load_ff_pipelines_model(weights_path=None)
# Use weights_only=True to avoid pickle warning and future breakage
state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ============================================================
# 1) Polynomial spec
# ============================================================
POLY_LINE = "0a96c8d6dc0ed2d3d4032f3df7e72eef;0,1,0,1;0,0,1,1;-0.11769,-0.018251,-0.56895,0.0072423"

def _parse_csv_int(s):   return np.array([int(v) for v in s.split(',') if v.strip()!=''], dtype=int)
def _parse_csv_float(s): return np.array([float(v) for v in s.split(',') if v.strip()!=''], dtype=np.float64)

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

EXPS_X, EXPS_Y, COEFFS = parse_poly_line(POLY_LINE)
NORMALIZE_POWER = 0
INSIDE_TOL = 1e-8

# ============================================================
# 1.5) Sorting + Weight Scaling Utilities (in-memory)
# ============================================================
def sort_predicted_nodes_weights(x, y, w, per_line=8):
    """
    Sort predicted nodes (x, y, w) into consistent row-major 8x8 raster order:
    rows bottom->top (by y), columns left->right (by x within each row).
    Inputs: 1D numpy arrays; outputs: 1D numpy arrays.
    """
    n = len(x)
    assert n == len(y) == len(w), "x,y,w lengths must match"
    assert n % per_line == 0, "Number of nodes must be a multiple of per_line"
    idx = list(range(n))
    # primary sort by y (ascending), then by x (ascending)
    idx.sort(key=lambda i: (y[i], x[i]))
    # within each row of 'per_line', resort strictly by x to ensure left->right
    for row_start in range(0, n, per_line):
        row = idx[row_start:row_start+per_line]
        row.sort(key=lambda i: x[i])
        idx[row_start:row_start+per_line] = row
    x_sorted = np.asarray([x[i] for i in idx], dtype=float)
    y_sorted = np.asarray([y[i] for i in idx], dtype=float)
    w_sorted = np.asarray([w[i] for i in idx], dtype=float)
    return x_sorted, y_sorted, w_sorted

def scale_weights(x_sorted, y_sorted, w_sorted):
    """
    Scale weights using Gauss–Legendre tensor-product references and the
    measured span in x (per row) and y (per column). Returns flat (64,) weights.
    """
    weightRefs = np.array([0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
                           0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
                           0.2223810344533745, 0.1012285362903763], dtype=float)
    deltaRef = 0.9602898564975363 * 2.0  # fixed span from Gauss–Legendre

    nx = x_sorted.reshape(8, 8)  # rows; left->right
    ny = y_sorted.reshape(8, 8)  # rows; left->right

    xdeltas = nx[:, 7] - nx[:, 0]  # per-row span in x
    ydeltas = ny[7, :] - ny[0, :]  # per-col span in y

    xscale = xdeltas / deltaRef    # shape (8,)
    yscale = ydeltas / deltaRef    # shape (8,)

    w_ref = np.outer(weightRefs, weightRefs)  # (8,8)
    w_scaled = (xscale[:, None] * yscale[None, :]) * w_ref
    return w_scaled.ravel()

def postprocess_nodes_weights(ox, oy, h, xn, yn, w):
    """
    After model inference:
      - map to physical coords (numpy),
      - sort nodes into consistent 8x8 raster,
      - scale weights from GL tensor refs,
      - return BOTH flat numpy arrays (for plotting) AND batched torch tensors (for integration):
        x_b, y_b, w_b have shape (1, 64).
    """
    x_np = (ox + h * xn).cpu().numpy().ravel()
    y_np = (oy + h * yn).cpu().numpy().ravel()
    w_np = w.cpu().numpy().ravel()

    x_sorted, y_sorted, w_sorted = sort_predicted_nodes_weights(x_np, y_np, w_np, per_line=8)
    w_scaled = scale_weights(x_sorted, y_sorted, w_sorted)

    # batched tensors for utilities.compute_integration (expects dim=1 to exist)
    x_b = torch.tensor(x_sorted, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 64)
    y_b = torch.tensor(y_sorted, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 64)
    w_b = torch.tensor(w_scaled, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 64)

    return x_b, y_b, w_b, x_sorted, y_sorted, w_scaled

# ============================================================
# 2) Level-set utilities
# ============================================================
def phi_eval(x, y, exps_x=EXPS_X, exps_y=EXPS_Y, coeffs=COEFFS):
    """Evaluate φ(x,y) = Σ c * x^a y^b (supports scalars or numpy arrays)."""
    val = 0.0
    X = np.asarray(x)
    Y = np.asarray(y)
    for a, b, c in zip(exps_x, exps_y, coeffs):
        val += c * (X ** a) * (Y ** b)
    return val

def classify_cell(ox, oy, h):
    # 4 corners
    corners = [
        (ox - h, oy - h), (ox - h, oy + h),
        (ox + h, oy - h), (ox + h, oy + h)
    ]
    phi_c = [phi_eval(x, y) for (x, y) in corners]

    # definitely inside
    if all(p <= -INSIDE_TOL for p in phi_c):
        return 'inside'

    # conservative outside check (corners + probes must be strictly positive)
    probes = [(ox, oy), (ox - h, oy), (ox + h, oy), (ox, oy - h), (ox, oy + h)]
    phi_p = [phi_eval(x, y) for (x, y) in probes]
    if all(p >= INSIDE_TOL for p in phi_c) and all(p >= INSIDE_TOL for p in phi_p):
        return 'outside'

    return 'partial'

def expand_to_subcell_monomials(exps_x, exps_y, coeffs, ox, oy, h, normalize_power=0):
    terms = {}  # (k,l) -> coeff
    for a, b, c in zip(exps_x, exps_y, coeffs):
        for k in range(a + 1):
            for l in range(b + 1):
                coeff_kl = (
                    c
                    * comb(a, k) * (ox ** (a - k)) * (h ** k)
                    * comb(b, l) * (oy ** (b - l)) * (h ** l)
                )
                terms[(k, l)] = terms.get((k, l), 0.0) + coeff_kl

    if normalize_power != 0:
        s = (h ** normalize_power)
        for key in terms:
            terms[key] /= s

    ks, ls, cs = [], [], []
    for (k, l), val in terms.items():
        if val != 0.0:
            ks.append(k); ls.append(l); cs.append(val)

    exps_x_sub = torch.tensor([ks], dtype=torch.float32, device=device)
    exps_y_sub = torch.tensor([ls], dtype=torch.float32, device=device)
    coeffs_sub = torch.tensor([cs], dtype=torch.float32, device=device)
    return exps_x_sub, exps_y_sub, coeffs_sub

# ============================================================
# 3) Integration on h-refined grid with ML quadrature on partial cells only
# ============================================================
def compute_h_refined_integral(n, model, normalize_power=NORMALIZE_POWER):
    h = 1.0 / n
    jac = h * h
    centers = np.linspace(-1 + h, 1 - h, n)
    total = 0.0

    for ox in centers:
        for oy in centers:
            cell_case = 'partial' if n == 1 else classify_cell(ox, oy, h)

            if cell_case == 'inside':
                sub = 4.0                      # reference integral
                total += jac * sub             # scale to physical 
            elif cell_case == 'outside':
                sub = 0.0
                total += jac * sub
            else:
                ex_x, ex_y, cf = expand_to_subcell_monomials(
                    EXPS_X, EXPS_Y, COEFFS, ox, oy, h, normalize_power=normalize_power
                )
                with torch.no_grad():
                    xn, yn, w = model(ex_x, ex_y, cf)

                # postprocess returns weights already scaled to PHYSICAL space
                x_b, y_b, w_b, _, _, _ = postprocess_nodes_weights(ox, oy, h, xn, yn, w)

                # integrate in physical space (no extra jac here)
                sub_t = utilities.compute_integration(x_b, y_b, w_b, lambda *_: 1.0)
                sub = sub_t[0].item()
                total += sub                   
    return total
# ============================================================
# 4) High-res reference area (no ML) for convergence
# ============================================================
def reference_area_grid(res=2048):
    xs = np.linspace(-1, 1, res, dtype=np.float64)
    ys = np.linspace(-1, 1, res, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    mask = (phi_eval(X, Y) <= 0.0)
    pixel_area = (2.0 / (res - 1))**2
    return float(mask.sum()) * pixel_area

# ============================================================
# 5) Plots (use sorted nodes; color by SCALED weights)
# ============================================================
def save_subcell_nodes_plot(n, model, filename, normalize_power=NORMALIZE_POWER):
    h = 1.0 / n
    centers = np.linspace(-1 + h, 1 - h, n)

    par_x_all, par_y_all, par_w_all = [], [], []
    plt.figure(figsize=(8, 8))

    for ox in centers:
        for oy in centers:
            cell_case = 'partial' if n == 1 else classify_cell(ox, oy, h)

            if cell_case in ('inside', 'outside'):
                rect = plt.Rectangle(
                    (ox - h, oy - h), 2*h, 2*h,
                    facecolor=('lightgreen' if cell_case == 'inside' else 'lightgray'),
                    edgecolor='blue', alpha=(0.5 if cell_case == 'inside' else 0.3),
                    linestyle='--'
                )
                plt.gca().add_patch(rect)
            else:
                ex_x, ex_y, cf = expand_to_subcell_monomials(
                    EXPS_X, EXPS_Y, COEFFS, ox, oy, h, normalize_power=normalize_power
                )
                with torch.no_grad():
                    xn, yn, w = model(ex_x, ex_y, cf)

                # postprocess (get flat arrays for plotting)
                _, _, _, x_sorted, y_sorted, w_scaled = postprocess_nodes_weights(ox, oy, h, xn, yn, w)

                par_x_all.append(x_sorted)
                par_y_all.append(y_sorted)
                par_w_all.append(w_scaled)

    if par_x_all:
        xs = np.concatenate(par_x_all); ys = np.concatenate(par_y_all); ws = np.concatenate(par_w_all)
        sc = plt.scatter(xs, ys, c=ws, s=10, edgecolors='k', cmap='viridis')
        plt.colorbar(sc, label="Scaled Weight")

    # grid lines
    sub_w = 2.0 / n
    for i in range(n + 1):
        coord = -1 + i * sub_w
        plt.axvline(x=coord, color='blue', linestyle='--', linewidth=0.5)
        plt.axhline(y=coord, color='blue', linestyle='--', linewidth=0.5)

    plt.title(f"Subcell Predicted Nodes for φ≤0 (n={n})")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-1, 1); plt.ylim(-1, 1)
    plt.grid(True); plt.tight_layout()
    plt.savefig(filename, dpi=300); plt.close()

def plot_all_subcell_nodes_landscape(ref_levels, model, filename, normalize_power=NORMALIZE_POWER):
    nL = len(ref_levels)
    fig, axes = plt.subplots(1, nL, figsize=(6*nL, 6), squeeze=False)

    last_scatter = None
    for ax, n in zip(axes[0], ref_levels):
        h = 1.0 / n
        centers = np.linspace(-1 + h, 1 - h, n)
        par_x_all, par_y_all, par_w_all = [], [], []

        for ox in centers:
            for oy in centers:
                cell_case = 'partial' if n == 1 else classify_cell(ox, oy, h)

                if cell_case in ('inside', 'outside'):
                    rect = plt.Rectangle(
                        (ox - h, oy - h), 2*h, 2*h,
                        facecolor=('lightgreen' if cell_case == 'inside' else 'lightgray'),
                        edgecolor='blue', alpha=(0.5 if cell_case == 'inside' else 0.3),
                        linestyle='--'
                    )
                    ax.add_patch(rect)
                else:
                    ex_x, ex_y, cf = expand_to_subcell_monomials(
                        EXPS_X, EXPS_Y, COEFFS, ox, oy, h, normalize_power=normalize_power
                    )
                    with torch.no_grad():
                        xn, yn, w = model(ex_x, ex_y, cf)

                    _, _, _, x_sorted, y_sorted, w_scaled = postprocess_nodes_weights(ox, oy, h, xn, yn, w)

                    par_x_all.append(x_sorted)
                    par_y_all.append(y_sorted)
                    par_w_all.append(w_scaled)

        if par_x_all:
            xs = np.concatenate(par_x_all); ys = np.concatenate(par_y_all); ws = np.concatenate(par_w_all)
            last_scatter = ax.scatter(xs, ys, c=ws, s=10, edgecolors='k', cmap='viridis')

        ax.set_title(f"n = {n}")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
        ax.grid(True, linestyle='--', linewidth=0.5)

    if last_scatter is not None:
        cbar = fig.colorbar(last_scatter, ax=axes[0].tolist(), orientation='horizontal',
                            fraction=0.04, pad=0.12, location='top', aspect=40)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.set_label("Scaled Weight")

    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.98, wspace=0.3)
    plt.tight_layout(rect=[0, 0.10, 1, 0.85])
    plt.savefig(filename, dpi=300); plt.close()

# ============================================================
# 6) Convergence driver
# ============================================================
def compute_error_polynomial():
    ref_area = reference_area_grid(res=2048)

    ref_levels = [1, 2, 4, 8 ,16, 32, 64 ,128]
    error_list, area_list = [], []

    print("\nPolynomial φ convergence via h-refinement (ML on partial cells only):")
    print(f"  Reference area (grid) ≈ {ref_area:.12f}")
    for n in ref_levels:
        print(f"  Subcells: {n}x{n}")
        pred_area = compute_h_refined_integral(n, model, normalize_power=NORMALIZE_POWER)
        area_list.append(pred_area)
        rel_err = abs(pred_area - ref_area) / max(ref_area, 1e-16)
        error_list.append(rel_err)
        print(f"    Predicted area: {pred_area:.16f}")
        print(f"    Reference area: {ref_area:.16f}")
        print(f"    Relative error: {rel_err:.16f}\n")

        plot_fn = os.path.join(output_folder, f"predicted_nodes_poly_n{n}.png")
        save_subcell_nodes_plot(n, model, filename=plot_fn, normalize_power=NORMALIZE_POWER)
        print(f"Aggregate subcell plot saved as '{plot_fn}'")

    # Error vs element size (log–log with slope)
    hs = np.array([2.0 / n for n in ref_levels])
    errs = np.array(error_list)

    plt.figure(figsize=(8, 6))
    plt.loglog(hs, errs, 'o-', label="Relative Error")

    # Compute slope (between last two points for stability)
    log_h = np.log(hs)
    log_e = np.log(errs)
    slope, _ = np.polyfit(log_h, log_e, 1)

    # Annotate slope line (draw at mid-range)
    x0 = hs[len(hs)//2]
    y0 = errs[len(hs)//2]
    factor = 0.5 * hs[0]
    plt.loglog([x0, x0/factor], [y0, y0*(factor**slope)], 'r--', label=f"Slope ≈ {slope:.2f}")

    plt.xlabel("Element Size h = 2/n [log]")
    plt.ylabel("Relative Error [log]")
    plt.title("Relative Error vs Element Size (Polynomial φ)")
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    err_plot_fn = os.path.join(output_folder, "error_vs_element_size_poly.png")
    plt.savefig(err_plot_fn, dpi=300)
    plt.close()

    print(f"Estimated convergence slope ≈ {slope:.3f}")
    print(f"Relative error plot saved as '{err_plot_fn}'")

    return error_list, ref_levels

def main():
    errs, levels = compute_error_polynomial()
    landscape_fn = os.path.join(output_folder, "all_subcell_nodes_landscape_poly.png")
    plot_all_subcell_nodes_landscape(levels, model, filename=landscape_fn, normalize_power=NORMALIZE_POWER)
    print(f"Combined landscape plot saved as '{landscape_fn}'")

if __name__ == "__main__":
    main()
