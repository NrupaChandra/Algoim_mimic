#!/usr/bin/env python
import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# --- your imports ---
from multidataloader_fnn import MultiChunkDataset
import utilities

# =========================
# Config
# =========================
device = torch.device('cpu')
torch.set_default_dtype(torch.float32)

results_dir = r"C:\Git\Algoim_mimic\Weight_scalling"
data_dir    = r"C:\Git\Algoim_mimic\Pre_processing"
pred_file   = os.path.join(results_dir, "Weight_scalling.txt")

# Folders
plots_dir   = os.path.join(results_dir, "plots")
metrics_dir = os.path.join(results_dir, "metrics")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# =========================
# Prediction file parser
# =========================
ARR_RE = re.compile(r"\[(.*?)\]")

def _parse_array(line: str) -> np.ndarray:
    m = ARR_RE.search(line)
    return np.fromstring(m.group(1).strip(), sep=" ", dtype=np.float32) if m else np.array([], dtype=np.float32)

def load_predictions(path: str):
    """Parses a file with blocks:
       id: <string>
       nodes_x: [x1 x2 ...]
       nodes_y: [y1 y2 ...]
       weights: [w1 w2 ...]
    """
    pred_by_id = {}
    with open(path, "r", encoding="utf-8") as f:
        cur_id = None
        cur_x = cur_y = cur_w = None
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("id:"):
                # commit previous block
                if cur_id is not None and cur_x is not None and cur_y is not None and cur_w is not None:
                    pred_by_id[cur_id] = (cur_x, cur_y, cur_w)
                # start new
                cur_id = line.split("id:")[1].strip()
                cur_x = cur_y = cur_w = None
            elif line.startswith("nodes_x:"):
                cur_x = _parse_array(line)
            elif line.startswith("nodes_y:"):
                cur_y = _parse_array(line)
            elif line.startswith("weights:"):
                cur_w = _parse_array(line)

        # commit last
        if cur_id is not None and cur_x is not None and cur_y is not None and cur_w is not None:
            pred_by_id[cur_id] = (cur_x, cur_y, cur_w)

    return pred_by_id

pred_by_id = load_predictions(pred_file)
print(f"Loaded predictions for {len(pred_by_id)} ids from '{pred_file}'")

# =========================
# Dataset & Loader
# =========================
dataset = MultiChunkDataset(
    index_file=os.path.join(data_dir, 'preprocessed_chuncks_10kMonotonic_functions/index.txt'),
    base_dir=data_dir
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# =========================
# Integration test function
# =========================
def test_fn(x, y):
    # constant integrand: integral equals sum of weights over the domain
    return 1.0

# =========================
# Metrics accumulators
# =========================
total_abs_diff = 0.0
total_sq_diff  = 0.0
total_ids      = 0
relative_errors = []
per_sample_rows = []  # [id, true_int, pred_int, abs_diff, rel_err_pct]

# =========================
# Main: compare + plot + save
# =========================
saved = 0
skipped = 0

with torch.no_grad():
    for sample in dataloader:
        exp_x, exp_y, coeff, true_x, true_y, true_w, id_t = sample
        sid = id_t[0].decode("utf-8") if isinstance(id_t[0], bytes) else str(id_t[0])

        if sid not in pred_by_id:
            skipped += 1
            print(f"[skip] no predictions for id={sid}")
            continue

        # Ground-truth integral
        true_val = utilities.compute_integration(true_x, true_y, true_w, test_fn)[0].item()

        # Predicted arrays (np → torch) and reshape to match true shapes
        px_np, py_np, pw_np = pred_by_id[sid]
        px_t = torch.from_numpy(px_np.reshape(true_x.shape)).to(torch.float32)
        py_t = torch.from_numpy(py_np.reshape(true_y.shape)).to(torch.float32)
        pw_t = torch.from_numpy(pw_np.reshape(true_w.shape)).to(torch.float32)

        # Predicted integral
        pred_val = utilities.compute_integration(px_t, py_t, pw_t, test_fn)[0].item()

        # Metrics
        abs_diff = abs(pred_val - true_val)
        sq_diff  = (pred_val - true_val) ** 2
        rel_err  = abs_diff / abs(true_val) if abs(true_val) > 1e-10 else 0.0

        total_abs_diff += abs_diff
        total_sq_diff  += sq_diff
        total_ids      += 1
        relative_errors.append(rel_err)

        per_sample_rows.append([sid, f"{true_val:.10e}", f"{pred_val:.10e}", f"{abs_diff:.10e}", f"{rel_err*100:.6f}"])

        # Plot (overlay true vs predicted)
        gx = true_x.numpy().flatten()
        gy = true_y.numpy().flatten()
        gw = true_w.numpy().flatten()
        px = px_np.flatten()
        py = py_np.flatten()
        pw = pw_np.flatten()

        plt.figure(figsize=(6, 6))
        sc_true = plt.scatter(gx, gy, c=gw, cmap='viridis', marker='x', label='True (Algoim)', alpha=0.9)
        sc_pred = plt.scatter(px, py, c=pw, cmap='plasma', marker='o', label='Predicted', alpha=0.75)
        cbar = plt.colorbar(sc_pred)
        cbar.set_label('Weight')

        plt.legend()
        plt.title(f"True vs Predicted Nodes — id {sid}")
        plt.xlabel('x'); plt.ylabel('y')
        plt.xlim(-1, 1); plt.ylim(-1, 1)
        plt.grid(True, linewidth=0.3, alpha=0.4)

        # Annotation: true/pred integrals + rel err
        txt = f"True ∫: {true_val:.8f}\nPred ∫: {pred_val:.8f}\nRelErr: {rel_err*100:.2f}%"
        plt.text(0.02, 0.98, txt, transform=plt.gca().transAxes,
                 va='top', ha='left', fontsize=10,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        outpath = os.path.join(plots_dir, f"{sid}.png")
        plt.savefig(outpath, dpi=200, bbox_inches='tight')
        plt.close()

        saved += 1
        print(f"[saved] {outpath}")

# =========================
# Write metrics
# =========================
overall_MAE = total_abs_diff / total_ids if total_ids > 0 else 0.0
overall_MSE = total_sq_diff  / total_ids if total_ids > 0 else 0.0
mean_rel_err_pct   = (np.mean(relative_errors) * 100) if total_ids > 0 else 0.0
median_rel_err_pct = (np.median(relative_errors) * 100) if total_ids > 0 else 0.0

# metrics.txt
with open(os.path.join(metrics_dir, "metrics.txt"), "w", encoding="utf-8") as mf:
    mf.write(f"Total samples compared: {total_ids}\n")
    mf.write(f"Overall MAE: {overall_MAE:.6e}\n")
    mf.write(f"Overall MSE: {overall_MSE:.6e}\n")
    mf.write(f"Mean Relative Error: {mean_rel_err_pct:.4f}%\n")
    mf.write(f"Median Relative Error: {median_rel_err_pct:.4f}%\n")

# per-sample CSV (easy to sort/filter later)
with open(os.path.join(metrics_dir, "per_sample.csv"), "w", newline="", encoding="utf-8") as cf:
    writer = csv.writer(cf)
    writer.writerow(["id", "true_integral", "pred_integral", "abs_diff", "rel_err_percent"])
    writer.writerows(per_sample_rows)

print(f"\nDone.\nSaved {saved} plots to: {plots_dir}\nSkipped {skipped} samples without predictions.")
print(f"Metrics written to: {os.path.join(metrics_dir, 'metrics.txt')}")
print(f"Per-sample CSV:     {os.path.join(metrics_dir, 'per_sample.csv')}")
