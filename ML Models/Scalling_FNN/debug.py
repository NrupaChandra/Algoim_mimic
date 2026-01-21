import pandas as pd
import numpy as np

# ============================================================
# Paths
# ============================================================
ref_file  = r"C:\Git\Algoim_mimic\ML Models\Scalling_FNN\Model\1kTestBernstein_p1_ScaleCenter.txt"
pred_file = r"C:\Git\Algoim_mimic\ML Models\Scalling_FNN\Model\predicted_scales_centers.txt"

# ============================================================
# Helpers
# ============================================================
def parse_vec(s):
    """Parse comma-separated string into 1D numpy array"""
    return np.array([float(x) for x in s.split(",")], dtype=float)

# ============================================================
# Read data
# ============================================================
ref  = pd.read_csv(ref_file,  sep=";", dtype=str)
pred = pd.read_csv(pred_file, sep=";", dtype=str)

# Required columns
ref_cols  = ["id", "xscales", "yscales", "xcenters", "ycenters"]
pred_cols = ["id", "xscale", "yscale", "xcenter", "ycenter"]

assert all(c in ref.columns for c in ref_cols),  "Reference file missing columns"
assert all(c in pred.columns for c in pred_cols), "Prediction file missing columns"

# ============================================================
# Align by id
# ============================================================
ref  = ref.set_index("id")
pred = pred.set_index("id")

common_ids = ref.index.intersection(pred.index)
assert len(common_ids) > 0, "No matching IDs found"

# ============================================================
# Error accumulators
# ============================================================
err_xscale  = []
err_yscale  = []
err_xcenter = []
err_ycenter = []

# ============================================================
# Loop over samples
# ============================================================
for idx in common_ids:

    xs_ref = parse_vec(ref.loc[idx, "xscales"])
    ys_ref = parse_vec(ref.loc[idx, "yscales"])
    xc_ref = parse_vec(ref.loc[idx, "xcenters"])
    yc_ref = parse_vec(ref.loc[idx, "ycenters"])

    xs_prd = parse_vec(pred.loc[idx, "xscale"])
    ys_prd = parse_vec(pred.loc[idx, "yscale"])
    xc_prd = parse_vec(pred.loc[idx, "xcenter"])
    yc_prd = parse_vec(pred.loc[idx, "ycenter"])

    err_xscale.append(np.abs(xs_ref - xs_prd))
    err_yscale.append(np.abs(ys_ref - ys_prd))
    err_xcenter.append(np.abs(xc_ref - xc_prd))
    err_ycenter.append(np.abs(yc_ref - yc_prd))

# ============================================================
# Stack & compute MAE
# ============================================================
err_xscale  = np.vstack(err_xscale)
err_yscale  = np.vstack(err_yscale)
err_xcenter = np.vstack(err_xcenter)
err_ycenter = np.vstack(err_ycenter)

MAE = {
    "xscale":  err_xscale.mean(),
    "yscale":  err_yscale.mean(),
    "xcenter": err_xcenter.mean(),
    "ycenter": err_ycenter.mean(),
}

print("=== Mean Absolute Errors ===")
for k, v in MAE.items():
    print(f"{k:8s}: {v:.6e}")
