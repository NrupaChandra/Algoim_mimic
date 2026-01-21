#!/usr/bin/env python
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from model_scalling_fnn_v2 import load_ff_pipelines_model

# -----------------------------
# Device
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# Paths
# -----------------------------
data_dir = r"C:\Git\Algoim_mimic\Pre_processing\1kpreprocessed_chunks_scale_center"
chunk_path = os.path.join(data_dir, "preprocessed_chunk0.pt")

model_dir = r"C:\Git\Algoim_mimic\ML Models\Scalling_FNN\Model"
weights_path = os.path.join(model_dir, "fnn_model_weights_v7.pth")
output_path = os.path.join(model_dir, "predicted_scales_centers.txt")
output_path_2 = os.path.join(model_dir, "traintrue_scales_centers.txt")

# -----------------------------
# Load model
# -----------------------------
model = load_ff_pipelines_model()
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Load data
# -----------------------------
print(f"Loading test chunk from: {chunk_path}")
chunk_data = torch.load(chunk_path, map_location=device)
print(f"Loaded {len(chunk_data)} samples.")

# -----------------------------
# Storage
# -----------------------------
mae_sx, mae_sy = [], []
mae_cx, mae_cy = [], []

# -----------------------------
# Inference
# -----------------------------
with torch.no_grad(), open(output_path, "w") as f:
    f.write(
        "number;id;"
        "xscale;yscale;"
        "xcenter;ycenter\n"
    )

    for idx, sample in enumerate(chunk_data):

        (
            exp_x,
            exp_y,
            coeff,
            true_scales_x,
            true_scales_y,
            true_centers_x,
            true_centers_y,
            sample_id
        ) = sample

        exp_x = exp_x.unsqueeze(0).to(device)
        exp_y = exp_y.unsqueeze(0).to(device)
        coeff = coeff.unsqueeze(0).to(device)

        # ---- model forward
        pred_sx, pred_sy, pred_cx, pred_cy = model(exp_x, exp_y, coeff)

        # ---- numpy
        pred_sx = pred_sx.squeeze().cpu().numpy()
        pred_sy = pred_sy.squeeze().cpu().numpy()
        pred_cx = pred_cx.squeeze().cpu().numpy()
        pred_cy = pred_cy.squeeze().cpu().numpy()

        true_sx = true_scales_x.squeeze().cpu().numpy()
        true_sy = true_scales_y.squeeze().cpu().numpy()
        true_cx = true_centers_x.squeeze().cpu().numpy()
        true_cy = true_centers_y.squeeze().cpu().numpy()

        # ---- MAE per sample
        mae_sx.append(np.mean(np.abs(pred_sx - true_sx)))
        mae_sy.append(np.mean(np.abs(pred_sy - true_sy)))
        mae_cx.append(np.mean(np.abs(pred_cx - true_cx)))
        mae_cy.append(np.mean(np.abs(pred_cy - true_cy)))

        # ---- write output
        sx_str = ",".join(f"{v:.16f}" for v in pred_sx)
        sy_str = ",".join(f"{v:.16f}" for v in pred_sy)
        cx_str = ",".join(f"{v:.16f}" for v in pred_cx)
        cy_str = ",".join(f"{v:.16f}" for v in pred_cy)

        f.write(
            f"{idx};{sample_id};"
            f"{sx_str};{sy_str};"
            f"{cx_str};{cy_str}\n"
        )

with torch.no_grad(), open(output_path_2, "w") as f:
    f.write(
        "number;id;"
        "xscale;yscale;"
        "xcenter;ycenter\n"
    )

    for idx, sample in enumerate(chunk_data):

        (
            exp_x,
            exp_y,
            coeff,
            true_scales_x,
            true_scales_y,
            true_centers_x,
            true_centers_y,
            sample_id
        ) = sample

        exp_x = exp_x.unsqueeze(0).to(device)
        exp_y = exp_y.unsqueeze(0).to(device)
        coeff = coeff.unsqueeze(0).to(device)

        # ---- model forward
        pred_sx, pred_sy, pred_cx, pred_cy = model(exp_x, exp_y, coeff)

        # ---- numpy
        pred_sx = pred_sx.squeeze().cpu().numpy()
        pred_sy = pred_sy.squeeze().cpu().numpy()
        pred_cx = pred_cx.squeeze().cpu().numpy()
        pred_cy = pred_cy.squeeze().cpu().numpy()

        true_sx = true_scales_x.squeeze().cpu().numpy()
        true_sy = true_scales_y.squeeze().cpu().numpy()
        true_cx = true_centers_x.squeeze().cpu().numpy()
        true_cy = true_centers_y.squeeze().cpu().numpy()

        # ---- MAE per sample
        mae_sx.append(np.mean(np.abs(pred_sx - true_sx)))
        mae_sy.append(np.mean(np.abs(pred_sy - true_sy)))
        mae_cx.append(np.mean(np.abs(pred_cx - true_cx)))
        mae_cy.append(np.mean(np.abs(pred_cy - true_cy)))

        # ---- write output
        sx_str = ",".join(f"{v:.16f}" for v in true_sx)
        sy_str = ",".join(f"{v:.16f}" for v in true_sy)
        cx_str = ",".join(f"{v:.16f}" for v in true_cx)
        cy_str = ",".join(f"{v:.16f}" for v in true_cy)

        f.write(
            f"{idx};{sample_id};"
            f"{sx_str};{sy_str};"
            f"{cx_str};{cy_str}\n"
        )


# -----------------------------
# Report
# -----------------------------
print("\n=== Mean Absolute Error ===")
print(f"Scale X : {np.mean(mae_sx):.8e}")
print(f"Scale Y : {np.mean(mae_sy):.8e}")
print(f"Center X: {np.mean(mae_cx):.8e}")
print(f"Center Y: {np.mean(mae_cy):.8e}")
print(f"Predictions saved to: {output_path}")
