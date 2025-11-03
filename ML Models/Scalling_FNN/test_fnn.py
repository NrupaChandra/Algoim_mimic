#!/usr/bin/env python
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from model_scalling_fnn import load_ff_pipelines_model

device = torch.device("cpu")  # or "cuda" if available

# ------------------ Paths ------------------
data_dir = r"C:\Git\Algoim_mimic\Pre_processing\1kpreprocessed_chunks_weight_scaled"
chunk_path = os.path.join(data_dir, "preprocessed_chunk0.pt")

model_dir = r"C:\Git\Algoim_mimic\ML Models\Scalling_FNN\Model"
weights_path = os.path.join(model_dir, "fnn_model_weights_v6.pth")
output_path = os.path.join(model_dir, "predicted_scales.txt")

# ------------------ Model ------------------
model = load_ff_pipelines_model()
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

print(f"Loading test chunk from: {chunk_path}")
chunk_data = torch.load(chunk_path, map_location=device)
print(f"Loaded {len(chunk_data)} samples.")

# ------------------ Lists to collect data ------------------
mae_x_list, mae_y_list = [], []
true_x_all, pred_x_all = [], []
true_y_all, pred_y_all = [], []

# ------------------ Inference ------------------
with torch.no_grad(), open(output_path, "w") as f:
    f.write("number;id;pred_scale_x;pred_scale_y\n")

    for idx, sample in enumerate(chunk_data):
        exp_x, exp_y, coeff, true_scales_x, true_scales_y, mask64, sample_id = sample

        exp_x = exp_x.unsqueeze(0).to(device)
        exp_y = exp_y.unsqueeze(0).to(device)
        coeff = coeff.unsqueeze(0).to(device)

        pred_scale_x, pred_scale_y = model(exp_x, exp_y, coeff)

        pred_x = pred_scale_x.squeeze().cpu().numpy()
        pred_y = pred_scale_y.squeeze().cpu().numpy()
        true_x = true_scales_x.squeeze().cpu().numpy()
        true_y = true_scales_y.squeeze().cpu().numpy()

        # store for later plotting
        pred_x_all.append(pred_x)
        pred_y_all.append(pred_y)
        true_x_all.append(true_x)
        true_y_all.append(true_y)

        # mean absolute error for this sample
        mae_x = np.mean(np.abs(pred_x - true_x))
        mae_y = np.mean(np.abs(pred_y - true_y))
        mae_x_list.append(mae_x)
        mae_y_list.append(mae_y)

        pred_x_str = ",".join(f"{v:.16f}" for v in pred_x)
        pred_y_str = ",".join(f"{v:.16f}" for v in pred_y)
        f.write(f"{idx};{sample_id};{pred_x_str};{pred_y_str}\n")

# ------------------ Compute overall stats ------------------
overall_mae_x = np.mean(mae_x_list)
overall_mae_y = np.mean(mae_y_list)

print("\n=== Mean Absolute Error (MAE) ===")
print(f"MAE_x: {overall_mae_x:.8e}")
print(f"MAE_y: {overall_mae_y:.8e}")
print(f"Predictions saved to: {output_path}")

# ------------------ Prepare flattened arrays ------------------
true_x_all = np.concatenate(true_x_all)
pred_x_all = np.concatenate(pred_x_all)
true_y_all = np.concatenate(true_y_all)
pred_y_all = np.concatenate(pred_y_all)

# ------------------ PLOTS ------------------

threshold = 0.05  # tweak if needed
filtered_x = [m for m in mae_x_list if m <= threshold]
filtered_y = [m for m in mae_y_list if m <= threshold]

filtered_mae_x = np.mean(filtered_x)
filtered_mae_y = np.mean(filtered_y)

print("\n=== Filtered Mean Absolute Error (MAE, ignoring outliers) ===")
print(f"MAE_x (filtered): {filtered_mae_x:.8e}")
print(f"MAE_y (filtered): {filtered_mae_y:.8e}")

# ---------- side-by-side histograms ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# MAE_x
axes[0].hist(filtered_x, bins=30, color='royalblue', alpha=0.8)
axes[0].set_xlabel("MAE_x per sample")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Error Distribution (x)")
axes[0].grid(True)
axes[0].text(0.95, 0.95, f"MAE_x_scales = {filtered_mae_x:.4e}",
             transform=axes[0].transAxes,
             ha='right', va='top',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# MAE_y
axes[1].hist(filtered_y, bins=30, color='seagreen', alpha=0.8)
axes[1].set_xlabel("MAE_y per sample")
axes[1].set_title("Error Distribution (y)")
axes[1].grid(True)
axes[1].text(0.95, 0.95, f"MAE_y_scales = {filtered_mae_y:.4e}",
             transform=axes[1].transAxes,
             ha='right', va='top',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(model_dir, "mae_histogram_filtered_side_by_side.png"), dpi=300)
plt.show()
