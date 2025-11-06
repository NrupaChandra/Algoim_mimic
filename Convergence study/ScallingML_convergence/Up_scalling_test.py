import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import utilities
from torch.utils.data import DataLoader
from model_scalling_fnn import load_ff_pipelines_model
from multidataloader_fnn import MultiChunkDataset

# double precision and runs on cpu
torch.set_default_dtype(torch.float64)
device = torch.device('cpu')

# dir paths
data_dir = r"C:\Git\Algoim_mimic\Pre_processing\1kpreprocessed_chunks_weight_scaled"
chunk_path = os.path.join(data_dir, "preprocessed_chunk0.pt")

# Data loading
test_data_dir = r"C:\Git\Algoim_mimic\Pre_processing"
dataset = MultiChunkDataset(
    index_file=os.path.join(test_data_dir, 'preprocessed_chuncks_1kTestMonotonic_functions/index.txt'),
    base_dir=data_dir
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model_dir = r"C:\Git\Algoim_mimic\ML Models\Scalling_FNN\Model"
weights_path = os.path.join(model_dir, "fnn_model_weights_v6.pth")

output_path = r"C:\Git\Algoim_mimic\Convergence study\ScallingML_convergence\ScallingML_weights_scaled.txt"
os.makedirs(model_dir, exist_ok=True)

results_dir = r"C:\Git\Algoim_mimic\Convergence study\ScallingML_convergence"
output_folder = results_dir
os.makedirs(output_folder, exist_ok=True)

# model
model = load_ff_pipelines_model()
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

print(f"Loading test chunk from: {chunk_path}")
chunk_data = torch.load(chunk_path, map_location=device)
print(f"Loaded {len(chunk_data)} samples.")

# scaling data 
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

deltaRef = 0.9602898564975363 * 2  # always same since it comes from fixed quadrature
w = np.outer(weightRefs, weightRefs)  

scaled_weights_all = [] 
sample_ids = []      

with torch.no_grad(), open(output_path, "w", encoding="utf-8") as f_dummy:
    f_dummy.write("number;id;pred_scale_x;pred_scale_y\n")

    for idx, sample in enumerate(chunk_data):
        exp_x, exp_y, coeff, true_scales_x, true_scales_y, mask64, sample_id = sample

        exp_x = exp_x.unsqueeze(0).to(device)
        exp_y = exp_y.unsqueeze(0).to(device)
        coeff = coeff.unsqueeze(0).to(device)

        pred_scale_x, pred_scale_y = model(exp_x, exp_y, coeff)

        pred_x = pred_scale_x.squeeze(0).cpu().numpy()  
        pred_y = pred_scale_y.squeeze(0).cpu().numpy()  

        # outer product of scaling factors
        scale_mat = pred_x[:, None] * pred_y[None, :]   
        scaled = scale_mat * w                          

        scaled_weights_all.append(scaled)
        sample_ids.append(sample_id)

# Write scaled weights to file (flattened)
with open(output_path, "w", encoding="utf-8") as f_w:
    f_w.write("number;id;weights_scaled\n")
    for idx, (sample_id, scaled) in enumerate(zip(sample_ids, scaled_weights_all), start=1):
        flat_weights = np.array(scaled.flatten(order="C"), dtype=np.float64)
        w_str = " ".join(f"{v:.16e}" for v in flat_weights)
        f_w.write(f"{idx};{sample_id};{w_str}\n")

print(f"Saved scaled weights (double precision) to {output_path}")


def test_fn(x, y):
    return 1.0


# Sanity check
assert len(scaled_weights_all) == len(chunk_data), "scaled_weights_all and chunk_data size mismatch!"
assert len(scaled_weights_all) == len(dataset), "scaled_weights_all and dataset size mismatch!"

# Error tracking variables
total_absolute_difference = 0.0
total_squared_difference = 0.0
total_ids = 0
predicted_integrals = []
true_integrals = []
relative_errors = []
rel_error_info = []
number = 1

with torch.no_grad():
    for idx, sample in enumerate(dataloader):
        exp_x, exp_y, coeff, true_values_x, true_values_y, true_values_w, id = sample

        # True integral
        true_val = utilities.compute_integration(
            true_values_x, true_values_y, true_values_w, test_fn
        )[0].item()
        true_integrals.append(true_val)

        # Use the *matching* scaled weights for this sample
        scaled = scaled_weights_all[idx]    # (8, 8) numpy array
        pred_val = float(np.sum(scaled))    # since test_fn = 1
        predicted_integrals.append(pred_val)

        # Error metrics for this sample
        absolute_difference = abs(pred_val - true_val)
        squared_difference = (pred_val - true_val) ** 2
        total_absolute_difference += absolute_difference
        total_squared_difference += squared_difference
        total_ids += 1

        rel_error = absolute_difference / abs(true_val) if abs(true_val) > 1e-10 else 0.0
        relative_errors.append(rel_error)

        # store id + error (id from dataset or from sample_ids if you prefer)
        rel_error_info.append((id[0], rel_error))

        print(f"Result of integration for {id}:")
        print(f"Algoim (True):  {true_val:.4e}")
        print(f"QuadNET (Pred): {pred_val:.4e}")
        print(f"Absolute Difference: {absolute_difference:.4e}")
        print(f"Relative Error: {rel_error*100:.2f}%")

        number += 1

# Metrics
overall_MAE = total_absolute_difference / total_ids if total_ids > 0 else 0.0
overall_MSE = total_squared_difference / total_ids if total_ids > 0 else 0.0
mean_relative_error = (sum(relative_errors) / total_ids * 100) if total_ids > 0 else 0.0
median_relative_error = (np.median(relative_errors) * 100) if total_ids > 0 else 0.0

print(f"Overall MAE: {overall_MAE:.8e}")
print(f"Overall MSE: {overall_MSE:.8e}")

