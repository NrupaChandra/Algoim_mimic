import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import utilities
from torch.utils.data import DataLoader
from model_scalling_fnn import load_ff_pipelines_model
from multidataloader_fnn import MultiChunkDataset

#double precision and runs on cpu
torch.set_default_dtype(torch.float64)
device = torch.device('cpu')

#dir paths
data_dir = r"C:\Git\Algoim_mimic\Pre_processing\1kpreprocessed_chunks_weight_scaled"
chunk_path = os.path.join(data_dir, "preprocessed_chunk0.pt")

# Data loading
test_data_dir = r"C:\Git\Algoim_mimic\Pre_processing"
dataset = MultiChunkDataset(
    index_file=os.path.join(data_dir, 'preprocessed_chuncks_1kTestMonotonic_functions/index.txt'),
    base_dir=data_dir
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model_dir = r"C:\Git\Algoim_mimic\ML Models\Scalling_FNN\Model"
weights_path = os.path.join(model_dir,"fnn_model_weights_v6.pth")

output_path = r"C:\Git\Algoim_mimic\Convergence study\ScallingML_convergence\ScallingML_weights_scaled.txt"
os.makedirs(model_dir, exist_ok=True)

results_dir = r"C:\Git\Algoim_mimic\Convergence study\ScallingML_convergence"
output_folder = results_dir
os.makedirs(output_folder, exist_ok=True)

#model
model = load_ff_pipelines_model()
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

print(f"Loading test chunk from: {chunk_path}")
chunk_data = torch.load(chunk_path, map_location=device)
print(f"Loaded {len(chunk_data)} samples.")

#scalling data 
weightRefs = np.array([0.1012285362903763 ,0.2223810344533745 ,0.3137066458778873 ,0.3626837833783620 ,0.3626837833783620 ,0.3137066458778873 ,0.2223810344533745, 0.1012285362903763], dtype= np.float64)
deltaRef = 0.9602898564975363*2 #always same since it comes from fixed quadrature 
w = np.outer(weightRefs,weightRefs)


scaled_weights_all = []
sample_ids = []

#Inference with the model 
with torch.no_grad(), open(output_path, "w") as f:
    f.write("number;id;pred_scale_x;pred_scale_y\n")

    for idx, sample in enumerate(chunk_data):
        exp_x, exp_y, coeff, true_scales_x, true_scales_y, mask64, sample_id = sample

        exp_x = exp_x.unsqueeze(0).to(device)
        exp_y = exp_y.unsqueeze(0).to(device)
        coeff = coeff.unsqueeze(0).to(device)

        pred_scale_x, pred_scale_y = model(exp_x, exp_y, coeff)

        pred_x = pred_scale_x.squeeze(0).cpu().numpy()
        pred_y = pred_scale_y.squeeze(0).cpu().numpy()
        true_x = true_scales_x.squeeze(0).cpu().numpy()
        true_y = true_scales_y.squeeze(0).cpu().numpy()

        scale_mat = pred_x[:, None] * pred_y[None, :] 
        scaled = scale_mat * w   


        scaled_weights_all.append(scaled)
        sample_ids.append(sample_id)

with open(output_path, "w", encoding="utf-8") as f_w:
    f_w.write("number;id;weights_scaled\n")

    for idx, (sample_id, scaled) in enumerate(zip(sample_ids, scaled_weights_all), start=1):
        flat_weights = np.array(scaled.flatten(order="C"), dtype=np.float64)
        w_str = " ".join(f"{v:.16e}" for v in flat_weights)
        f_w.write(f"{idx};{sample_id};{w_str}\n")

print(f"Saved scaled weights (double precision) to {output_path}")


def test_fn(x, y):
    return 1

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
    for sample in dataloader:
        exp_x, exp_y, coeff, true_values_x, true_values_y, true_values_w, id = sample

        exp_x, exp_y, coeff = (exp_x.to(device, dtype=torch.float32),
                               exp_y.to(device, dtype=torch.float32),
                               coeff.to(device, dtype=torch.float32))

        true_nodes_x = true_values_x.numpy().astype(np.float32)
        true_nodes_y = true_values_y.numpy().astype(np.float32)
        true_weights = true_values_w.numpy().astype(np.float32)

        predicted_values_x, predicted_values_y, predicted_values_w = model(exp_x, exp_y, coeff)

        predicted_nodes_x = predicted_values_x.cpu().numpy().astype(np.float32)
        predicted_nodes_y = predicted_values_y.cpu().numpy().astype(np.float32)
        predicted_weights = predicted_values_w.cpu().numpy().astype(np.float32)

        pred_val = utilities.compute_integration(predicted_values_x, predicted_values_y, predicted_values_w, test_fn)[0].item()
        true_val = utilities.compute_integration(true_values_x, true_values_y, true_values_w, test_fn)[0].item()

        predicted_integrals.append(pred_val)
        true_integrals.append(true_val)
    
        absolute_difference = abs(pred_val - true_val)
        squared_difference = (pred_val - true_val) ** 2
        total_absolute_difference += absolute_difference
        total_squared_difference += squared_difference
        total_ids += 1

        rel_error = absolute_difference / abs(true_val) if abs(true_val) > 1e-10 else 0.0
        relative_errors.append(rel_error)
        rel_error_info.append((id[0], rel_error))

        print(f"Result of integration for {id}:")
        print(f"Algoim (True):  {true_val:.4e}")
        print(f"QuadNET (Pred): {pred_val:.4e}")
        print(f"Absolute Difference: {absolute_difference:.4e}")
        print(f"Relative Error: {rel_error*100:.2f}%")

        plt.figure(figsize=(10, 6))
        grid = np.linspace(-1, 1, 400)
        XX, YY = np.meshgrid(grid, grid)
        ZZ = np.zeros_like(XX)
        for ex, ey, c in zip(exp_x.cpu().numpy().reshape(-1), exp_y.cpu().numpy().reshape(-1), coeff.cpu().numpy().reshape(-1)):
            ZZ += c * (XX**ex) * (YY**ey)
        plt.contour(XX, YY, ZZ, levels=[0], colors='k', linewidths=1.5)
        plt.scatter(true_nodes_x, true_nodes_y, c=true_weights, cmap='viridis', label='Reference Points (Algoim)', alpha=0.6, marker='x')
        plt.scatter(predicted_nodes_x, predicted_nodes_y, c=predicted_weights, cmap='plasma', label='Predicted Points', alpha=0.6)
        plt.title('Reference(Algoim) vs Predicted Nodes')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.colorbar(label='Weight (Coefficient)')
        plt.legend()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.text(0.05, 0.95, f"True Int (Algoim): {true_val:.8f}\nPred Int : {pred_val:.8f}",
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.5))
        sample_plot_path = os.path.join(output_folder, f'{id[0]}.png')
        plt.savefig(sample_plot_path)
        plt.close()

        with open(output_file, 'a') as f:
            f.write(f"{number};{id[0]};{','.join(map(str, predicted_nodes_x))};{','.join(map(str, predicted_nodes_y))};{','.join(map(str, predicted_weights))}\n")

        number += 1

# Metrics
overall_MAE = total_absolute_difference / total_ids if total_ids > 0 else 0
overall_MSE = total_squared_difference / total_ids if total_ids > 0 else 0
mean_relative_error = (sum(relative_errors) / total_ids * 100) if total_ids > 0 else 0
median_relative_error = (np.median(relative_errors) * 100) if total_ids > 0 else 0

print(f"Overall MAE: {overall_MAE:.4e}")
print(f"Overall MSE: {overall_MSE:.4e}")
print(f"Mean Relative Error: {mean_relative_error:.2f}%")
print(f"Median Relative Error: {median_relative_error:.2f}%")