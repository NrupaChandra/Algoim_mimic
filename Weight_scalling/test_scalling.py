import numpy as np
from multidataloader_fnn import MultiChunkDataset
import os
from torch.utils.data import DataLoader
import torch
import utilities
import matplotlib.pyplot as plt

device = torch.device('cpu')

# Data loading

data_dir = r"C:\Git\Algoim_mimic\Pre_processing"
pre_txt  = r"C:\Git\Algoim_mimic\Weight_scalling\Weight_scalling.txt"

results_dir = r"C:\Git\Algoim_mimic\Weight_scalling\Results"
os.makedirs(results_dir, exist_ok=True)

output_folder = results_dir
output_file   = os.path.join(output_folder, "predicted_data_fnn.txt")
with open(output_file, 'w') as f:
    f.write("number;id;nodes_x;nodes_y;weights\n")

def test_fn(x, y):
    return 1

# ------------------------------------------------------
# Simple parser for predictions text file
# ------------------------------------------------------
def load_predictions(path):
    preds = {}
    cur_id = None
    cur_x = cur_y = cur_w = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("number:"):
                # commit previous
                if cur_id is not None and cur_x is not None and cur_y is not None and cur_w is not None:
                    preds[cur_id] = (cur_x, cur_y, cur_w)
                cur_id = None; cur_x = cur_y = cur_w = None
            elif line.startswith("id:"):
                cur_id = line.split("id:")[1].strip()
            elif line.startswith("nodes_x:"):
                txt = line.split("nodes_x:")[1].strip().strip("[]")
                cur_x = np.fromstring(txt, sep=" ", dtype=np.float32)
            elif line.startswith("nodes_y:"):
                txt = line.split("nodes_y:")[1].strip().strip("[]")
                cur_y = np.fromstring(txt, sep=" ", dtype=np.float32)
            elif line.startswith("weights:"):
                txt = line.split("weights:")[1].strip().strip("[]")
                cur_w = np.fromstring(txt, sep=" ", dtype=np.float32)
        # commit last
        if cur_id is not None and cur_x is not None and cur_y is not None and cur_w is not None:
            preds[cur_id] = (cur_x, cur_y, cur_w)
    return preds

pred_by_id = load_predictions(pre_txt)

# ------------------------------------------------------
# Dataset
# ------------------------------------------------------
dataset = MultiChunkDataset(
    index_file=os.path.join(data_dir, 'preprocessed_chuncks_10kMonotonic_functions/index.txt'),
    base_dir=data_dir
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# ------------------------------------------------------
# Error tracking
# ------------------------------------------------------
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

        # === NEW: load predicted nodes from file instead of model ===
        id_str = id[0] if isinstance(id, (list,tuple)) else str(id)
        if id_str not in pred_by_id:
            print(f"[WARN] no prediction for id {id_str}, skipping")
            continue
        px, py, pw = pred_by_id[id_str]  # these are NumPy arrays
        predicted_nodes_x = px.astype(np.float32).ravel()
        predicted_nodes_y = py.astype(np.float32).ravel()
        predicted_weights = pw.astype(np.float32).ravel()

        pnx_t = torch.from_numpy(predicted_nodes_x).to(device=device, dtype=torch.float32).unsqueeze(0)  # [1, N]
        pny_t = torch.from_numpy(predicted_nodes_y).to(device=device, dtype=torch.float32).unsqueeze(0)  # [1, N]
        pw_t  = torch.from_numpy(predicted_weights).to(device=device, dtype=torch.float32).unsqueeze(0)  # [1, N]

        # Integrals
        pred_val = utilities.compute_integration(pnx_t, pny_t, pw_t ,test_fn)[0].item()
        true_val = utilities.compute_integration(true_values_x, true_values_y, true_values_w, test_fn)[0].item()

        predicted_integrals.append(pred_val)
        true_integrals.append(true_val)

        absolute_difference = abs(pred_val - true_val)
        squared_difference  = (pred_val - true_val) ** 2
        total_absolute_difference += absolute_difference
        total_squared_difference  += squared_difference
        total_ids += 1

        rel_error = absolute_difference / abs(true_val) if abs(true_val) > 1e-10 else 0.0
        relative_errors.append(rel_error)
        rel_error_info.append((id_str, rel_error))

        print(f"Result of integration for {id_str}:")
        print(f"Algoim (True):  {true_val:.4e}")
        print(f"From file (Pred): {pred_val:.4e}")
        print(f"Absolute Difference: {absolute_difference:.4e}")
        print(f"Relative Error: {rel_error*100:.2f}%")

        # Plot
        plt.figure(figsize=(10, 6))
        grid = np.linspace(-1, 1, 400)
        XX, YY = np.meshgrid(grid, grid)
        ZZ = np.zeros_like(XX)
        for ex, ey, c in zip(exp_x.cpu().numpy().reshape(-1),
                             exp_y.cpu().numpy().reshape(-1),
                             coeff.cpu().numpy().reshape(-1)):
            ZZ += c * (XX**ex) * (YY**ey)
        plt.contour(XX, YY, ZZ, levels=[0], colors='k', linewidths=1.5)
        plt.scatter(true_nodes_x, true_nodes_y, c=true_weights, cmap='viridis',
                    label='Reference Points (Algoim)', alpha=0.6, marker='x')
        plt.scatter(predicted_nodes_x, predicted_nodes_y, c=predicted_weights, cmap='plasma',
                    label='Predicted Points', alpha=0.6)
        plt.title('Reference (Algoim) vs Predicted Nodes')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.colorbar(label='Weight (Coefficient)')
        plt.legend()
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.text(0.05, 0.95, f"True Int (Algoim): {true_val:.8f}\nPred Int : {pred_val:.8f}",
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.5))
        sample_plot_path = os.path.join(output_folder, f'{id_str}.png')
        plt.savefig(sample_plot_path)
        plt.close()

        with open(output_file, 'a') as f:
            f.write(f"{number};{id_str};{','.join(map(str, predicted_nodes_x))};"
                    f"{','.join(map(str, predicted_nodes_y))};"
                    f"{','.join(map(str, predicted_weights))}\n")

        number += 1
