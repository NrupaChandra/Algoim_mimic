import torch
from model_scalling_fnn_v2 import load_ff_pipelines_model

# --------------------------------------------------
# Paths (UNCHANGED weights path)
# --------------------------------------------------
weights_path = r"C:\Git\Algoim_mimic\ML Models\Scalling_FNN\Model\fnn_model_weights_v7.pth"
onnx_path    = r"C:\Git\Algoim_mimic\ML Models\scalling_ML_matlab\model_ML\scaling_fnn_v2.onnx"

device = torch.device("cpu")

# --------------------------------------------------
# Load model (v2 architecture)
# --------------------------------------------------
model = load_ff_pipelines_model()
state_dict = torch.load(weights_path, map_location=device)

# If this file is a raw state_dict
if isinstance(state_dict, dict) and "model_state_dict" not in state_dict:
    model.load_state_dict(state_dict)
else:
    model.load_state_dict(state_dict["model_state_dict"])

model.to(device).float()
model.eval()

# --------------------------------------------------
# Dummy inputs (must match training)
# --------------------------------------------------
B = 1
m = 4   # polynomial order used during training

exp_x = torch.zeros(B, m, dtype=torch.float32)
exp_y = torch.zeros(B, m, dtype=torch.float32)
coeff = torch.zeros(B, m, dtype=torch.float32)

# --------------------------------------------------
# Export to ONNX
# --------------------------------------------------
torch.onnx.export(
    model,
    (exp_x, exp_y, coeff),
    onnx_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=[
        "exp_x",
        "exp_y",
        "coeff"
    ],
    output_names=[
        "scales_x",
        "scales_y",
        "center_x",
        "center_y"
    ],
    dynamic_axes={
        "exp_x":    {0: "batch"},
        "exp_y":    {0: "batch"},
        "coeff":    {0: "batch"},
        "scales_x": {0: "batch"},
        "scales_y": {0: "batch"},
        "center_x": {0: "batch"},
        "center_y": {0: "batch"},
    }
)

print(f"\nONNX model successfully exported to:\n{onnx_path}")
