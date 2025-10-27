#!/usr/bin/env python
import torch
import torch.nn as nn
import numpy as np

class NodalPreprocessor(nn.Module):

    def __init__(self, num_nodes=64, domain=(-1, 1)):
        super().__init__()
        self.num_nodes = num_nodes
        self.domain = domain
        grid_size = int(np.sqrt(num_nodes))
        if grid_size * grid_size != num_nodes:
            raise ValueError("num_nodes must be a perfect square, e.g. 64.")
        xs = torch.linspace(domain[0], domain[1], grid_size, dtype=torch.float32)
        ys = torch.linspace(domain[0], domain[1], grid_size, dtype=torch.float32)
        X, Y = torch.meshgrid(xs, ys, indexing='ij')
        self.register_buffer("X", X.flatten())
        self.register_buffer("Y", Y.flatten())

    def forward(self, exp_x, exp_y, coeff):
        # Ensure batch
        if exp_x.dim() == 1:
            exp_x = exp_x.unsqueeze(0)
            exp_y = exp_y.unsqueeze(0)
            coeff = coeff.unsqueeze(0)

        X = self.X.unsqueeze(0).unsqueeze(2)  # (1, N, 1)
        Y = self.Y.unsqueeze(0).unsqueeze(2)  # (1, N, 1)
        exp_x = exp_x.unsqueeze(1)            # (B, 1, M)
        exp_y = exp_y.unsqueeze(1)            # (B, 1, M)
        coeff = coeff.unsqueeze(1)            # (B, 1, M)

        nodal_vals = torch.sum(coeff * (X ** exp_x) * (Y ** exp_y), dim=2)  # (B, N)
        max_val = nodal_vals.max(dim=1, keepdim=True)[0].clamp_min(1e-6)
        return nodal_vals / max_val

class ScalesNet(nn.Module):

    def __init__(self,
                 hidden_dim=256,
                 out_len=8,
                 num_nodes=64,
                 domain=(-1, 1),
                 dropout=0.07,
                 num_shared_layers=1,
                 activation="softplus"):
        super().__init__()
        self.nodal = NodalPreprocessor(num_nodes=num_nodes, domain=domain)

        # Shared trunk
        layers = []
        in_dim = num_nodes
        for _ in range(num_shared_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            in_dim = hidden_dim
        self.shared = nn.Sequential(*layers)

        # Heads
        self.head_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_len),
            nn.Identity()  # activation later
        )
        self.head_y = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_len),
            nn.Identity()
        )

        if activation.lower() == "softplus":
            self.out_act = nn.Softplus()
        elif activation.lower() == "tanh":
            self.out_act = nn.Tanh()
        else:
            self.out_act = nn.Identity()

    def forward(self, exp_x, exp_y, coeff):
        z = self.nodal(exp_x, exp_y, coeff)
        f = self.shared(z)
        xhat = self.out_act(self.head_x(f))
        yhat = self.out_act(self.head_y(f))
        return xhat, yhat


def load_model(weights_path=None,
               map_location="cpu",
               **kwargs):

    model = ScalesNet(**kwargs).float()
    if weights_path:
        state = torch.load(weights_path, map_location=map_location)
        # Accept either raw state_dict or full checkpoint
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict)
    return model


def save_checkpoint(model, optimizer, epoch, loss, filename="scales_checkpoint.pth"):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss
    }, filename)
    print(f"Checkpoint saved → {filename}")


def load_checkpoint(model, optimizer, filename="scales_checkpoint.pth", map_location="cpu"):
    ckpt = torch.load(filename, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt["epoch"]
    loss = ckpt["loss"]
    print(f"Checkpoint loaded from {filename} (epoch={epoch}, loss={loss:.6f})")
    return epoch, loss
