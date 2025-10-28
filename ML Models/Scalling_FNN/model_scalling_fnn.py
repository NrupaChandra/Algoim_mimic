import torch
import torch.nn as nn
import numpy as np

class NodalPreprocessor(nn.Module):
    def __init__(self, num_nodes, domain):
        super(NodalPreprocessor, self).__init__()
        self.num_nodes = num_nodes
        self.domain = domain
        self.grid_size = int(np.sqrt(num_nodes))
        if self.grid_size ** 2 != num_nodes:
            raise ValueError("num_nodes must be a perfect square (e.g., 4, 9, 16, ...)")
        xs = torch.linspace(domain[0], domain[1], self.grid_size, dtype=torch.float32)
        ys = torch.linspace(domain[0], domain[1], self.grid_size, dtype=torch.float32)
        X, Y = torch.meshgrid(xs, ys, indexing='ij')
        self.register_buffer("X", X.flatten())
        self.register_buffer("Y", Y.flatten())

    def forward(self, exp_x, exp_y, coeff):
        if exp_x.dim() == 1:
            exp_x = exp_x.unsqueeze(0)
            exp_y = exp_y.unsqueeze(0)
            coeff = coeff.unsqueeze(0)
        X = self.X.unsqueeze(0).unsqueeze(2)  # (1, N, 1)
        Y = self.Y.unsqueeze(0).unsqueeze(2)  # (1, N, 1)
        exp_x = exp_x.unsqueeze(1)            # (B, 1, m)
        exp_y = exp_y.unsqueeze(1)            # (B, 1, m)
        coeff = coeff.unsqueeze(1)            # (B, 1, m)

        nodal = torch.sum(coeff * (X ** exp_x) * (Y ** exp_y), dim=2)
        max_val = nodal.max(dim=1, keepdim=True)[0] + 1e-6
        return nodal / max_val
    


class FeedforwardNN(nn.Module):
    def __init__(self, num_nodes=64, domain =(-1,1)):
        super().__init__()
        self.nodal_preprocessor = NodalPreprocessor(num_nodes=num_nodes, domain=domain)

        self.shared = nn.Sequential(
            nn.Linear(num_nodes, 256),
            nn.ReLU()
        )

        self.scale_x_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.ReLU()
        )

        self.scale_y_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.ReLU()
        )

    def forward(self, exp_x, exp_y, coeff):
        x = self.nodal_preprocessor(exp_x, exp_y, coeff)
        shared = self.shared(x)
        scales_x = self.scale_x_head(shared)
        scales_y = self.scale_y_head(shared)

        return scales_x, scales_y
    
def load_ff_pipelines_model(weights_path= None, num_nodes = 64, domain = (-1,1), map_location=torch.device('cpu') ):
    model = FeedforwardNN(num_nodes=num_nodes, domain=domain).float()
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location=map_location)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from: {weights_path}")
    return model

def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {filename}, Epoch: {epoch}, Loss: {loss:.4f}")
    return epoch, loss



'''class FeedForwardNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_output_len,
                 num_nodes=64, domain=(-1,1),
                 dropout_rate=0.0, num_shared_layers=1):
        super(FeedForwardNN, self).__init__()
        self.nodal_preprocessor = NodalPreprocessor(num_nodes=num_nodes, domain=domain)
        input_dim = num_nodes

        shared_layers = []
        in_dim = input_dim
        for _ in range(num_shared_layers):
            shared_layers.append(nn.Linear(in_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            if dropout_rate > 0:
                shared_layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        self.shared_layer = nn.Sequential(*shared_layers)

        # two heads: 8 scale factors for x, 8 for y
        self.scale_x_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
            nn.Softplus()  # ≥0 scales; switch to Tanh if you want signed factors
        )
        self.scale_y_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
            nn.Softplus()
        )

    def forward(self, exp_x, exp_y, coeff):
        x = self.nodal_preprocessor(exp_x, exp_y, coeff)
        shared = self.shared_layer(x)
        scales_x = self.scale_x_head(shared)  # (B, 8)
        scales_y = self.scale_y_head(shared)  # (B, 8)
        return scales_x, scales_y


def load_ff_pipelines_model(weights_path=None,
                            hidden_dim=256,
                            output_dim=256,
                            max_output_len=64,
                            num_nodes=64,
                            domain=(-1,1),
                            dropout_rate=0.07,
                            num_shared_layers=1,
                            map_location=torch.device('cpu')):
    model = FeedForwardNN(hidden_dim,
                          output_dim,
                          max_output_len,
                          num_nodes,
                          domain,
                          dropout_rate,
                          num_shared_layers).float()
    if weights_path:
        state_dict = torch.load(weights_path, map_location=map_location)
        model.load_state_dict(state_dict)
    return model'''



