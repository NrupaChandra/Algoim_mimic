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
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,8)
        )

        self.scale_y_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,8)
        )

        self.scale_cx_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,8),
            nn.Tanh()
        )

        self.scale_cy_head = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128,8),
                    nn.Tanh()
                )

    def forward(self, exp_x, exp_y, coeff):
        x = self.nodal_preprocessor(exp_x, exp_y, coeff)
        shared = self.shared(x)
        scales_x = self.scale_x_head(shared)
        scales_y = self.scale_y_head(shared)
        center_x = self.scale_cx_head(shared)
        center_y = self.scale_cy_head(shared)

        return (scales_x, scales_y, center_x, center_y)
    
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


'''
model = FeedforwardNN(num_nodes=64, domain=(-1, 1))
B, m = 2, 4  # two samples, four terms each

exp_x = torch.tensor([[0., 1., 0., 1.],
                      [1., 2., 1., 0.]], dtype=torch.float32)
exp_y = torch.tensor([[0., 0., 1., 1.],
                      [1., 0., 2., 1.]], dtype=torch.float32)
coeff = torch.tensor([[1.0, -0.5, 0.3, 0.2],
                      [0.8, 0.1, -0.4, 0.5]], dtype=torch.float32)


scales_x, scales_y = model(exp_x, exp_y, coeff)

print("scales_x shape:", scales_x.shape)
print("scales_y shape:", scales_y.shape)

print("\nscales_x sample:\n", scales_x)
print("\nscales_y sample:\n", scales_y)

'''
