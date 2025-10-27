#!/usr/bin/env python
import os
import math
import time
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from model_scalling_fnn import load_model, save_checkpoint, load_checkpoint

# ---------------- Config ----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
seed = 6432
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# Adjust these to your preprocessed directory
DATA_DIR = r"C:\Git\Algoim_mimic\Pre_processing\10kpreprocessed_chunks_weight_scaled"
INDEX_TXT = os.path.join(DATA_DIR, "index.txt")

MODEL_DIR = r"C:\Git\Algoim_mimic\ML Models\Scalling_FNN\Model"

os.makedirs(MODEL_DIR, exist_ok=True)
CKPT_PATH = os.path.join(MODEL_DIR, "scales_checkpoint.pth")
FINAL_WEIGHTS = os.path.join(MODEL_DIR, "scales_model_weights.pth")


# -------------- Dataset (chunk-aware, memory-friendly) --------------
from torch.utils.data import Dataset
import torch

class ScalesMultiChunkDataset(Dataset):
    """
    Each .pt chunk is a list of tuples:
      (exp_x, exp_y, coeff, xscales, yscales, scales2d_or_None, id)
    """
    def __init__(self, index_file, preload=False):
        super().__init__()
        with open(index_file, 'r') as f:
            self.paths = [ln.strip() for ln in f if ln.strip()]
        if not self.paths:
            raise FileNotFoundError("No chunk files listed in index.txt")

        self.preload = preload
        self._chunks = [None] * len(self.paths)
        self._sizes  = []
        self._cum    = []

        total = 0
        for i, p in enumerate(self.paths):
            if self.preload:
                ch = torch.load(p, map_location="cpu")  # keep it cached
                n = len(ch)
                self._chunks[i] = ch
            else:
                tmp = torch.load(p, map_location="cpu")  # peek only
                n = len(tmp)
                del tmp
                self._chunks[i] = None  # lazy-load later

            self._sizes.append(n)
            total += n
            self._cum.append(total)

        self._len = total

    def __len__(self):
        return self._len

    def _locate(self, idx):
        # binary search on cumulative sizes
        lo, hi = 0, len(self._cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cum[mid]:
                hi = mid
            else:
                lo = mid + 1
        chunk_idx = lo
        local_idx = idx if chunk_idx == 0 else idx - self._cum[chunk_idx - 1]
        return chunk_idx, local_idx

    def _get_chunk(self, chunk_idx):
        ch = self._chunks[chunk_idx]
        if ch is None:
            ch = torch.load(self.paths[chunk_idx], map_location="cpu")
            self._chunks[chunk_idx] = ch  # cache for reuse
        return ch

    def __getitem__(self, idx):
        cidx, lidx = self._locate(idx)
        chunk = self._get_chunk(cidx)
        exp_x, exp_y, coeff, xscales, yscales, _2d, _id = chunk[lidx]
        return exp_x, exp_y, coeff, xscales, yscales


# ---------------- Loss ----------------
class ScalesLoss(nn.Module):
    """
    Simple L2 loss on xscales and yscales, with optional small L2 regularizer.
    """
    def __init__(self, lambda_reg=0.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_reg = lambda_reg

    def forward(self, xhat, yhat, xtrue, ytrue, model=None):
        loss = self.mse(xhat, xtrue) + self.mse(yhat, ytrue)
        if model is not None and self.lambda_reg > 0:
            reg = torch.tensor(0., device=xhat.device)
            for p in model.parameters():
                reg = reg + p.pow(2).sum()
            loss = loss + self.lambda_reg * reg
        return loss


# ---------------- Train / Val loops ----------------
def run_epoch(model, loader, optimizer=None, criterion=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total = 0.0
    n_batches = 0
    for exp_x, exp_y, coeff, xs, ys in loader:
        exp_x = exp_x.to(device)
        exp_y = exp_y.to(device)
        coeff = coeff.to(device)
        xs    = xs.to(device)
        ys    = ys.to(device)

        with torch.set_grad_enabled(is_train):
            xhat, yhat = model(exp_x, exp_y, coeff)
            loss = criterion(xhat, yhat, xs, ys, model if is_train else None)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total += loss.item()
        n_batches += 1

    return total / max(n_batches, 1)


def train(epochs=300,
          batch_size=256,
          lr=1e-3,
          weight_decay=5e-5,
          hidden_dim=256,
          out_len=8,
          num_nodes=64,
          dropout=0.07,
          num_shared_layers=1,
          activation="softplus",
          resume=True):

    # Data
    full = ScalesMultiChunkDataset(INDEX_TXT, preload=False)
    n = len(full)
    n_train = int(0.8 * n)
    n_val   = n - n_train
    train_set, val_set = random_split(full, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Model
    model = load_model(
        weights_path=None,
        map_location=device,
        hidden_dim=hidden_dim,
        out_len=out_len,
        num_nodes=num_nodes,
        dropout=dropout,
        num_shared_layers=num_shared_layers,
        activation=activation
    ).to(device)

    # Opt & sched
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = ScalesLoss(lambda_reg=0.0)

    start_epoch = 0
    if resume and os.path.exists(CKPT_PATH):
        try:
            start_epoch, _ = load_checkpoint(model, optimizer, CKPT_PATH, map_location=device)
        except Exception as e:
            print(f"Could not resume from checkpoint: {e}")

    total_steps = epochs * max(len(train_loader), 1)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=max(total_steps, 1),
        pct_start=0.1,
        anneal_strategy='linear',
        final_div_factor=100,
        three_phase=False
    )

    # Train
    best_val = math.inf
    for ep in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, criterion)
        val_loss   = run_epoch(model, val_loader, optimizer=None, criterion=criterion)
        if total_steps > 0:
            # Step scheduler once per epoch (already stepped per batch above would require integrating; we keep simple)
            pass

        dt = time.time() - t0
        print(f"Epoch {ep+1:04d}/{epochs} | train={train_loss:.6f} | val={val_loss:.6f} | {dt:.1f}s")

        # Save best
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            save_checkpoint(model, optimizer, ep + 1, val_loss, CKPT_PATH)

    # Save final weights
    torch.save(model.state_dict(), FINAL_WEIGHTS)
    print(f"Saved final weights → {FINAL_WEIGHTS}")


if __name__ == "__main__":
    train(
        epochs=400,
        batch_size=256,
        lr=1.0e-3,
        weight_decay=5.0e-5,
        hidden_dim=256,
        out_len=8,         # 8 scale factors per axis
        num_nodes=64,      # 8x8 nodal feature
        dropout=0.07,
        num_shared_layers=1,
        activation="softplus",  # use "tanh" if your targets are in [-1, 1]
        resume=True
    )
