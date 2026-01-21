#!/usr/bin/env python
import torch
import random
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from multidataloader_fnn import MultiChunkDataset 
from model_scalling_fnn_v2 import load_ff_pipelines_model, save_checkpoint, load_checkpoint
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# -----------------------------
# Device setup
# -----------------------------
device = torch.device('cpu')
torch.set_num_threads(4)

# -----------------------------
# Paths
# -----------------------------
data_dir = r"C:\Git\Algoim_mimic\Pre_processing"
model_dir = r"C:\Git\Algoim_mimic\ML Models\Scalling_FNN\Model"
os.makedirs(model_dir, exist_ok=True)

# -----------------------------
# Loss
# -----------------------------
def loss_function(
    Ts_x, Ts_y, Tc_x, Tc_y,
    Ps_x, Ps_y, Pc_x, Pc_y
):
    # per-sample L1 sums
    L_sx = torch.abs(Ts_x - Ps_x).sum(dim=1)
    L_sy = torch.abs(Ts_y - Ps_y).sum(dim=1)
    L_cx = torch.abs(Tc_x - Pc_x).sum(dim=1)
    L_cy = torch.abs(Tc_y - Pc_y).sum(dim=1)

    per_sample = (L_sx +L_sy + 0.5 * L_cx + 0.5 *L_cy)
    return per_sample.mean()


# -----------------------------
# Training function
# -----------------------------
def train_fnn(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    epochs=1000,
    checkpoint_path=None,
    save_every=5
):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(model_dir, "fnn_checkpoint_v2.pth")

    model.to(device)

    train_losses = []
    val_losses = []
    epoch_list = []
    epoch_times = []

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print("Checkpoint found. Loading checkpoint...")
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        print("No checkpoint found. Starting training from epoch 1.")

    total_steps = epochs * len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='linear',
        final_div_factor=100
    )

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        epoch_list.append(epoch + 1)

        print(f"\nEpoch {epoch+1}/{epochs}  (lr={optimizer.param_groups[0]['lr']:.2e})")

        # -------------------------
        # Training
        # -------------------------
        model.train()
        train_loss = 0.0

        for i, batch in enumerate(train_dataloader):
            t0 = time.time()

            exp_x, exp_y, coeff, Ts_x, Ts_y, Tc_x, Tc_y = batch[:7]

            exp_x = exp_x.to(device).float()
            exp_y = exp_y.to(device).float()
            coeff = coeff.to(device).float()
            Ts_x = Ts_x.to(device).float()
            Ts_y = Ts_y.to(device).float()
            Tc_x = Tc_x.to(device).float()
            Tc_y = Tc_y.to(device).float()

            optimizer.zero_grad(set_to_none=True)

            Ps_x, Ps_y, Pc_x, Pc_y = model(exp_x, exp_y, coeff)

            loss = loss_function(Ts_x, Ts_y, Tc_x, Tc_y,
                                 Ps_x, Ps_y, Pc_x, Pc_y)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss detected")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            if i % 10 == 0:
                print(
                    f"  batch {i:4d}/{len(train_dataloader)}  "
                    f"loss={loss.item():.6f}  "
                    f"step_time={time.time()-t0:.3f}s"
                )

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                exp_x, exp_y, coeff, Ts_x, Ts_y, Tc_x, Tc_y = batch[:7]

                exp_x = exp_x.to(device).float()
                exp_y = exp_y.to(device).float()
                coeff = coeff.to(device).float()
                Ts_x = Ts_x.to(device).float()
                Ts_y = Ts_y.to(device).float()
                Tc_x = Tc_x.to(device).float()
                Tc_y = Tc_y.to(device).float()

                Ps_x, Ps_y, Pc_x, Pc_y = model(exp_x, exp_y, coeff)

                loss = loss_function(Ts_x, Ts_y, Tc_x, Tc_y,
                                     Ps_x, Ps_y, Pc_x, Pc_y)

                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

        epoch_times.append(time.time() - epoch_start_time)

        print(
            f"Epoch {epoch+1} done | "
            f"train={train_loss:.6f}  "
            f"val={val_loss:.6f}  "
            f"time={epoch_times[-1]:.2f}s"
        )

        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)

    return epoch_list, train_losses, val_losses, epoch_times

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    seed = 6464
    torch.manual_seed(seed)
    random.seed(seed)

    dataset = MultiChunkDataset(
        index_file=os.path.join(
            data_dir,
            r'100kpreprocessed_chunks_scale_center\index.txt'
        ),
        base_dir=data_dir
    )

    print("Dataset length:", len(dataset))
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    model = load_ff_pipelines_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    epochs = 400
    epoch_list, train_losses, val_losses, epoch_times = train_fnn(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        epochs=epochs,
        checkpoint_path=os.path.join(model_dir, "fnn_checkpoint_v2.pth")
    )
    torch.save(
    model.state_dict(),
    os.path.join(model_dir, "fnn_model_weights_v7.pth")
    )


    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_list, train_losses, label="Train")
    plt.plot(epoch_list, val_losses, label="Validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "loss_curves.png"), dpi=300)
    plt.show()

    print(f"\nAverage epoch time: {np.mean(epoch_times):.2f} s")
