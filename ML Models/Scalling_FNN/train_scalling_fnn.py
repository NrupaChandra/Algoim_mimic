#!/usr/bin/env python
import torch
import random
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from multidataloader_fnn import MultiChunkDataset 
from model_scalling_fnn import load_ff_pipelines_model, save_checkpoint, load_checkpoint
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Set default dtype to single precision.
torch.set_default_dtype(torch.float64)

# Device setup
device = torch.device('cpu')

# Define paths for data and saving model/checkpoints.
data_dir = r"C:\Git\Algoim_mimic\Pre_processing"
model_dir = r"C:\Git\Algoim_mimic\ML Models\Scalling_FNN\Model"
os.makedirs(model_dir, exist_ok=True)


#  Loss

def loss_function(Ts_x, Ts_y, Ps_x, Ps_y, reduction="mean"):

    B = Ts_x.size(0)
    Loss_scales_x = torch.zeros(B, 8, device=Ts_x.device)
    Loss_scales_y = torch.zeros(B, 8, device=Ts_y.device)

    for i in range(8):
        Loss_scales_x[:, i] = torch.abs(Ts_x[:, i] - Ps_x[:, i])
        Loss_scales_y[:, i] = torch.abs(Ts_y[:, i] - Ps_y[:, i])

    per_sample = (Loss_scales_x.sum(dim=1) + Loss_scales_y.sum(dim=1))  
    return per_sample.mean() if reduction == "mean" else per_sample.sum()  

# Training Function

def train_fnn(model, train_dataloader, val_dataloader, optimizer, epochs=1000, checkpoint_path=None, save_every=5):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(model_dir, "fnn_checkpoint.pth")
    model.to(device)
    train_losses = []
    val_losses = []
    epoch_list = []
    epoch_times = []  # To store the time taken per epoch
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print("Checkpoint found. Loading checkpoint...")
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        print("No checkpoint found. Starting training from epoch 1.")
    
    total_steps = epochs * len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=total_steps, 
                                                    pct_start=0.1, anneal_strategy='linear', 
                                                    final_div_factor=100, verbose=True)
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        #torch.cuda.empty_cache()
        epoch_list.append(epoch + 1)

        # Training
        model.train()
        train_loss = 0
        for exp_x, exp_y, coeff, true_scales_x, true_scales_y, masks, sample_ids in train_dataloader:
            exp_x, exp_y, coeff, true_scales_x, true_scales_y = (
                x.to(device) for x in (exp_x, exp_y, coeff, true_scales_x, true_scales_y)
            )
            optimizer.zero_grad()
            pred_scales_x , pred_scales_y = model(exp_x, exp_y, coeff)

            loss = loss_function(true_scales_x, true_scales_y, pred_scales_x, pred_scales_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        
        # Validation 
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for exp_x, exp_y, coeff, true_scales_x, true_scales_y, masks, sample_ids in val_dataloader:
                exp_x, exp_y, coeff, true_scales_x, true_scales_y = (
                    x.to(device) for x in (exp_x, exp_y, coeff,  true_scales_x, true_scales_y)
                )
                pred_scales_x , pred_scales_y = model(exp_x, exp_y, coeff)
                loss = loss_function(true_scales_x, true_scales_y, pred_scales_x, pred_scales_y)
                val_loss += loss.item()
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        print(f"\nEpoch {epoch + 1}/{epochs}: Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, Time: {epoch_time:.2f} sec")
        
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, epoch + 1, train_loss, checkpoint_path)
    
    final_model_path = os.path.join(model_dir, 'fnn_model_weights_v6.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Model weights saved: {final_model_path}")
    
    return epoch_list, train_losses, val_losses, epoch_times


# Main function

if __name__ == "__main__":
    seed = 6464
    torch.manual_seed(seed)
    random.seed(seed)
    num_workers = 0
    pin_memory = False
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MultiChunkDataset(
        index_file=os.path.join(data_dir, r'100kpreprocessed_chunks_weight_scaled\index.txt'),
        base_dir=data_dir
    )
    
    print("Dataset length:", len(dataset))
    if len(dataset) == 0:
        raise ValueError("The dataset is empty. Please verify your index file and data directory.")
    
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    model = load_ff_pipelines_model()
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1.0e-05)
    
    epochs = 1000
    epoch_list, train_losses, val_losses, epoch_times = train_fnn(
        model, train_dataloader, val_dataloader,
        optimizer, epochs=epochs, checkpoint_path=os.path.join(model_dir, "fnn_checkpoint.pth"), save_every=5
    )
    
    plt.figure(figsize=(10,5))
    plt.plot(epoch_list, train_losses, label="Training Loss")
    plt.plot(epoch_list, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.title("Training and Validation Loss (Log Scale)")
    plt.savefig(os.path.join(model_dir, "loss_curves.png"), dpi=300)
    plt.show()
    
    avg_epoch_time = np.mean(epoch_times)
    print(f"\nAverage epoch time: {avg_epoch_time:.2f} seconds")