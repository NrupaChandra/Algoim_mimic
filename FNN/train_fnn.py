#!/usr/bin/env python
import torch
import random
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from multidataloader_fnn import MultiChunkDataset 
from model_fnn import load_ff_pipelines_model, save_checkpoint, load_checkpoint
import os
import numpy as np
import matplotlib.pyplot as plt

# Set default dtype to single precision.
torch.set_default_dtype(torch.float32)

# Device setup
device = torch.device('cpu')
'''device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}, GPU Name: {torch.cuda.get_device_name(0)}")'''

# Define paths for data and saving model/checkpoints.
data_dir = r"C:\Git\Algoim_mimic\Pre_processing"
model_dir = r"C:\Git\Algoim_mimic\FNN\Model"
os.makedirs(model_dir, exist_ok=True)


#  Loss function

def loss_function(pred_x, pred_y, true_x, true_y):
    dx = pred_x - true_x       
    dy = pred_y - true_y
    dist = dx.pow(2) + dy.pow(2)  
    return torch.sqrt(dist + 1e-12).mean()
        

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

        # ---- Training ----
        model.train()
        train_loss = 0
        for exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, masks in train_dataloader:
            exp_x, exp_y, coeff, true_nodes_x, true_nodes_y = (
                x.to(device) for x in (exp_x, exp_y, coeff, true_nodes_x, true_nodes_y)
            )
            optimizer.zero_grad()
            pred_nodes_x, pred_nodes_y, pred_weights = model(exp_x, exp_y, coeff)

            loss = loss_function(pred_nodes_x, pred_nodes_y, true_nodes_x, true_nodes_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        
        # ---- Validation ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for exp_x, exp_y, coeff, true_nodes_x, true_nodes_y, true_weights, masks in val_dataloader:
                exp_x, exp_y, coeff, true_nodes_x, true_nodes_y = (
                    x.to(device) for x in (exp_x, exp_y, coeff, true_nodes_x, true_nodes_y)
                )
                pred_nodes_x, pred_nodes_y, pred_weights = model(exp_x, exp_y, coeff)
                loss = loss_function(pred_nodes_x, pred_nodes_y, true_nodes_x, true_nodes_y)
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
    seed = 6432
    torch.manual_seed(seed)
    random.seed(seed)
    num_workers = 0
    pin_memory = False
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MultiChunkDataset(
        index_file=os.path.join(data_dir, r'preprocessed_chuncks_10kMonotonic_functions\index.txt'),
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
