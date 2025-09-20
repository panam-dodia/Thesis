#!/usr/bin/env python3
"""
Train two SAEs as professor requested:
SAE-1: On pre-diffusion features (has degradation)
SAE-2: On post-diffusion features (no degradation)
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader

# Add paths
sys.path.insert(0, "../../")
sys.path.insert(0, "../../open_clip")
sys.path.insert(0, "/mnt/DATA/panam/daclip-uir/universal-image-restoration/open_clip")
sys.path.insert(0, "/mnt/DATA/panam/daclip-uir/universal-image-restoration/utils")
# Add other necessary imports based on your project structure

import open_clip
import options as option

class SparseAutoencoder(nn.Module):
    """SAE with 8x expansion"""
    def __init__(self, input_dim=512):
        super().__init__()
        hidden_dim = input_dim * 8  # 512 * 8 = 4096
        
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        latent = torch.relu(self.encoder(x))
        reconstructed = self.decoder(latent)
        return reconstructed, latent

def create_stratified_split(feature_dir, feature_type, train_ratio=0.8, seed=42):
    """Split maintaining ratio per degradation type"""
    random.seed(seed)
    train_files = []
    val_files = []
    
    degradations = ['rainy', 'snowy', 'raindrop', 'low-light', 'shadowed', 'inpainting']
    
    print(f"\nStratified split for {feature_type}:")
    for deg in degradations:
        deg_path = Path(feature_dir) / 'train' / deg / feature_type
        if deg_path.exists():
            files = sorted(list(deg_path.glob('*.pt')))  # Sort for reproducibility
            if files:
                n_total = len(files)
                n_train = int(n_total * train_ratio)
                n_val = n_total - n_train
                
                # Shuffle and split
                random.shuffle(files)
                train_files.extend(files[:n_train])
                val_files.extend(files[n_train:])
                
                print(f"  {deg:12s}: {n_total:5d} total → {n_train:5d} train, {n_val:5d} val")
    
    print(f"  Total: {len(train_files)} train, {len(val_files)} val")
    return train_files, val_files

class StratifiedDataset(Dataset):
    """Dataset that loads from file list"""
    def __init__(self, file_list):
        self.files = file_list
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        features = torch.load(self.files[idx])
        if features.dim() > 1:
            features = features.squeeze()
        return features

def train_sae(model, train_loader, val_loader, device, epochs, lr, lambda_val):
    """Train a single SAE model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    train_losses = []
    val_losses = []
    sparsity_history = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_epoch = []
        
        for features in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            features = features.to(device)
            
            # Forward pass
            reconstructed, latent = model(features)
            
            # Calculate losses
            recon_loss = nn.MSELoss()(reconstructed, features)
            sparsity_loss = torch.mean(torch.abs(latent))
            total_loss = recon_loss + lambda_val * sparsity_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_epoch.append(total_loss.item())
        
        # Validation
        model.eval()
        val_loss_epoch = []
        val_sparsity = []
        
        with torch.no_grad():
            for features in val_loader:
                features = features.to(device)
                
                reconstructed, latent = model(features)
                
                recon_loss = nn.MSELoss()(reconstructed, features)
                sparsity_loss = torch.mean(torch.abs(latent))
                total_loss = recon_loss + lambda_val * sparsity_loss
                
                val_loss_epoch.append(total_loss.item())
                sparsity = (torch.abs(latent) < 0.01).float().mean().item()
                val_sparsity.append(sparsity)
        
        # Calculate epoch metrics
        avg_train_loss = np.mean(train_loss_epoch)
        avg_val_loss = np.mean(val_loss_epoch)
        avg_sparsity = np.mean(val_sparsity) * 100
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        sparsity_history.append(avg_sparsity)
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, "
                  f"Val Loss={avg_val_loss:.6f}, Sparsity={avg_sparsity:.1f}%")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state = model.state_dict().copy()
            best_sparsity = avg_sparsity
        
        scheduler.step()
    
    print(f"  Best epoch: {best_epoch+1}, Val Loss: {best_val_loss:.6f}, Sparsity: {best_sparsity:.1f}%")
    
    return best_state, best_val_loss, best_sparsity, train_losses, val_losses

def main():
    # Configuration
    FEATURE_DIR = "/mnt/DATA/panam/daclip-uir/universal-image-restoration/config/daclip-sde/thesis_features_full_new_original_baseline_20250918_124239"  # UPDATE THIS PATH
    OUTPUT_DIR = Path(f"thesis_trained_sae_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    epochs = 200
    learning_rate = 1e-3
    sparsity_lambdas = [0.001, 0.01, 0.05]
    train_ratio = 0.8
    
    print("="*60)
    print("SAE TRAINING WITH STRATIFIED SPLIT")
    print("="*60)
    print(f"Feature directory: {FEATURE_DIR}")
    print(f"Device: {device}")
    print(f"Train/Val split: {train_ratio*100:.0f}/{(1-train_ratio)*100:.0f}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Sparsity lambdas: {sparsity_lambdas}")
    
    results = {}
    
    # ========== Train SAE-1 (Pre-diffusion) ==========
    print("\n" + "="*60)
    print("TRAINING SAE-1 (Pre-diffusion features)")
    print("="*60)
    
    # Create stratified split
    train_files_pre, val_files_pre = create_stratified_split(
        FEATURE_DIR, 'pre_diffusion', train_ratio
    )
    
    # Create datasets and loaders
    train_dataset_pre = StratifiedDataset(train_files_pre)
    val_dataset_pre = StratifiedDataset(val_files_pre)
    
    train_loader_pre = DataLoader(
        train_dataset_pre, batch_size=batch_size, 
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader_pre = DataLoader(
        val_dataset_pre, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Train with different sparsity levels
    for lambda_val in sparsity_lambdas:
        print(f"\nTraining SAE-1 with λ={lambda_val}")
        
        model = SparseAutoencoder(input_dim=512).to(device)
        
        best_state, best_loss, best_sparsity, train_losses, val_losses = train_sae(
            model, train_loader_pre, val_loader_pre,
            device, epochs, learning_rate, lambda_val
        )
        
        # Save model
        save_path = OUTPUT_DIR / f'sae1_pre_lambda{lambda_val}.pth'
        torch.save({
            'model_state_dict': best_state,
            'val_loss': best_loss,
            'sparsity': best_sparsity,
            'lambda': lambda_val,
            'type': 'pre_diffusion'
        }, save_path)
        
        results[f'SAE1_λ={lambda_val}'] = {
            'val_loss': best_loss,
            'sparsity': best_sparsity
        }
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train', alpha=0.7)
        plt.plot(val_losses, label='Val', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'SAE-1 Training Curves (λ={lambda_val})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(OUTPUT_DIR / f'sae1_lambda{lambda_val}_curves.png')
        plt.close()
    
    # ========== Train SAE-2 (Post-diffusion) ==========
    print("\n" + "="*60)
    print("TRAINING SAE-2 (Post-diffusion features)")
    print("="*60)
    
    # Create stratified split
    train_files_post, val_files_post = create_stratified_split(
        FEATURE_DIR, 'post_diffusion', train_ratio
    )
    
    # Create datasets and loaders
    train_dataset_post = StratifiedDataset(train_files_post)
    val_dataset_post = StratifiedDataset(val_files_post)
    
    train_loader_post = DataLoader(
        train_dataset_post, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader_post = DataLoader(
        val_dataset_post, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Train with different sparsity levels
    for lambda_val in sparsity_lambdas:
        print(f"\nTraining SAE-2 with λ={lambda_val}")
        
        model = SparseAutoencoder(input_dim=512).to(device)
        
        best_state, best_loss, best_sparsity, train_losses, val_losses = train_sae(
            model, train_loader_post, val_loader_post,
            device, epochs, learning_rate, lambda_val
        )
        
        # Save model
        save_path = OUTPUT_DIR / f'sae2_post_lambda{lambda_val}.pth'
        torch.save({
            'model_state_dict': best_state,
            'val_loss': best_loss,
            'sparsity': best_sparsity,
            'lambda': lambda_val,
            'type': 'post_diffusion'
        }, save_path)
        
        results[f'SAE2_λ={lambda_val}'] = {
            'val_loss': best_loss,
            'sparsity': best_sparsity
        }
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train', alpha=0.7)
        plt.plot(val_losses, label='Val', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'SAE-2 Training Curves (λ={lambda_val})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(OUTPUT_DIR / f'sae2_lambda{lambda_val}_curves.png')
        plt.close()
    
    # Save training summary
    with open(OUTPUT_DIR / 'training_summary.json', 'w') as f:
        json.dump({
            'feature_dir': FEATURE_DIR,
            'results': results,
            'train_ratio': train_ratio,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'sparsity_lambdas': sparsity_lambdas,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nResults:")
    for name, metrics in results.items():
        print(f"  {name}: Val Loss={metrics['val_loss']:.6f}, "
              f"Sparsity={metrics['sparsity']:.1f}%")
    print(f"\nModels saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()