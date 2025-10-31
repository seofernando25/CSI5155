"""
Learning rate optimization for AdamW optimizer.
Tests common learning rate values over 50 epochs.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
import numpy as np
from model import ShaderMLP

# Setup output directories
os.makedirs('.cache/frames', exist_ok=True)
os.makedirs('.cache/models', exist_ok=True)

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("-" * 50)

# Load data directly to GPU
print("Loading dataset to GPU...")
data = torch.load('.cache/ds/shader_dataset.pt', map_location=device)
x, y, t, intensity = data['x'], data['y'], data['t'], data['intensity']
inputs = torch.stack([x, y, t], dim=1).float()  # (N, 3) - already on device
targets = intensity.reshape(-1, 1).float()      # (N, 1) - already on device
print(f"Dataset shape: inputs={inputs.shape}, targets={targets.shape}")

# Dataset & split (same as training)
print("Creating datasets...")
full_ds = TensorDataset(inputs, targets)
train_size = int(0.8 * len(full_ds))
val_size = int(0.1 * len(full_ds))
test_size = len(full_ds) - train_size - val_size
train_ds, val_ds, test_ds = random_split(full_ds, [train_size, val_size, test_size])
print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

train_loader = DataLoader(train_ds, batch_size=32768, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32768, num_workers=0)

# Learning rate candidates to test
lr_candidates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
print(f"\nTesting learning rates: {lr_candidates}")
print("=" * 50)

results = []
epochs = 50
criterion = nn.HuberLoss(delta=0.025)

for lr in lr_candidates:
    print(f"\n{'='*50}")
    print(f"Testing lr={lr:.5f}")
    print('='*50)
    
    # Create fresh model for this LR
    model = ShaderMLP().to(device)
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Track losses
    train_losses = []
    val_losses = []
    
    # Train for 50 epochs
    model.train()
    for epoch in range(epochs):
        # Training phase
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        model.train()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    # Store results
    results.append({
        'lr': lr,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'best_val_loss': min(val_losses),
        'best_val_epoch': np.argmin(val_losses)
    })
    
    print(f"Final Val Loss: {val_losses[-1]:.6f}, Best Val Loss: {min(val_losses):.6f}")

# Print summary
print("\n" + "=" * 70)
print("SUMMARY - LEARNING RATE SEARCH")
print("=" * 70)
print(f"{'LR':<15} {'Final Val Loss':<20} {'Best Val Loss':<20} {'Best Epoch':<15}")
print("-" * 70)

sorted_results = sorted(results, key=lambda x: x['best_val_loss'])
for r in sorted_results:
    best_marker = " ✓ BEST" if r == sorted_results[0] else ""
    print(f"{r['lr']:<15.5e} {r['final_val_loss']:<20.6f} {r['best_val_loss']:<20.6f} {r['best_val_epoch']+1:<15}{best_marker}")

print("=" * 70)
best_lr = sorted_results[0]['lr']
print(f"\nBest learning rate: {best_lr:.5f}")
print(f"Best validation loss: {sorted_results[0]['best_val_loss']:.6f}")

# Plot results
print("\nGenerating visualization...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Loss curves for all LRs
ax = axes[0]
for r in results:
    ax.plot(range(1, epochs + 1), r['val_losses'], 
            label=f"lr={r['lr']:.0e}", linewidth=2, alpha=0.8)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Loss', fontsize=12)
ax.set_title('Validation Loss for Different Learning Rates', fontsize=14)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

# Right: Best loss vs LR
ax = axes[1]
lrs = [r['lr'] for r in results]
best_losses = [r['best_val_loss'] for r in results]
ax.plot(lrs, best_losses, 'o-', linewidth=2, markersize=8)
ax.set_xlabel('Learning Rate', fontsize=12)
ax.set_ylabel('Best Validation Loss', fontsize=12)
ax.set_title('Best Validation Loss vs Learning Rate', fontsize=14)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.axvline(best_lr, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_lr:.0e}')

plt.tight_layout()
output_path = '.cache/frames/lr_search.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Visualization saved to {output_path}")

print("\nOptimization complete!")
