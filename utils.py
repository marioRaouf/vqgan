import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from configs import device

def save_model(model, filepath, is_best=False):
    """Save model state_dict (direct, not wrapped in checkpoint dict)"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(model.state_dict(), filepath)
    
    if is_best:
        best_path = filepath.replace('.pth', '_best.pth')
        torch.save(model.state_dict(), best_path)

def load_model(model, filepath, device=device):
    """Load model from direct state_dict (matching your saving format)"""
    if filepath is None or not os.path.exists(filepath):
        print(f"Model not found at {filepath}")
        return model
    
    # Load the state_dict directly (not wrapped in a checkpoint dict)
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Loaded model from {filepath}")
    return model

def extract_latent_codes(vqgan, dataloader, device=device):
    """Extract latent codes from the trained VQ-GAN for transformer training"""
    vqgan.eval()
    all_codes = []
    
    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc="Extracting latent codes"):
            data = data.to(device)
            encoding_indices = vqgan.encode(data)
            all_codes.append(encoding_indices.cpu())
    
    all_codes = torch.cat(all_codes, dim=0)
    print(f"Extracted latent codes shape: {all_codes.shape}")
    return all_codes

def plot_training_history(train_losses, val_losses, save_path=None):
    """Plot training and validation history"""
    epochs = [x['epoch'] for x in train_losses]
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(epochs, [x['recon_loss'] for x in train_losses], label='Train')
    plt.plot(epochs, [x['recon_loss'] for x in val_losses], label='Val')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.plot(epochs, [x['vq_loss'] for x in train_losses], label='Train')
    plt.plot(epochs, [x['vq_loss'] for x in val_losses], label='Val')
    plt.title('VQ Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if 'd_loss' in train_losses[0]:
        plt.subplot(2, 3, 3)
        plt.plot(epochs, [x['d_loss'] for x in train_losses], label='D Loss')
        plt.plot(epochs, [x['g_loss'] for x in train_losses], label='G Loss')
        plt.title('Discriminator/Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    if 'perplexity' in train_losses[0]:
        plt.subplot(2, 3, 4)
        plt.plot(epochs, [x['perplexity'] for x in train_losses])
        plt.title('Codebook Perplexity')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def save_sample_images(original, reconstructed, epoch, save_dir):
    """Save sample images for comparison"""
    os.makedirs(save_dir, exist_ok=True)
    
    original = (original * 0.5) + 0.5
    reconstructed = (reconstructed * 0.5) + 0.5
    
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(original[i].cpu().squeeze(), cmap='gray')
        axes[1, i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'reconstruction_epoch_{epoch}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()