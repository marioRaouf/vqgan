import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from model_components import VQGAN, PatchDiscriminator, TransformerGPT
from dataloaders import get_mnist_loaders, get_custom_loaders
from utils import save_model, load_model, extract_latent_codes, plot_training_history, save_sample_images
from configs import DataConfig, ModelConfig, Stage1TrainConfig, Stage2TrainConfig, validate_configs, device, TrainingCoordinator

def train_stage1():
    """Train VQ-GAN (Stage 1)"""
    
    # Validate configurations
    input_size, latent_size = validate_configs()
    print(f"Training Stage 1 with input size {input_size}x{input_size}, latent size {latent_size}x{latent_size}")
    
    # Get dataloaders
    if DataConfig.DATASET_TYPE == 'mnist':
        train_loader, val_loader = get_mnist_loaders()
    else:
        train_loader, val_loader = get_custom_loaders()
    
    # Initialize models
    vqgan = VQGAN().to(device)
    discriminator = PatchDiscriminator().to(device)
    
    # Optimizers
    optimizer_g = optim.Adam(vqgan.parameters(), 
                           lr=Stage1TrainConfig.LEARNING_RATE, 
                           betas=(0.5, 0.9))
    optimizer_d = optim.Adam(discriminator.parameters(), 
                           lr=Stage1TrainConfig.LEARNING_RATE, 
                           betas=(0.5, 0.9))
    
    # Training history
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    
    for epoch in range(Stage1TrainConfig.NUM_EPOCHS):
        # Training
        vqgan.train()
        discriminator.train()
        
        total_recon_loss = 0
        total_vq_loss = 0
        total_d_loss = 0
        total_g_loss = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}")):
            data = data.to(device)
            
            # Train Discriminator
            optimizer_d.zero_grad()
            
            vq_loss, recon_data, perplexity, _ = vqgan(data)
            
            real_output = discriminator(data)
            fake_output = discriminator(recon_data.detach())
            
            d_loss_real = nn.functional.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output))
            d_loss_fake = nn.functional.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))
            d_loss = (d_loss_real + d_loss_fake) / 2
            
            d_loss.backward()
            optimizer_d.step()
            
            # Train Generator (VQ-GAN)
            optimizer_g.zero_grad()
            
            fake_output = discriminator(recon_data)
            g_loss_adv = nn.functional.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output))
            
            recon_loss = nn.functional.mse_loss(recon_data, data)
            
            g_loss = Stage1TrainConfig.RECON_WEIGHT * recon_loss + vq_loss + Stage1TrainConfig.ADV_WEIGHT * g_loss_adv
            
            g_loss.backward()
            optimizer_g.step()
            
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
        
        # Calculate average training losses
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_vq_loss = total_vq_loss / len(train_loader)
        avg_d_loss = total_d_loss / len(train_loader)
        avg_g_loss = total_g_loss / len(train_loader)
        
        # Validation
        vqgan.eval()
        val_recon_loss = 0
        val_vq_loss = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                vq_loss, recon_data, perplexity, _ = vqgan(data)
                recon_loss = nn.functional.mse_loss(recon_data, data)
                
                val_recon_loss += recon_loss.item()
                val_vq_loss += vq_loss.item()
        
        val_recon_loss /= len(val_loader)
        val_vq_loss /= len(val_loader)
        val_total_loss = val_recon_loss + val_vq_loss
        
        # Save training history
        train_losses.append({
            'epoch': epoch + 1,
            'recon_loss': avg_recon_loss,
            'vq_loss': avg_vq_loss,
            'd_loss': avg_d_loss,
            'g_loss': avg_g_loss,
            'perplexity': perplexity.item()
        })
        
        val_losses.append({
            'epoch': epoch + 1,
            'recon_loss': val_recon_loss,
            'vq_loss': val_vq_loss
        })
        
        print(f'Epoch [{epoch+1}/{Stage1TrainConfig.NUM_EPOCHS}]')
        print(f'  Train - Recon: {avg_recon_loss:.4f}, VQ: {avg_vq_loss:.4f}, D: {avg_d_loss:.4f}, G: {avg_g_loss:.4f}')
        print(f'  Val   - Recon: {val_recon_loss:.4f}, VQ: {val_vq_loss:.4f}')
        print(f'  Perplexity: {perplexity:.4f}')
        
        # Save models (direct state_dict format)
        if Stage1TrainConfig.SAVE_LAST:
            save_model(vqgan, os.path.join(Stage1TrainConfig.SAVE_DIR, 'vqgan_last.pth'))
        
        if Stage1TrainConfig.SAVE_BEST and val_total_loss < best_loss:
            best_loss = val_total_loss
            save_model(vqgan, os.path.join(Stage1TrainConfig.SAVE_DIR, 'vqgan_best.pth'), is_best=True)
        
        # Save samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                vqgan.eval()
                sample = next(iter(val_loader))[0][:8].to(device)
                _, recon_sample, _, _ = vqgan(sample)
                save_sample_images(sample, recon_sample, epoch+1, Stage1TrainConfig.SAVE_DIR)
    
    # Plot training history
    plot_training_history(train_losses, val_losses, 
                         os.path.join(Stage1TrainConfig.SAVE_DIR, 'training_history.png'))
    
    return vqgan

def train_stage2(vqgan_checkpoint_path=None):
    """Train Transformer (Stage 2) using pre-trained VQ-GAN"""
    
    # Validate configurations
    input_size, latent_size = validate_configs()
    print(f"Training Stage 2 with input size {input_size}x{input_size}, latent size {latent_size}x{latent_size}")
    
    # Use provided path or default
    if vqgan_checkpoint_path is None:
        vqgan_checkpoint_path = './models/stage1/vqgan_best.pth'
    
    # Load pre-trained VQ-GAN
    vqgan = VQGAN().to(device)
    print(f"Loading VQ-GAN from: {vqgan_checkpoint_path}")
    vqgan = load_model(vqgan, vqgan_checkpoint_path, device)
    vqgan.eval()
    
    # Get dataloaders
    if DataConfig.DATASET_TYPE == 'mnist':
        train_loader, _ = get_mnist_loaders()
    else:
        train_loader, _ = get_custom_loaders()
    
    # Extract latent codes
    print("Extracting latent codes from training data...")
    latent_codes = extract_latent_codes(vqgan, train_loader, device)
    
    # Create transformer
    transformer = TransformerGPT().to(device)
    optimizer = optim.Adam(transformer.parameters(), lr=Stage2TrainConfig.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Prepare data for transformer training
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(latent_codes)
    dataloader = DataLoader(dataset, batch_size=DataConfig.BATCH_SIZE, shuffle=True)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(Stage2TrainConfig.NUM_EPOCHS):
        transformer.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Stage2 Epoch {epoch+1}"):
            codes = batch[0].to(device)
            codes = codes.transpose(0, 1)
            
            input_seq = codes[:-1, :]
            target_seq = codes[1:, :]
            
            optimizer.zero_grad()
            logits = transformer(input_seq)
            
            loss = criterion(logits.reshape(-1, ModelConfig.NUM_EMBEDDINGS), target_seq.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Stage2 Epoch [{epoch+1}/{Stage2TrainConfig.NUM_EPOCHS}], Loss: {avg_loss:.4f}')
        
        # Save models (direct state_dict format)
        if Stage2TrainConfig.SAVE_LAST:
            save_model(transformer, os.path.join(Stage2TrainConfig.SAVE_DIR, 'transformer_last.pth'))
        
        if Stage2TrainConfig.SAVE_BEST and avg_loss < best_loss:
            best_loss = avg_loss
            save_model(transformer, os.path.join(Stage2TrainConfig.SAVE_DIR, 'transformer_best.pth'), is_best=True)
    
    return transformer, vqgan

if __name__ == "__main__":
    if TrainingCoordinator.train_vqgan:
      vqgan = train_stage1()
      vqgan_checkpoint_path = ""
    else:
      if TrainingCoordinator.vqgan_checkpoint_path is None or not os.path.exists(TrainingCoordinator.vqgan_checkpoint_path):
        print("can't initiate traning with pretrained vqgan checkpoint please provie a valid checkpoint path for vqgan") 

    if TrainingCoordinator.train_transformer:
        Stage2TrainConfig.RESUME_CHECKPOINT = TrainingCoordinator.vqgan_checkpoint_path
        transformer = train_stage2()

