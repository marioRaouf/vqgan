import torch
from configs import device
from model_components import VQGAN, TransformerGPT
import matplotlib.pyplot as plt

def generate_with_transformer(vqgan, transformer, epoch, latent_size, num_samples=16):
    """Generate images using the trained transformer"""
    vqgan.eval()
    transformer.eval()
    
    with torch.no_grad():
        # Start with random tokens
        start_tokens = torch.randint(0, vqgan.vector_quantization.num_embeddings, 
                                   (num_samples, 1)).to(device)
        
        # Generate sequence
        seq_length = latent_size * latent_size
        generated_codes = transformer.generate(start_tokens, max_length=seq_length, temperature=0.9)
        
        # Decode to images
        generated_images = vqgan.decode(generated_codes)
        generated_images = (generated_images * 0.5) + 0.5
        
        # Plot
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(16):
            ax = axes[i // 4, i % 4]
            ax.imshow(generated_images[i].cpu().squeeze(), cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'transformer_generated_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.show()



def load_pretrained_vqgan(model_path, device):
    """Load pre-trained VQ-GAN model"""
    # Recreate the same architecture
    vqgan = VQGAN(
        in_channels=1,
        base_channels=64,
        num_downsample=2,
        num_embeddings=256,
        embedding_dim=32
    ).to(device)
    
    # Load saved weights
    checkpoint = torch.load(model_path, map_location=device)
    vqgan.load_state_dict(checkpoint)
    vqgan.eval()
    
    print(f"Loaded pre-trained VQ-GAN from {model_path}")
    return vqgan
# ==================== GENERATION WITH PRE-TRAINED MODELS ====================

def generate_with_pretrained_models(vqgan_model_path, transformer_model_path, num_samples=16):
    """Generate images using pre-trained VQ-GAN and Transformer"""
    
    # Load VQ-GAN
    vqgan = load_pretrained_vqgan(vqgan_model_path, device)
    
    # Calculate parameters
    latent_size = 7
    vocab_size = vqgan.vector_quantization.num_embeddings
    
    # Load transformer
    transformer = TransformerGPT(
        vocab_size=vocab_size, 
        d_model=128, 
        nhead=8, 
        num_layers=3,
        max_seq_length=latent_size * latent_size
    ).to(device)
    
    transformer.load_state_dict(torch.load(transformer_model_path, map_location=device))
    transformer.eval()
    
    print("Loaded both pre-trained models")
    
    # Generate samples
    generate_with_transformer(vqgan, transformer, "final", latent_size, num_samples)
    
    return vqgan, transformer

vqgan, transformer = generate_with_pretrained_models(
        '/kaggle/input/vqgan/pytorch/default/1/vqgan_final_thurthday.pth',
        '/kaggle/input/vqgan/pytorch/default/1/transformer_final_thurthday.pth'
    )