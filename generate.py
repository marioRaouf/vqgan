import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from model_components import VQGAN, TransformerGPT
from utils import load_model
from configs import DataConfig, ModelConfig, GenerationConfig, validate_configs, get_latent_size, device

def generate_samples():
    """Generate samples using trained VQ-GAN and Transformer"""
    
    # Validate configurations
    input_size, latent_size = validate_configs()
    
    # Check if checkpoint files exist
    if not os.path.exists(GenerationConfig.VQGAN_CHECKPOINT):
        print(f"Error: VQ-GAN checkpoint not found at {GenerationConfig.VQGAN_CHECKPOINT}")
        return
    
    if not os.path.exists(GenerationConfig.TRANSFORMER_CHECKPOINT):
        print(f"Error: Transformer checkpoint not found at {GenerationConfig.TRANSFORMER_CHECKPOINT}")
        return
    
    # Load VQ-GAN
    vqgan = VQGAN().to(device)
    vqgan = load_model(vqgan, GenerationConfig.VQGAN_CHECKPOINT, device)
    
    # Load Transformer
    transformer = TransformerGPT().to(device)
    transformer = load_model(transformer, GenerationConfig.TRANSFORMER_CHECKPOINT, device)
    
    print("Successfully loaded both pre-trained models")
    print(f"Generating {GenerationConfig.NUM_SAMPLES} samples with temperature {GenerationConfig.TEMPERATURE}")
    
    # Generate samples
    with torch.no_grad():
        # Start with random tokens
        start_tokens = torch.randint(0, ModelConfig.NUM_EMBEDDINGS, 
                                   (GenerationConfig.NUM_SAMPLES, 1)).to(device)
        
        # Generate sequence
        seq_length = get_latent_size(input_size, ModelConfig.NUM_DOWNSAMPLE) ** 2
        print(f"Generating sequence of length {seq_length}")
        
        generated_codes = transformer.generate(start_tokens, max_length=seq_length, 
                                             temperature=GenerationConfig.TEMPERATURE)
        
        # Decode to images
        print("Decoding generated codes to images...")
        generated_images = vqgan.decode(generated_codes)
        generated_images = (generated_images * 0.5) + 0.5
        
        # Plot and save
        os.makedirs(GenerationConfig.OUTPUT_DIR, exist_ok=True)
        
        # Calculate grid size
        num_samples = GenerationConfig.NUM_SAMPLES
        grid_size = int(num_samples ** 0.5)
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        for i in range(num_samples):
            ax = axes[i // grid_size, i % grid_size]
            if generated_images.shape[1] == 1:  # Grayscale
                ax.imshow(generated_images[i].cpu().squeeze(), cmap='gray')
            else:  # RGB
                ax.imshow(generated_images[i].cpu().permute(1, 2, 0))
            ax.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(GenerationConfig.OUTPUT_DIR, 'generated_samples.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Generated {num_samples} samples saved to {output_path}")
    
    return vqgan, transformer

if __name__ == "__main__":
    vqgan, transformer = generate_samples()