import torch
import math

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== DATASET CONFIGURATIONS ====================

class DataConfig:
    # Dataset Type
    DATASET_TYPE = 'mnist'  # 'mnist' or 'custom'
    
    # MNIST Configuration
    MNIST_PATH = './data'
    MNIST_IMAGE_SIZE = 28
    MNIST_GRAYSCALE = True
    MNIST_TRAIN_SAMPLES = 60000
    MNIST_VAL_SAMPLES = 10000
    
    # Custom Dataset Configuration
    CUSTOM_DATA_PATH = "path/to/your/custom_dataset"
    CUSTOM_IMAGE_SIZE = 128
    CUSTOM_GRAYSCALE = False
    
    # DataLoader settings
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    PIN_MEMORY = True

# ==================== MODEL HYPERPARAMETERS ====================

class ModelConfig:
    # VQ-GAN Parameters
    BASE_CHANNELS = 64
    NUM_DOWNSAMPLE = 2
    NUM_EMBEDDINGS = 256
    EMBEDDING_DIM = 32
    COMMITMENT_COST = 0.25
    
    # Discriminator Parameters
    DISC_BASE_CHANNELS = 64
    DISC_LAYERS = 3
    
    # Transformer Parameters
    TRANSFORMER_D_MODEL = 128
    TRANSFORMER_NHEAD = 8
    TRANSFORMER_LAYERS = 3
    TRANSFORMER_DROPOUT = 0.1

# ==================== TRAINING CONFIGURATIONS ====================

class TrainingCoordinator:
    train_vqgan= False
    train_transformer= True
    vqgan_checkpoint_path = "/content/drive/MyDrive/vqgan_thurthday/checkpoints/vqgan_final_thurthday.pth"

class Stage1TrainConfig:
    # Training Parameters
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 1
    RECON_WEIGHT = 1.0
    ADV_WEIGHT = 0.1
    
    # Resume Training
    RESUME = False
    RESUME_CHECKPOINT = None
    
    # Save Settings
    SAVE_DIR = './models/stage1'
    SAVE_BEST = True
    SAVE_LAST = True

class Stage2TrainConfig:
    # Training Parameters
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 1
    
    # Resume Training
    RESUME = False
    RESUME_CHECKPOINT = None
    
    # Save Settings
    SAVE_DIR = './models/stage2'
    SAVE_BEST = True
    SAVE_LAST = True

# ==================== GENERATION CONFIGURATIONS ====================

class GenerationConfig:
    VQGAN_CHECKPOINT = '/content/drive/MyDrive/vqgan_thurthday/checkpoints/vqgan_final_thurthday.pth'
    TRANSFORMER_CHECKPOINT = '/content/drive/MyDrive/vqgan_thurthday/checkpoints/transformer_final_thurthday.pth'
    NUM_SAMPLES = 16
    TEMPERATURE = 0.9
    OUTPUT_DIR = './generated'

# ==================== CALCULATION HELPERS ====================

def calculate_downsample_count(input_size, target_latent_size=7):
    """Calculate number of downsample steps needed to reach target latent size"""
    downsample_count = 0
    current_size = input_size
    while current_size > target_latent_size * 2:
        current_size = current_size // 2
        downsample_count += 1
    return downsample_count

def get_latent_size(input_size, num_downsample):
    """Calculate latent size based on input size and number of downsamples"""
    return input_size // (2 ** num_downsample)

def get_sequence_length(input_size, num_downsample):
    """Calculate sequence length for transformer"""
    latent_size = get_latent_size(input_size, num_downsample)
    return latent_size * latent_size

# ==================== CONFIG VALIDATION ====================

def validate_configs():
    """Validate configuration settings"""
    # Validate dataset type
    if DataConfig.DATASET_TYPE not in ['mnist', 'custom']:
        raise ValueError(f"Invalid DATASET_TYPE: {DataConfig.DATASET_TYPE}")
    
    # Validate image sizes are powers of 2 for clean downsampling
    input_size = DataConfig.MNIST_IMAGE_SIZE if DataConfig.DATASET_TYPE == 'mnist' else DataConfig.CUSTOM_IMAGE_SIZE
    if not (input_size & (input_size - 1) == 0) and input_size != 0:
        print(f"Warning: Input size {input_size} is not a power of 2, may cause issues with downsampling")
    
    # Calculate and validate downsample count
    num_downsample = ModelConfig.NUM_DOWNSAMPLE
    latent_size = get_latent_size(input_size, num_downsample)
    if latent_size < 4:
        raise ValueError(f"Latent size {latent_size} is too small. Reduce NUM_DOWNSAMPLE or increase input size.")
    
    print(f"Configuration validated: Input {input_size}x{input_size} -> Latent {latent_size}x{latent_size}")
    
    return input_size, latent_size