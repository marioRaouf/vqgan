# VQ-GAN + Transformer for Image Generation
## Project Overview

This project implements a complete VQ-GAN (Vector Quantized Generative Adversarial Network) combined with a Transformer for high-quality image generation. The system works in two stages:

**Stage 1:** Train a VQ-GAN to learn efficient image compression and reconstruction using discrete latent codes

**Stage 2:** Train a Transformer to model the distribution of these latent codes and generate new sequences

The architecture is based on the paper **"Taming Transformers for High-Resolution Image Synthesis"** but simplified and optimized for MNIST and custom datasets.

## Quick Introduction to VQ-GAN
VQ-GAN is a generative model that combines:

**Encoder:** Compresses images into a discrete latent space

**Vector Quantizer:** Maps continuous features to discrete codebook entries

**Decoder:** Reconstructs images from discrete codes

**Discriminator:v Ensures generated images are realistic

The Transformer then learns to generate sequences of these discrete codes, enabling the creation of new images by sampling from the learned distribution.

Project Structure
```
project/
├── dataloaders/
│   ├── __init__.py
│   ├── mnist.py              # MNIST dataset loader
│   └── custom.py             # Custom dataset loader
├── model_components/
│   ├── __init__.py
│   ├── encoder.py            # VQ-GAN Encoder
│   ├── decoder.py            # VQ-GAN Decoder
│   ├── vector_quantizer.py   # Vector Quantization layer
│   ├── discriminator.py      # PatchGAN Discriminator
│   ├── vqgan.py              # Complete VQ-GAN model
│   ├── transformer.py        # GPT-style Transformer
│   └── common.py             # Shared components (Residual blocks, etc.)
├── train.py                  # Training scripts for both stages
├── generate.py               # Image generation script
├── configs.py               # All configuration settings
└── utils.py                 # Utility functions
```
## Installation
**simply run:**
```
pip install einops tqdm pillow matplotlib torch torchvision
```
### Prerequisites
Python 3.7+

PyTorch 1.9+

CUDA-capable GPU (recommended)

### Installation Steps
Clone the repository

```
git clone <repository-url>
cd vqgan-transformer
Install dependencies
```
```
pip install torch torchvision matplotlib tqdm pillow

```
### Verify installation
```
python -c "import torch; print('PyTorch version:', torch.__version__)"
```
### Usage Examples

1. Training Stage 1 (VQ-GAN)
For MNIST:

### In configs.py
```

class DataConfig:
    DATASET_TYPE = 'mnist'
    MNIST_IMAGE_SIZE = 28
    MNIST_TRAIN_SAMPLES = 60000
    MNIST_VAL_SAMPLES = 10000
    BATCH_SIZE = 16

class ModelConfig:
    BASE_CHANNELS = 64
    NUM_DOWNSAMPLE = 2
    NUM_EMBEDDINGS = 256
    EMBEDDING_DIM = 32

class Stage1TrainConfig:
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 20
    SAVE_DIR = './models/stage1'

class TrainingCoordinator:
    train_vqgan = True
    train_transformer = False
    vqgan_checkpoint_path = ""
```

the Training Coordinator specifies if only train_vqgan or only train_transformer if the vqgan already trained then you will need to provide the vqgan_checkpoint_path.

### then Simply run:
```
python train.py
```
**The script will automatically use MNIST configuration. To customize, edit configs.py**



**2. Training Stage 2 (Transformer)** After training VQ-GAN, train the transformer:

### Update configs.py first:
```
class Stage2TrainConfig:
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    SAVE_DIR = './models/stage2'
```
### Then run:
```
python train.py
```

Or specify VQ-GAN checkpoint path:

```
from train import train_stage2
transformer, vqgan = train_stage2(vqgan_checkpoint_path='./models/stage1/vqgan_best.pth')
```
**3. Generating Images**

### Update configs.py with model paths:
```
class GenerationConfig:
    VQGAN_CHECKPOINT = './models/stage1/vqgan_best.pth'
    TRANSFORMER_CHECKPOINT = './models/stage2/transformer_best.pth'
    NUM_SAMPLES = 16
    TEMPERATURE = 0.9
    OUTPUT_DIR = './generated'
```
### Run generation:
```
python generate.py
```
## Adding Custom Datasets

### Prepare your custom dataset in this structure:
```
custom_dataset/
├── train/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── val/
    ├── image1.jpg
    ├── image2.png
    └── ...
```
### Configuration for Custom Dataset
**Update configs.py:**
```
class DataConfig:
    DATASET_TYPE = 'custom'  # Change from 'mnist' to 'custom'
    CUSTOM_DATA_PATH = "path/to/your/custom_dataset"
    CUSTOM_IMAGE_SIZE = 128      # Resize images to this size
    CUSTOM_GRAYSCALE = False     # Set to True for grayscale, False for RGB
    BATCH_SIZE = 16
    NUM_WORKERS = 4

class ModelConfig:
    # Adjust model parameters for your dataset
    BASE_CHANNELS = 128          # Increase for higher resolution
    NUM_DOWNSAMPLE = 3           # More downsamples for larger images
    NUM_EMBEDDINGS = 512         # Larger codebook
    EMBEDDING_DIM = 64           # Higher dimensional embeddings

class Stage1TrainConfig:
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 50              # More epochs for complex datasets
    SAVE_DIR = './models/stage1_custom'
```
**then run stage1 and stage2 training as usual**


## Generate samples:

### Update GenerationConfig in configs.py
```
class GenerationConfig:
    VQGAN_CHECKPOINT = './models/stage1_custom/vqgan_best.pth'
    TRANSFORMER_CHECKPOINT = './models/stage2_custom/transformer_best.pth'
    NUM_SAMPLES = 16
    OUTPUT_DIR = './generated_custom'
```
```
python generate.py
```

