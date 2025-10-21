import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import random
from configs import DataConfig

def get_mnist_loaders():
    """
    Create MNIST dataloaders using DataConfig
    """
    transform = transforms.Compose([
        transforms.Resize((DataConfig.MNIST_IMAGE_SIZE, DataConfig.MNIST_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load full datasets
    full_train_dataset = torchvision.datasets.MNIST(
        root=DataConfig.MNIST_PATH, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    full_val_dataset = torchvision.datasets.MNIST(
        root=DataConfig.MNIST_PATH, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create subsets if specified
    if DataConfig.MNIST_TRAIN_SAMPLES < len(full_train_dataset):
        train_indices = random.sample(range(len(full_train_dataset)), DataConfig.MNIST_TRAIN_SAMPLES)
        train_dataset = Subset(full_train_dataset, train_indices)
    else:
        train_dataset = full_train_dataset
    
    if DataConfig.MNIST_VAL_SAMPLES < len(full_val_dataset):
        val_indices = random.sample(range(len(full_val_dataset)), DataConfig.MNIST_VAL_SAMPLES)
        val_dataset = Subset(full_val_dataset, val_indices)
    else:
        val_dataset = full_val_dataset
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=DataConfig.BATCH_SIZE, 
        shuffle=True, 
        num_workers=DataConfig.NUM_WORKERS,
        pin_memory=DataConfig.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=DataConfig.BATCH_SIZE, 
        shuffle=False, 
        num_workers=DataConfig.NUM_WORKERS,
        pin_memory=DataConfig.PIN_MEMORY
    )
    
    print(f"Created MNIST training loader with {len(train_dataset)} samples")
    print(f"Created MNIST validation loader with {len(val_dataset)} samples")
    
    return train_loader, val_loader