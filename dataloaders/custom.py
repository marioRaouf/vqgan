import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from configs import DataConfig

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path)
        
        if DataConfig.CUSTOM_GRAYSCALE:
            image = image.convert('L')  # Convert to grayscale
        else:
            image = image.convert('RGB')  # Convert to RGB
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # Return dummy label

def get_custom_loaders():
    """
    Create custom dataloaders from directory structure using DataConfig
    Expected structure:
    train_dataset/
    ├── train/
    │   ├── train_img1
    │   ├── train_img2
    │   └── etc
    ├── val/
    │   ├── val_img1
    │   ├── val_img2
    │   └── etc
    """
    transform = transforms.Compose([
        transforms.Resize((DataConfig.CUSTOM_IMAGE_SIZE, DataConfig.CUSTOM_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if DataConfig.CUSTOM_GRAYSCALE else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = CustomDataset(os.path.join(DataConfig.CUSTOM_DATA_PATH, 'train'), transform=transform)
    val_dataset = CustomDataset(os.path.join(DataConfig.CUSTOM_DATA_PATH, 'val'), transform=transform)
    
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
    
    print(f"Created custom training loader with {len(train_dataset)} samples")
    print(f"Created custom validation loader with {len(val_dataset)} samples")
    
    return train_loader, val_loader