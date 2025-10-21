import torch
import torch.nn as nn
from .common import Swish, GroupNorm, ResidualBlock, DownsampleBlock
from configs import ModelConfig, DataConfig, get_latent_size

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        # Determine input channels based on dataset
        if DataConfig.DATASET_TYPE == 'mnist' or (DataConfig.DATASET_TYPE == 'custom' and DataConfig.CUSTOM_GRAYSCALE):
            in_channels = 1
        else:
            in_channels = 3
            
        base_channels = ModelConfig.BASE_CHANNELS
        num_downsample = ModelConfig.NUM_DOWNSAMPLE
        
        # Calculate latent dimension
        input_size = DataConfig.MNIST_IMAGE_SIZE if DataConfig.DATASET_TYPE == 'mnist' else DataConfig.CUSTOM_IMAGE_SIZE
        latent_size = get_latent_size(input_size, num_downsample)
        latent_dim = ModelConfig.EMBEDDING_DIM
        
        self.initial_conv = nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1)
        
        self.downsample_blocks = nn.ModuleList()
        current_channels = base_channels
        
        for i in range(num_downsample):
            self.downsample_blocks.append(ResidualBlock(current_channels, current_channels))
            self.downsample_blocks.append(DownsampleBlock(current_channels, current_channels * 2))
            current_channels *= 2
        
        self.middle_blocks = nn.Sequential(
            ResidualBlock(current_channels, current_channels),
            ResidualBlock(current_channels, current_channels)
        )
        
        self.final_norm = GroupNorm(current_channels)
        self.final_conv = nn.Conv2d(current_channels, latent_dim, 3, stride=1, padding=1)
        
        self.num_downsample = num_downsample
        
    def forward(self, x):
        x = self.initial_conv(x)
        
        for block in self.downsample_blocks:
            x = block(x)
        
        x = self.middle_blocks(x)
        x = self.final_norm(x)
        x = self.final_conv(x)
        
        return x