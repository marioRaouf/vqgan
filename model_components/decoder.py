import torch
import torch.nn as nn
from .common import Swish, GroupNorm, ResidualBlock, UpsampleBlock
from configs import ModelConfig, DataConfig

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Determine output channels based on dataset
        if DataConfig.DATASET_TYPE == 'mnist' or (DataConfig.DATASET_TYPE == 'custom' and DataConfig.CUSTOM_GRAYSCALE):
            out_channels = 1
        else:
            out_channels = 3
            
        num_upsample = ModelConfig.NUM_DOWNSAMPLE
        latent_dim = ModelConfig.EMBEDDING_DIM
        
        # Calculate base channels (same as encoder output)
        base_channels = ModelConfig.BASE_CHANNELS * (2 ** num_upsample)
        
        self.initial_conv = nn.Conv2d(latent_dim, base_channels, 3, stride=1, padding=1)
        
        self.middle_blocks = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels)
        )
        
        self.upsample_blocks = nn.ModuleList()
        current_channels = base_channels
        
        for i in range(num_upsample):
            self.upsample_blocks.append(UpsampleBlock(current_channels, current_channels // 2))
            current_channels = current_channels // 2
            self.upsample_blocks.append(ResidualBlock(current_channels, current_channels))
        
        self.final_norm = GroupNorm(current_channels)
        self.final_conv = nn.Conv2d(current_channels, out_channels, 3, stride=1, padding=1)
        
        self.num_upsample = num_upsample
        
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.middle_blocks(x)
        
        for block in self.upsample_blocks:
            x = block(x)
        
        x = self.final_norm(x)
        x = self.final_conv(x)
        
        return torch.tanh(x)