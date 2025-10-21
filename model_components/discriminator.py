import torch
import torch.nn as nn
from configs import ModelConfig, DataConfig

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        
        # Determine input channels based on dataset
        if DataConfig.DATASET_TYPE == 'mnist' or (DataConfig.DATASET_TYPE == 'custom' and DataConfig.CUSTOM_GRAYSCALE):
            in_channels = 1
        else:
            in_channels = 3
            
        base_channels = ModelConfig.DISC_BASE_CHANNELS
        num_layers = ModelConfig.DISC_LAYERS
        
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            layers.extend([
                nn.Conv2d(current_channels, out_channels, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            current_channels = out_channels
        
        layers.extend([
            nn.Conv2d(current_channels, base_channels * 4, 4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, 4, stride=1, padding=1)
        ])
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)