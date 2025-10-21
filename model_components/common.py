import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels, eps=1e-6)
        self.swish = Swish()
        
    def forward(self, x):
        return self.swish(self.gn(x))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            GroupNorm(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        )
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
            
    def forward(self, x):
        return self.block(x) + self.residual_conv(x)

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)