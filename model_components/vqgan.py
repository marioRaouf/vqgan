import torch
import torch.nn as nn
import math
from .encoder import Encoder
from .decoder import Decoder
from .vector_quantizer import VectorQuantizer
from configs import ModelConfig

class VQGAN(nn.Module):
    def __init__(self):
        super(VQGAN, self).__init__()
        
        self.encoder = Encoder()
        self.vector_quantization = VectorQuantizer()
        self.decoder = Decoder()
        
        self.num_downsample = ModelConfig.NUM_DOWNSAMPLE
        
    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, encoding_indices = self.vector_quantization(z)
        x_recon = self.decoder(quantized)
        
        return loss, x_recon, perplexity, encoding_indices
    
    def encode(self, x):
        z = self.encoder(x)
        _, _, _, encoding_indices = self.vector_quantization(z)
        return encoding_indices
    
    def decode(self, encoding_indices):
        batch_size, seq_len = encoding_indices.shape
        h = w = int(math.sqrt(seq_len))
        
        quantized = self.vector_quantization.embedding(encoding_indices)
        quantized = quantized.view(batch_size, h, w, self.vector_quantization.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return self.decoder(quantized)