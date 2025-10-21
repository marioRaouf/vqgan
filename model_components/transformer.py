import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from configs import ModelConfig, DataConfig, get_sequence_length

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerGPT(nn.Module):
    def __init__(self):
        super(TransformerGPT, self).__init__()
        
        vocab_size = ModelConfig.NUM_EMBEDDINGS
        d_model = ModelConfig.TRANSFORMER_D_MODEL
        nhead = ModelConfig.TRANSFORMER_NHEAD
        num_layers = ModelConfig.TRANSFORMER_LAYERS
        
        # Calculate max sequence length
        input_size = DataConfig.MNIST_IMAGE_SIZE if DataConfig.DATASET_TYPE == 'mnist' else DataConfig.CUSTOM_IMAGE_SIZE
        max_seq_length = get_sequence_length(input_size, ModelConfig.NUM_DOWNSAMPLE)
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=ModelConfig.TRANSFORMER_DROPOUT, 
            activation='gelu',
            batch_first=False
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.max_seq_length = max_seq_length
        
        self.register_buffer("causal_mask", 
                           torch.triu(torch.ones(max_seq_length, max_seq_length) * float('-inf'), diagonal=1))
        
    def forward(self, x):
        seq_len, batch_size = x.shape
        
        x_embed = self.embedding(x) * math.sqrt(self.d_model)
        x_embed = self.pos_encoder(x_embed)
        
        output = self.transformer(
            tgt=x_embed, 
            memory=x_embed,
            tgt_mask=self.causal_mask[:seq_len, :seq_len],
            memory_mask=self.causal_mask[:seq_len, :seq_len]
        )
        
        logits = self.output_layer(output)
        return logits
    
    def generate(self, start_tokens, max_length, temperature=1.0):
        self.eval()
        with torch.no_grad():
            generated = start_tokens
            batch_size = generated.size(0)
            
            for _ in range(max_length - generated.size(1)):
                logits = self.forward(generated.transpose(0, 1))
                next_token_logits = logits[-1, :, :] / temperature
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if generated.size(1) >= self.max_seq_length:
                    break
            
            return generated