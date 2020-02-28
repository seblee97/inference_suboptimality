import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    
    def __init__(self, encoder, decoder):
        
        super(VAE, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):

        latent_vector = self.encoder(x)
        
        decoding = self.decoder(latent_vector)
        
        return decoding
        