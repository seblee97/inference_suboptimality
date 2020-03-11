import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    
    def __init__(self, encoder, decoder):
        """
        Constructor for VAE. Takes an encoder and a decoder.

        """
        super(VAE, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
    
    
    def forward(self, x):

        encoder_output = self.encoder(x)

        latent_vector = encoder_output['z']
        decoding = torch.sigmoid(self.decoder(latent_vector))

        return {**encoder_output, **{'x_hat': decoding}}
        
