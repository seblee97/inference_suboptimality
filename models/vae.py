import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import os

class VAE(nn.Module):
    
    def __init__(self, encoder, decoder):
        """
        Constructor for VAE. Takes an encoder and a decoder.

        """
        super(VAE, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder

        self.encoder_frozen = False
        self.decoder_frozen = False

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def forward(self, x):

        encoder_output = self.encoder(x)

        latent_vector = encoder_output['z']
        decoding = torch.sigmoid(self.decoder(latent_vector))

        return {**encoder_output, **{'x_hat': decoding}}
        
