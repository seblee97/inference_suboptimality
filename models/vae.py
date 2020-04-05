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

    def load_weights(self, weights_path: str) -> None:
        """Load saved weights into model"""
        self.load_state_dict(torch.load(weights_path))

    def checkpoint_model_weights(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "saved_vae_weights.pt"))
    
    def forward(self, x):

        encoder_output = self.encoder(x)

        latent_vector = encoder_output['z']
        decoding = self.decoder(latent_vector)

        return {**encoder_output, **{'x_hat': decoding}}
        
