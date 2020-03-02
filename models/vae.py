import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    
    def __init__(self, encoder, decoder, parameter):
        """
        Constructor for VAE. Takes an encoder and a decoder as well
        as a set of parameter containing:
        - size of the latent (z)
        - boolean variable for configuration (has_flow, cuda)
        - An activation function: act_func
        - n_flows info
            
            
        """
        super(VAE, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
    
    
    def forward(self, x):
        """
        Performs the encoding-decoding step.
        """
        # Sampling for the latent
        z, logpz, logqz = self.encoder.forward(x) # samples
        
        decoding = self.decoder.forward(latent_vector)
        
        
        return decoding

    def Compute_ELBO(self, x, k=1):
        """
        Performs the ELBO computation as well as other elements.
        It receives the latent as sampled by the forward in VAE (encoded-decoded)
        """







