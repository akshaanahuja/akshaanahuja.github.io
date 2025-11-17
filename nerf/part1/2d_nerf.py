import torch
import torch.nn as nn
import numpy as np


#MLP that takes in the 2-dim pixel coords and output the 3dim color of that pixel
#sequential (x, pe, linear (256), relu, linear (256), relu, linear (256), relu, linear (3), sigmoid, rgb)

#pe is positional encoding -> need to use frequency of image

#hyperparams: 1. Layer width (channel size)
#2. max freq L in pe calc


def positional_encoding(x, L):
    # x = x.view(-1, 1)
    # pe = torch.cat([x, torch.sin(2 * np.pi * x * (2 ** torch.arange(L).float()))], dim=1)

    x = x.view(-1, 1)  # Ensure shape (N, 1)

    frequencies = 2 ** torch.arange(L, dtype=torch.float32, device=x.device)
    

    angles = 2 * torch.pi * x * frequencies.unsqueeze(0)
    
    #sin and cos for each frequency
    sin_terms = torch.sin(angles)  
    cos_terms = torch.cos(angles) 
    
    #use torch.empty and fill in alternating pattern
    pe_terms = torch.empty(x.shape[0], 2 * L, device=x.device, dtype=x.dtype)
    pe_terms[:, 0::2] = sin_terms  
    pe_terms[:, 1::2] = cos_terms  
    
 
    pe = torch.cat([x, pe_terms], dim=1)  
    
    return pe


class NeRF2D(nn.Module):
    
    def __init__(self, L = 10): #L is hyperparam to tune as well
        #coordinates will map from 1 pixel to 1 + 2*L encoded terms (L = 10 -> x maps to 21 encoded terms, y maps to 21 encoded terms, concat to get 42 encoded terms per pixel)
        super(NeRF2D, self).__init__()

        self.L = L
        
        #compute input dims to the nn.sequential after positional encoding
        #it is 2 * L because we look at both sin and cos pairs.  
        pe_dim = 2 * (1 + 2 * L)  
        
        self.layers = nn.Sequential(
            nn.Linear(pe_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, 256), 
            nn.ReLU(), 
            nn.Linear(256, 256), 
            nn.ReLU(), 
            nn.Linear(256, 3), 
            nn.Sigmoid()
        )
    
    def forward(self, coords):
     
        x = coords[:, 0:1]  
        y = coords[:, 1:2]  #N, 1 
        
  
        pe_x = positional_encoding(x, self.L)  
        pe_y = positional_encoding(y, self.L)  
        
        #reconcatenate  encoded coordinates
        pe_combined = torch.cat([pe_x, pe_y], dim=1)  
        
        #pass through MLP
        rgb = self.layers(pe_combined) #self.layers is callable (passes through our pe coords through the nn.sequential)
        
        return rgb



# if __name__ == "__main__":
#     # Create model
#     model = NeRF2D(L=10)
    

    # test_coords = torch.tensor([
    #     [0.0, 0.0],
    #     [0.5, 0.5],
    #     [1.0, 1.0],
    #     [100.0, 200.0]
    # ], dtype=torch.float32)
    
    # print("Testing NeRF2D model:")

    # with torch.no_grad():
    #     output = model(test_coords)
    
    # print(f"Output RGB shape: {output.shape}")
    # print(f"Output RGB values (should be in [0, 1]):")
    # print(output)
    

    # print(f"\nOutput range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    # print(f"Expected range: [0.0, 1.0]")
