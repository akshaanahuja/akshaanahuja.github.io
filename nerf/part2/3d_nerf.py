import torch
import torch.nn as nn
import numpy as np


#MLP takes in 3D coord and 3D vector rd (ray direction), output 3d rgb and 1d density (alpha of that point)


def positional_encoding(x, L):
    """
    Positional encoding for multi-dimensional input.
    For 3D input (N, 3), encodes each dimension separately and concatenates.
    
    Args:
        x: Input tensor of shape (N, D) where D is the dimension (e.g., 3 for 3D coords)
        L: Number of frequency levels
    
    Returns:
        Encoded tensor of shape (N, D * (1 + 2*L))
    """
    if x.dim() == 1:
        x = x.view(-1, 1)
    
    N, D = x.shape
    pe_list = []
    
    for d in range(D):
        x_d = x[:, d:d+1]  # (N, 1) - single dimension
        
        frequencies = 2 ** torch.arange(L, dtype=torch.float32, device=x.device)
        angles = 2 * torch.pi * x_d * frequencies.unsqueeze(0)  # (N, L)
        
        sin_terms = torch.sin(angles)  # (N, L)
        cos_terms = torch.cos(angles)  # (N, L)
        
        # Interleave sin and cos: [sin(2^0), cos(2^0), sin(2^1), cos(2^1), ...]
        pe_terms = torch.empty(N, 2 * L, device=x.device, dtype=x.dtype)
        pe_terms[:, 0::2] = sin_terms
        pe_terms[:, 1::2] = cos_terms
        
        # Concatenate original coordinate with encoded terms: [x, sin(2^0), cos(2^0), ...]
        pe_d = torch.cat([x_d, pe_terms], dim=1)  # (N, 1 + 2*L)
        pe_list.append(pe_d)
    
    # Concatenate all dimensions
    pe = torch.cat(pe_list, dim=1)  # (N, D * (1 + 2*L))
    return pe

class NeRF3D(nn.Module):
    def __init__(self, L_x=10, L_rd=4):
        super(NeRF3D, self).__init__()
        self.L_x = L_x  # Positional encoding levels for coordinates
        self.L_rd = L_rd  # Positional encoding levels for ray directions
        
        # PE dimension for 3D input: 3 * (1 + 2*L)
        pe_dim_x = 3 * (1 + 2 * L_x)  # For 3D coordinates
        pe_dim_rd = 3 * (1 + 2 * L_rd)  # For 3D ray direction
        
        # First 4 blocks of main MLP (before skip connection)
        self.block1 = nn.Sequential(
            nn.Linear(pe_dim_x, 256),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Next 4 blocks after skip connection (input is 256 + pe_dim_x)
        self.block5 = nn.Sequential(
            nn.Linear(256 + pe_dim_x, 256),
            nn.ReLU()
        )
        self.block6 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.block7 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.block8 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Density branch: Linear(256, 1) -> ReLU
        self.density_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.ReLU()
        )
        
        # RGB branch: Concat with pe_rd, then Linear(256+pe_dim_rd, 256) -> ReLU -> Linear(256, 128) -> Linear(128, 3) -> Sigmoid
        self.rgb_head = nn.Sequential(
            nn.Linear(256 + pe_dim_rd, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
    
    def forward(self, coords, rd):
        """
        Args:
            coords: 3D coordinates, shape (N, 3)
            rd: Ray directions, shape (N, 3)
        
        Returns:
            rgb: RGB colors, shape (N, 3)
            density: Density values, shape (N, 1)
        """
        # Positional encoding
        pe_x = positional_encoding(coords, self.L_x)  # (N, 3*(1+2*L_x))
        pe_rd = positional_encoding(rd, self.L_rd)  # (N, 3*(1+2*L_rd))
        
        # Main MLP branch: first 4 blocks
        x = self.block1(pe_x)  # (N, 256)
        x = self.block2(x)  # (N, 256)
        x = self.block3(x)  # (N, 256)
        x = self.block4(x)  # (N, 256)
        
        # Skip connection: concat original pe_x with output
        x = torch.cat([x, pe_x], dim=1)  # (N, 256 + pe_dim_x)
        
        # Main MLP branch: next 4 blocks
        x = self.block5(x)  # (N, 256)
        x = self.block6(x)  # (N, 256)
        x = self.block7(x)  # (N, 256)
        x = self.block8(x)  # (N, 256)
        
        # Density branch
        density = self.density_head(x)  # (N, 1)
        
        # RGB branch: concat with pe_rd
        rgb_input = torch.cat([x, pe_rd], dim=1)  # (N, 256 + pe_dim_rd)
        rgb = self.rgb_head(rgb_input)  # (N, 3)
        
        return rgb, density