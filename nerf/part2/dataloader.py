import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from helper_funcs import sample_rays_from_images, sample_points_along_rays

#dataloader for 3d nerf
#sample rays from images
#sample points along rays
#return (ray_origins, ray_directions, pixel_colors) tuples

class NeRFDataset(Dataset):
    def __init__(self, K, c2w_matrices, images, N):
        self.K = K
        self.c2w_matrices = c2w_matrices
        self.images = images
        self.N = N  

    def __len__(self):
        # Return a large number to allow many iterations

        return 1000000

    def __getitem__(self, idx):

        ray_origins, ray_directions, ray_colors = sample_rays_from_images(
            self.K, self.c2w_matrices, self.images, self.N
        )
        
   
        ray_origins = torch.from_numpy(ray_origins).float()
        ray_directions = torch.from_numpy(ray_directions).float()
        ray_colors = torch.from_numpy(ray_colors).float()
        
        return ray_origins, ray_directions, ray_colors

        #my dataloader will take in intrinsics matrix, all the c2w matrices (inverse of extrinsic matrices), and all the images. here, I sample N rays from all the images and return the ray origins, ray directions 


def create_dataloader(K, c2w_matrices, images, N, batch_size=1, shuffle=True, num_workers=0):
   
    dataset = NeRFDataset(K, c2w_matrices, images, N)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader