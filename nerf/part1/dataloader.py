import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os

#dataloader for 2d nerf
#randomly sample N pixels at every iteration for training. 
#make N x 2 3d coords and N x 3 rgb vals for those pixels which will be train values and target rgb values (so we have actual y and can take (y - y_pred) loss (WE WILL USE MSE LOSS))

class ImageDataset(Dataset):
    def __init__(self, image_path, num_pixels_to_sample = None):
        #image path is the path to the image file
        #num_pixels_to_sample  is the number of pixels to sample per iteration (None is all pixels)
        img = Image.open(image_path).convert('RGB')
        self.image = np.array(img) 
        
        self.height, self.width = self.image.shape[:2]
        self.num_pixels_to_sample = num_pixels_to_sample if num_pixels_to_sample is not None else self.height * self.width
        
        y_coords, x_coords = np.meshgrid(
            np.arange(self.height, dtype=np.float32),
            np.arange(self.width, dtype=np.float32),
            indexing='ij'
        ) #create a grid of all possible pixel coords to sample from during trainign
        
        self.coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1) #flatten the grid to get a list of all possible pixel coords - looks like [[0, 0], [0, 1], [0, 2], ...]
        
        self.colors = self.image.reshape(-1, 3) #get list of all possible pixel colors
        self.total_pixels = self.height * self.width #total number of pixels in the image
    
    def __len__(self):
        return self.total_pixels
    
    def __getitem__(self, idx):
        #idx is the index of the pixel to sample
      
        if self.num_pixels_to_sample < self.total_pixels:
            #sample w/o replacement
            indices = np.random.choice(self.total_pixels, size = self.num_pixels_to_sample, replace = False)
        else:
            #use all pixels
            indices = np.arange(self.total_pixels)
        
        #get sampled coordinates and colors 
        sampled_coords = self.coords[indices]  #(N = num_pixels_to_sample, 2)
        sampled_colors = self.colors[indices]   #N = num_pixels_to_sample, 3
        
        #normalize
        normalized_coords = sampled_coords.copy()
        normalized_coords[:, 0] = normalized_coords[:, 0] / self.width 
        normalized_coords[:, 1] = normalized_coords[:, 1] / self.height

        normalized_colors = sampled_colors.astype(np.float32) / 255.0

        
        #convert to torch tensors
        coords_tensor = torch.from_numpy(normalized_coords).float()
        colors_tensor = torch.from_numpy(normalized_colors).float()
        
        return coords_tensor, colors_tensor


def create_dataloader(image_path, batch_size = 1, num_pixels_to_sample = 4096, shuffle = True, num_workers = 0):
    #image path is the path to the image file
    #batch size is the number of pixels to sample per iteration
    #num_pixels_to_sample is the number of pixels to sample per iteration
    #shuffle is whether to shuffle the data
    #num_workers is the number of worker processes for data loading
    dataset = ImageDataset(image_path, num_pixels_to_sample = num_pixels_to_sample)
    
    dataloader = DataLoader( #pytorch object that loads in data in batches for training. yields (coords, colors) tuples
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        pin_memory = True if torch.cuda.is_available() else False #pin memory if we using gpu 
    )
    return dataloader 

        # for i, (coords, colors) in enumerate(dataloader):
        #     print(f"\nBatch {i+1}:")
        #     print(f"  Coordinates shape: {coords.shape}")
        #     print(f"  Colors shape: {colors.shape}")
            
        #     # If batch_size=1, squeeze the batch dimension for display
        #     if coords.shape[0] == 1:
        #         coords_display = coords.squeeze(0)
        #         colors_display = colors.squeeze(0)
        #         print(f"  (After squeeze: coords {coords_display.shape}, colors {colors_display.shape})")
        #     else:
        #         coords_display = coords[0]
        #         colors_display = colors[0]
            
        #     print(f"  Coords range: [{coords.min():.4f}, {coords.max():.4f}] (should be [0, 1])")
        #     print(f"  Colors range: [{colors.min():.4f}, {colors.max():.4f}] (should be [0, 1])")
        #     print(f"  Sample coords (first 5): {coords_display[:5, :]}")
        #     print(f"  Sample colors (first 5): {colors_display[:5, :]}")
            
        #     if i >= 2:  # Test a few batches
        #         break
        
        # print("\nDataloader test complete!")
