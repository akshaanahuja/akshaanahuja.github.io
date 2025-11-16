import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

from dataloader import create_dataloader

import importlib.util
import sys


spec = importlib.util.spec_from_file_location("d2d_nerf", "nerf/part1/2d_nerf.py")
d2d_nerf = importlib.util.module_from_spec(spec)

spec.loader.exec_module(d2d_nerf)
NeRF2D = d2d_nerf.NeRF2D


def compute_psnr(mse_loss):
    #we wannt to MAXIMIZE psnr -> peak signal noise ratio. 
    if mse_loss <= 0:
        return float('inf')  #no errors in reconstruction of the image, no loss
    return 10.0 * np.log10(1.0/mse_loss)


def train_2d_nerf(
    image_path = "nerf/datasets/2dnerf_myimage.jpg", 
    num_iterations = 2000, #spec says 1-3k
    num_pixels_to_sample = 10000,  #spec says 10k
    learning_rate = 1e-2,
    L = 10,  #positional encoding frequency levels #THIS IS HYPERPARAM TO TUNE / SEE DIFFS IN VALS
    device= 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir= "/Users/akshaanahuja/akshaanahuja.github.io/nerf/part1/checkpoints/my_image_L10_width256",
    log_interval = 100 #best practice to log training progress 
):

    print("=" * 60)
    print("Training 2D NeRF")
    print("=" * 60)

    

    print("\n[Step 1] Creating model...")
    model = NeRF2D(L = L).to(device) #init model structure
    

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    print(f"Model architecture: {model}")
    
    #create dataloader obj to load batches of data for training
    print("\n[Step 2] Creating dataloader...")

    dataloader = create_dataloader(image_path = image_path, batch_size = 1, num_pixels_to_sample = num_pixels_to_sample, shuffle = True)

    
    #get image dimensions for later use
    dataset = dataloader.dataset #dataset is the ImageDataset object
    print(f"Image size: {dataset.width}x{dataset.height}") #width and height of the image
    print(f"Total pixels: {dataset.total_pixels:,}") #total number of pixels in the image
    print(f"Pixels per batch: {num_pixels_to_sample:,}") #number of pixels to sample per batch
    

    #create loss func
    criterion = nn.MSELoss()
   
    
    #create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    #train model for num_iterations (2k)
    model.train()
    
    #track losses for logging
    running_loss = 0.0
    losses = []
    psnrs = []
    
    checkpoint_iters = [0, 500, 1000, 1500]

    os.makedirs(save_dir, exist_ok = True) #create save directory if it doesn't exist

    pbar = tqdm(enumerate(dataloader), total = num_iterations, desc = "Training") #creates a progress bar around the dataloader object. enumerate = (index, item) pairs, item = (coords, target_colors) tuples.
    
    for iteration, (coords, target_colors) in pbar:
        if iteration >= num_iterations:
            break
        
        coords = coords.squeeze(0).to(device) #gets rid of batch dim so we can focus on the pixel coords
        target_colors = target_colors.squeeze(0).to(device) #same for colors
        
        #zero the gradients
        optimizer.zero_grad()
        
        pred_colors = model(coords) #pass coords through model to get predicted colors - THIS CALLS THE FORWARD PASS OF MODEL object
        
        #compute loss
        loss = criterion(pred_colors, target_colors) #criterion is modular so you can change loss easily
        
        #this computes all gradients of loss w.rt. all model parameters (updates all weights in the model)
        loss.backward()
        optimizer.step()
        
        #track 
        loss_value = loss.item()
        running_loss += loss_value
        losses.append(loss_value)
        
        #compute PSNR
        psnr = compute_psnr(loss_value)
        psnrs.append(psnr)
        
        #logging information 
        if (iteration + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            avg_psnr = np.mean(psnrs[-log_interval:])
            
            pbar.set_postfix({
                'Loss': f'{loss_value:.6f}',
                'Avg Loss': f'{avg_loss:.6f}',
                'PSNR': f'{psnr:.2f} dB',
                'Avg PSNR': f'{avg_psnr:.2f} dB'
            })
            
            running_loss = 0.0 #reset running loss for next batch
        
        #visualize reconstruction
        if iteration in checkpoint_iters:
            checkpoint_path = os.path.join(save_dir, f"nerf2d_L{L}_iter{iteration}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'L': L,
                'num_iterations': iteration,
                'final_loss': loss_value,
                'final_psnr': psnr,
                'losses': losses,
                'psnrs': psnrs
            }, checkpoint_path)
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    

    final_loss = np.mean(losses[-100:])  #average of last 100 iterations
    final_psnr = np.mean(psnrs[-100:]) 
    
    print(f"\nFinal Metrics (last 100 iterations):")
    print(f"Average Loss: {final_loss:.6f}")
    print(f"Average PSNR: {final_psnr:.2f} dB")
    print(f"Best PSNR: {max(psnrs):.2f} dB")
    

    #save the model
    os.makedirs(save_dir, exist_ok = True)
    
    checkpoint_path = os.path.join(save_dir, f"nerf2d_L{L}_iter{num_iterations}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'L': L,
        'num_iterations': num_iterations,
        'final_loss': final_loss,
        'final_psnr': final_psnr,
        'losses': losses,
        'psnrs': psnrs
    }, checkpoint_path)
    
    print(f"Model saved to: {checkpoint_path}")
    
    return model, losses, psnrs


if __name__ == "__main__":
    #call to our training loop above
    train_2d_nerf(
        image_path="nerf/datasets/2dnerf_myimage.jpg",
        num_iterations=2000,  
        num_pixels_to_sample = 10000,     
        learning_rate = 1e-2,  
        L = 10,             
        device= 'cuda' if torch.cuda.is_available() else 'cpu',
        log_interval=100 #print our loss, avg loss, psnr, avg psnr every 100 iterations
    )

