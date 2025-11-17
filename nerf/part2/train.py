import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import importlib.util

from dataloader import create_dataloader
from helper_funcs import sample_rays_from_images, pixel_to_ray, sample_points_along_rays

data = np.load(f"nerf/part2/datasets/lego_200x200.npz")

# Training images: [100, 200, 200, 3]
images_train = data["images_train"] / 255.0

# Cameras for the training images 
# (camera-to-world transformation matrix): [100, 4, 4]
c2ws_train = data["c2ws_train"]

# Validation images: 
images_val = data["images_val"] / 255.0

# Cameras for the validation images: [10, 4, 4]
# (camera-to-world transformation matrix): [10, 200, 200, 3]
c2ws_val = data["c2ws_val"]

# Test cameras for novel-view video rendering: 
# (camera-to-world transformation matrix): [60, 4, 4]
c2ws_test = data["c2ws_test"]

# Camera focal length
focal = data["focal"]  # float



spec = importlib.util.spec_from_file_location("d3d_nerf", "nerf/part2/3d_nerf.py")
d3d_nerf = importlib.util.module_from_spec(spec)

spec.loader.exec_module(d3d_nerf)
NeRF3D = d3d_nerf.NeRF3D


def compute_psnr(mse_loss):
    #we wannt to MAXIMIZE psnr -> peak signal noise ratio. 
    if mse_loss <= 0:
        return float('inf')  #no errors in reconstruction of the image, no loss
    return 10.0 * np.log10(1.0/mse_loss)


def evaluate_validation(model, K, images_val, c2ws_val, device, num_val_rays=5000):
    """Evaluate model on validation set"""
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        # Sample rays from validation images
        ray_origins, ray_directions, ray_colors = sample_rays_from_images(
            K, c2ws_val, images_val, num_val_rays
        )
        
        # Convert to tensors
        ray_o = torch.from_numpy(ray_origins).float().to(device)  # (N, 3)
        ray_d = torch.from_numpy(ray_directions).float().to(device)  # (N, 3)
        target_rgb = torch.from_numpy(ray_colors).float().to(device)  # (N, 3)
        
        # Forward pass
        pred_rgb, pred_density = model(ray_o, ray_d)
        
        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(pred_rgb, target_rgb)
        val_losses.append(loss.item())
    
    model.train()
    avg_val_loss = np.mean(val_losses)
    val_psnr = compute_psnr(avg_val_loss)
    return avg_val_loss, val_psnr


def render_test_camera(model, K, c2w_test, H, W, device, near=2.0, far=6.0, N_samples=64):
    """Render an image from a test camera pose"""
    model.eval()
    
    # Create pixel coordinate grid
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    
    # Add 0.5 offset for pixel center
    uv = np.stack([x_flat, y_flat], axis=1) + 0.5  # (H*W, 2)
    
    # Generate rays from test camera
    ray_o, ray_d = pixel_to_ray(K, c2w_test, uv)
    
    # Convert to tensors
    ray_o = torch.from_numpy(ray_o).float().to(device)  # (H*W, 3)
    ray_d = torch.from_numpy(ray_d).float().to(device)  # (H*W, 3)
    
    # Sample points along rays
    points = sample_points_along_rays(
        ray_o.cpu().numpy(), 
        ray_d.cpu().numpy(), 
        N_samples=N_samples, 
        near=near, 
        far=far, 
        during_training=False
    )  # (H*W, N_samples, 3)
    
    # Reshape for batch processing
    N_rays = H * W
    points_flat = points.reshape(-1, 3)  # (H*W * N_samples, 3)
    ray_d_expanded = ray_d.unsqueeze(1).expand(-1, N_samples, -1).reshape(-1, 3)  # (H*W * N_samples, 3)
    
    # Query model at all points
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 10000
        rgb_batch = []
        density_batch = []
        
        for i in range(0, len(points_flat), batch_size):
            end_idx = min(i + batch_size, len(points_flat))
            points_batch = torch.from_numpy(points_flat[i:end_idx]).float().to(device)
            ray_d_batch = ray_d_expanded[i:end_idx]
            
            rgb, density = model(points_batch, ray_d_batch)
            rgb_batch.append(rgb.cpu())
            density_batch.append(density.cpu())
        
        rgb_all = torch.cat(rgb_batch, dim=0)  # (H*W * N_samples, 3)
        density_all = torch.cat(density_batch, dim=0)  # (H*W * N_samples, 1)
    
    # Reshape back
    rgb_all = rgb_all.reshape(N_rays, N_samples, 3)  # (H*W, N_samples, 3)
    density_all = density_all.reshape(N_rays, N_samples, 1)  # (H*W, N_samples, 1)
    
    # Simple volume rendering (alpha compositing)
    # Compute distances between samples
    t_vals = np.linspace(near, far, N_samples)
    t_diffs = np.diff(t_vals)
    t_diffs = np.concatenate([t_diffs, [t_diffs[-1]]])  # (N_samples,)
    t_diffs = torch.from_numpy(t_diffs).float()  # Convert to tensor
    
    # Convert density to alpha (opacity)
    alpha = 1 - torch.exp(-density_all.squeeze(-1) * t_diffs)  # (H*W, N_samples)
    
    # Compute transmittance and weights
    transmittance = torch.cumprod(1 - alpha + 1e-10, dim=1)  # (H*W, N_samples)
    weights = alpha * transmittance  # (H*W, N_samples)
    
    # Weighted sum of colors
    rendered_rgb = torch.sum(weights.unsqueeze(-1) * rgb_all, dim=1)  # (H*W, 3)
    
    # Reshape to image
    rendered_image = rendered_rgb.reshape(H, W, 3).numpy()
    
    model.train()
    return rendered_image



def train_3d_nerf(
    images_train = images_train,
    c2ws_train = c2ws_train,
    images_val = images_val,
    c2ws_val = c2ws_val,
    c2ws_test = c2ws_test,
    focal = focal,
    num_iterations = 1000,
    num_rays_to_sample = 10000,
    learning_rate = 5e-4,
    L = 10,
    device= 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir= "/Users/akshaanahuja/akshaanahuja.github.io/nerf/part2/checkpoints/lego_200x200_L10_width256",
    log_interval = 100 #best practice to log training progress 
):

    print("=" * 60)
    print("Training 3D NeRF")
    print("=" * 60)

    print("\n[Step 1] Creating model...")
    model = NeRF3D(L_x = 10, L_rd = 4).to(device) #init model structure
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    print(f"Model architecture: {model}")

    #create dataloader obj to load batches of data for training
    print("\n[Step 2] Creating dataloader...")
    
    # Get image dimensions
    H, W = images_train.shape[1:3]
    
    # Construct camera intrinsic matrix K from focal length
    K = np.array([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ])
    
    dataloader = create_dataloader(K, c2ws_train, images_train, num_rays_to_sample, batch_size=1) 

    #get image dimensions for later use
    print(f"Image size: {W}x{H}") #width and height of the image
    print(f"Total pixels per image: {H*W:,}") #total number of pixels in the image
    print(f"Rays per batch: {num_rays_to_sample:,}") #number of rays to sample per batch

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
    val_losses = []
    val_psnrs = []

    checkpoint_iters = []

    from tqdm import tqdm
    import os

    os.makedirs(save_dir, exist_ok=True)
    pbar = tqdm(range(num_iterations), desc="Training")

    for iteration in pbar:
        # Get a batch of rays and ground truth RGB
        # Assuming dataloader returns (ray_origins, ray_directions, target_rgb) per batch, shape (batch_size=1, N, dims)
        ray_batch = next(iter(dataloader))  # dataloader has essentially infinite batches

        ray_o, ray_d, target_rgb = ray_batch  # each is (1, N, 3)
        ray_o = ray_o.squeeze(0).to(device)    # (N, 3)
        ray_d = ray_d.squeeze(0).to(device)    # (N, 3)
        target_rgb = target_rgb.squeeze(0).to(device)  # (N, 3)

        optimizer.zero_grad()

        pred_rgb, pred_density = model(ray_o, ray_d)  # pred_rgb: (N, 3), pred_density: (N, 1)

        # --- Volume Rendering ---
        # Compose output colors using alpha compositing along sampled points for each ray.
        # For now, use one sample per ray (the format from your dataloader), so rendering reduces to predicted rgb per point.

        loss = criterion(pred_rgb, target_rgb)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        running_loss += loss_value
        losses.append(loss_value)

        # Compute PSNR
        mse = loss_value
        psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        psnrs.append(psnr)

        # Logging and validation
        if (iteration + 1) % log_interval == 0 or iteration == 0:
            avg_loss = running_loss / (log_interval if iteration != 0 else 1)
            avg_psnr = np.mean(psnrs[-log_interval:]) if len(psnrs) >= log_interval else psnrs[-1]
            
            # Run validation
            val_loss, val_psnr = evaluate_validation(model, K, images_val, c2ws_val, device)
            val_losses.append(val_loss)
            val_psnrs.append(val_psnr)
            
            print(f"Iteration {iteration+1}/{num_iterations} | Train Loss: {avg_loss:.6f} | Train PSNR: {avg_psnr:.2f} | Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.2f}")
            running_loss = 0.0
            
            # Render a test camera view
            test_camera_idx = (iteration // log_interval) % len(c2ws_test)  # Cycle through test cameras
            test_image = render_test_camera(model, K, c2ws_test[test_camera_idx], H, W, device)
            
            # Save test render
            import matplotlib.pyplot as plt
            test_render_path = os.path.join(save_dir, f"test_render_iter_{iteration+1}_cam_{test_camera_idx}.png")
            plt.imsave(test_render_path, np.clip(test_image, 0, 1))
            print(f"  Test render saved: {test_render_path}")

        # Save checkpoint
        if (iteration + 1) in checkpoint_iters or (iteration + 1) == num_iterations:
            checkpoint_path = os.path.join(save_dir, f"model_iter_{iteration+1}.pt")
            torch.save({
                'iteration': iteration+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_value,
            }, checkpoint_path)
            print(f"Checkpoint saved at Iter {iteration+1} to {checkpoint_path}")


if __name__ == "__main__":
    train_3d_nerf(
        images_train = images_train,
        c2ws_train = c2ws_train,
        images_val = images_val,
        c2ws_val = c2ws_val,
        c2ws_test = c2ws_test,
        focal = focal,
        num_iterations = 1000,
        num_rays_to_sample = 10000,
        learning_rate = 5e-4,
        L = 10,
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_dir = "/Users/akshaanahuja/akshaanahuja.github.io/nerf/part2/checkpoints/lego_200x200_L10_width256",
        log_interval = 100
    )