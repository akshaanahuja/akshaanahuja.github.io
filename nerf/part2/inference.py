import numpy as np
import torch
import torch.nn as nn
import importlib.util
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no GUI windows)
import matplotlib.pyplot as plt
from helper_funcs import pixel_to_ray, sample_points_along_rays
from volume_rendering import volrend
import os
from imageio import mimsave

# Load the NeRF3D model class
spec = importlib.util.spec_from_file_location("d3d_nerf", "nerf/part2/3d_nerf.py")
d3d_nerf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d3d_nerf)
NeRF3D = d3d_nerf.NeRF3D

# Load data
data = np.load(f"nerf/part2/datasets/lego_200x200.npz")

# Test cameras for novel-view video rendering: 
# (camera-to-world transformation matrix): [60, 4, 4]
c2ws_test = data["c2ws_test"]

# Camera focal length
focal = data["focal"]  # float

# Image dimensions
H, W = 200, 200

def render_image_from_camera(model, K, c2w, H, W, device, near=2.0, far=6.0, N_samples=64):
    """Render an image from a camera pose."""
    model.eval()
    
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    
    # Add 0.5 offset for pixel center
    uv = np.stack([x_flat, y_flat], axis=1) + 0.5  # (H*W, 2)
    
    ray_o, ray_d = pixel_to_ray(K, c2w, uv)
    
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
    points_flat = points.reshape(-1, 3)  # (H*W*N_samples, 3)
    ray_d_expanded = ray_d.unsqueeze(1).expand(-1, N_samples, -1).reshape(-1, 3)  # (H*W*N_samples, 3)
    
    points_tensor = torch.from_numpy(points_flat).float().to(device)
    ray_d_tensor = ray_d_expanded.to(device)
    
    # Query model in batches to avoid memory issues
    batch_size = 8192
    rgb_list = []
    density_list = []
    
    with torch.no_grad():
        for i in range(0, points_tensor.shape[0], batch_size):
            end_idx = min(i + batch_size, points_tensor.shape[0])
            points_batch = points_tensor[i:end_idx]
            ray_d_batch = ray_d_tensor[i:end_idx]
            
            rgb_batch, density_batch = model(points_batch, ray_d_batch)
            rgb_list.append(rgb_batch.cpu())
            density_list.append(density_batch.cpu())
    
    rgb = torch.cat(rgb_list, dim=0).reshape(H*W, N_samples, 3)  # (H*W, N_samples, 3)
    density = torch.cat(density_list, dim=0).reshape(H*W, N_samples, 1)  # (H*W, N_samples, 1)
    
    # Volume rendering
    step_size = (far - near) / N_samples
    rgb_rendered = volrend(density, rgb, step_size)  # (H*W, 3)
    
    # Reshape to image
    image = rgb_rendered.reshape(H, W, 3).numpy()
    
    return image

def create_spherical_rendering_gif(checkpoint_path, output_path="spherical_rendering.gif", 
                                    num_iterations=10000, L_x=10, L_rd=4):
    """Load a trained model and render all test cameras to create a spherical rendering GIF."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with same architecture
    model = NeRF3D(L_x=checkpoint.get('L_x', L_x), L_rd=checkpoint.get('L_rd', L_rd)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded (trained for {checkpoint.get('iteration', 'unknown')} iterations)")
    
    # Construct camera intrinsic matrix K
    K = np.array([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ])
    
    # Render all test cameras
    print(f"Rendering {len(c2ws_test)} test camera views...")
    images = []
    for i, c2w in enumerate(c2ws_test):
        print(f"  Rendering camera {i+1}/{len(c2ws_test)}...", end='\r')
        image = render_image_from_camera(model, K, c2w, H, W, device, near=2.0, far=6.0, N_samples=64)
        images.append((np.clip(image, 0, 1) * 255).astype(np.uint8))
    print()
    
    # Save as GIF
    print(f"Saving GIF to {output_path}...")
    mimsave(output_path, images, fps=30, loop=0)
    print(f"✓ Spherical rendering GIF saved: {output_path}")

if __name__ == "__main__":
    # Path to the saved model checkpoint
    checkpoint_path = "nerf/part2/checkpoints/lego_200x200_L10_width256/model_final.pth"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
    else:
        create_spherical_rendering_gif(checkpoint_path, 
                                      output_path="nerf/part2/checkpoints/lego_200x200_L10_width256/spherical_rendering.gif")
