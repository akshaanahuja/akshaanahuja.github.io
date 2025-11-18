import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import importlib.util
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no GUI windows)
import matplotlib.pyplot as plt

from dataloader import create_dataloader
from helper_funcs import sample_rays_from_images, pixel_to_ray, sample_points_along_rays
from volume_rendering import volrend
import os
from tqdm import tqdm

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


#For each iteration
    #sample rays from training images - randomly sample N pixels (10,000 in spec), convert these pixels to rays (helper_func) and then get ground truth rgb colors. 
    #sample points along the rays - for each ray, sample M points (32) between near and far. -> Shape is (N_rays, N_samples, 3)
    #Query model - for each sampled point, query model(point3d, ray_dir) getting (N_rays, N_samples, 3) and (N_rays, N_samples, 1) (rgb and density)
    #NOW VOLUME RENDERING:
    #composite colors along each ray using alpha compositing
    #convert densities to opacities to weights and then take weighted sum (sum(weight_i * rgb_i))
    #final shape is (N_rays, 3) - one color per ray

    #NOW LOSS
    #MSE loss pred_rgb, target_rgb (target_rgb comes from dataloader)
    #loss.backward() and then optimizer.step()

    #LOGGING AND VALIDATION


def compute_psnr(mse_loss):
    #we wannt to MAXIMIZE psnr -> peak signal noise ratio. 
    if mse_loss <= 0:
        return float('inf')  #no errors in reconstruction of the image, no loss
    return 10.0 * np.log10(1.0 / mse_loss)


def render_image_from_camera(model, K, c2w, H, W, device, near=2.0, far=6.0, N_samples=32):
    #this is inference , model.eval() is used to set model to evaluation mode
    model.eval()
    
   
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    
    # Add 0.5 offset for pixel center - this is to center the pixel on the center of the pixel
    uv = np.stack([x_flat, y_flat], axis=1) + 0.5  # (H*W, 2)
    
    ray_o, ray_d = pixel_to_ray(K, c2w, uv)
    

    ray_o = torch.from_numpy(ray_o).float().to(device)  # (H*W, 3)
    ray_d = torch.from_numpy(ray_d).float().to(device)  # (H*W, 3)
    
    # Sample points along rays
    points = sample_points_along_rays(
        ray_o.cpu().numpy(), 
        ray_d.cpu().numpy(), 
        N_samples = N_samples, 
        near= near, 
        far= far, 
        during_training= False
    )  #(H*W, N_samples, 3)
    
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
    
    # Volume rendering
    step_size = (far - near) / N_samples
    rendered_rgb = volrend(density_all, rgb_all, step_size)  # (H*W, 3)
    
    # Reshape to image
    rendered_image = rendered_rgb.reshape(H, W, 3).numpy()
    
    model.train()
    return rendered_image


def evaluate_validation(model, K, images_val, c2ws_val, device, near=2.0, far=6.0, N_samples=64):
    """Evaluate model on validation set and return average PSNR"""
    model.eval()
    val_psnrs = []
    
    with torch.no_grad():
        for i in range(len(images_val)):
            # Render image from validation camera
            H, W = images_val[i].shape[:2]
            rendered = render_image_from_camera(model, K, c2ws_val[i], H, W, device, near, far, N_samples)
            
            # Compute PSNR
            mse = np.mean((rendered - images_val[i]) ** 2)
            psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else float('inf')
            val_psnrs.append(psnr)
    
    model.train()
    return np.mean(val_psnrs), val_psnrs


def visualize_rays_and_samples(ray_o, ray_d, points, c2ws_train, K, H, W, save_path, max_rays=100):
    """Visualize rays, sampled points, and cameras using matplotlib (no viser)"""
    # Sample up to max_rays rays
    N_rays = min(len(ray_o), max_rays)
    indices = np.random.choice(len(ray_o), N_rays, replace=False)
    
    ray_o_subset = ray_o[indices]  # (N_rays, 3)
    ray_d_subset = ray_d[indices]  # (N_rays, 3)
    points_subset = points[indices]  # (N_rays, N_samples, 3)
    
    # Use matplotlib for 3D visualization
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot a few camera positions
    num_cameras_to_show = min(5, len(c2ws_train))
    for i in range(num_cameras_to_show):
        cam_pos = c2ws_train[i][:3, 3]
        ax.scatter(*cam_pos, c='red', s=100, marker='^', label='Camera' if i == 0 else '')
    
    # Plot rays
    for i, (o, d) in enumerate(zip(ray_o_subset[:20], ray_d_subset[:20])):  # Show first 20 rays
        far_point = o + d * 6.0
        ax.plot([o[0], far_point[0]], [o[1], far_point[1]], [o[2], far_point[2]], 
               'b-', alpha=0.3, linewidth=0.5)
    
    # Plot sampled points (flattened)
    points_flat = points_subset.reshape(-1, 3)
    ax.scatter(points_flat[:, 0], points_flat[:, 1], points_flat[:, 2], 
              c='gray', s=1, alpha=0.5, label='Sampled Points')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f'Rays and Sampled Points (showing {N_rays} rays)')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def train_3d_nerf(images_train = images_train, 
    c2ws_train = c2ws_train,
    images_val = images_val,
    c2ws_val = c2ws_val,
    c2ws_test = c2ws_test,
    focal = focal,
    num_iterations = 1000,
    num_rays_to_sample = 10000,
    learning_rate = 5e-4,
    L_x = 10,
    L_rd = 4,
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir = "nerf/part2/checkpoints/lego_200x200_L10_width256",
    log_interval = 100):

    print("=" * 60)
    print("Training 3D NeRF")
    print("=" * 60)

    print("\n[Step 1] Creating model...")
    model = NeRF3D(L_x = L_x, L_rd = L_rd).to(device) #init model structure
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    print(f"Model architecture: {model}")
    
    #create dataloader obj to load batches of data for training
    print("\n[Step 2] Creating dataloader...")

    # Get image dimensions first
    H, W = images_train.shape[1:3]

    # Construct camera intrinsic matrix K with principal point in 3rd column
    K = np.array([
    [focal, 0, W / 2],      # fx, 0, cx (principal point x)
    [0, focal, H / 2],      # 0, fy, cy (principal point y)
    [0, 0, 1]               # 0, 0, 1
    ])

    dataloader = create_dataloader(K, c2ws_train, images_train, num_rays_to_sample, batch_size=1)

    #get image dimensions
    dataset = dataloader.dataset
    print(f"Image size: {W}x{H}")  # width x height
    print(f"Total pixels per image: {H*W:,}")
    print(f"Rays per batch: {num_rays_to_sample:,}")
        
    #create loss func
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #train model for num_iterations (1k)
    model.train() #doesnt need to be inside for loop because we are just setting model to train mode

    #track losses for logging
    running_loss = 0.0
    losses = []
    psnrs = []
    val_psnrs = []  # Validation PSNRs
    val_iterations = []  # Iterations where validation was run

    checkpoint_iters = [0, 200, 400, 600, 800]
    
    # For visualization - render ONE test camera at 4 training checkpoints
    test_camera_idx = 0  # Always use the same test camera
    test_render_checkpoints = []  # Store (iteration, rendered_image) tuples
    test_render_iterations = [0, 250, 500, 750, 1000]  # 4-5 checkpoints during training

    os.makedirs(save_dir, exist_ok = True) #create save directory if it doesnt exist

    pbar = tqdm(enumerate(dataloader), total = num_iterations, desc = "Training")

    # NeRF training parameters
    near = 2.0
    far = 6.0
    N_samples = 32  # Number of points to sample along each ray

    for iteration, batch in pbar:
        if iteration >= num_iterations:
            break
        
        # Step 1: Unpack batch from dataloader
        # Dataloader returns: (ray_origins, ray_directions, ray_colors)
        ray_o, ray_d, target_rgb = batch
        
        # Remove batch dimension (batch_size=1) and move to device
        ray_o = ray_o.squeeze(0).to(device)  # (N_rays, 3)
        ray_d = ray_d.squeeze(0).to(device)  # (N_rays, 3)
        target_rgb = target_rgb.squeeze(0).to(device)  # (N_rays, 3)
        
        # Step 2: Sample points along rays
        # Convert to numpy for helper function, then back to tensor
        points = sample_points_along_rays(
            ray_o.cpu().numpy(),
            ray_d.cpu().numpy(),
            N_samples=N_samples,
            near=near,
            far=far,
            during_training=True
        )  # (N_rays, N_samples, 3)
        
        # Convert to tensor and reshape for batch processing
        points_flat = torch.from_numpy(points).float().to(device)  # (N_rays, N_samples, 3)
        points_flat = points_flat.reshape(-1, 3)  # (N_rays * N_samples, 3)
        
        # Expand ray directions to match each sampled point
        ray_d_expanded = ray_d.unsqueeze(1).expand(-1, N_samples, -1)  # (N_rays, N_samples, 3)
        ray_d_expanded = ray_d_expanded.reshape(-1, 3)  # (N_rays * N_samples, 3)
        
        #zero the gradients
        optimizer.zero_grad()
        
        # Query model: model expects (coords, ray_direction)
        rgb_all, density_all = model(points_flat, ray_d_expanded)
        # rgb_all: (N_rays * N_samples, 3)
        # density_all: (N_rays * N_samples, 1)
        
        # Reshape back to (N_rays, N_samples, ...)
        rgb_all = rgb_all.reshape(-1, N_samples, 3)  # (N_rays, N_samples, 3)
        density_all = density_all.reshape(-1, N_samples, 1)  # (N_rays, N_samples, 1)
        
        # Step 4: Volume rendering
        # Compute step_size (distance between samples)
        step_size = (far - near) / N_samples
        
        # Volume render to get final RGB per ray
        rendered_rgb = volrend(density_all, rgb_all, step_size)  # (N_rays, 3)
        
        # Step 5: Compute loss
        loss = criterion(rendered_rgb, target_rgb)
        
        # Step 6: Backpropagation
        loss.backward()
        optimizer.step()
        
        #track losses and psnrs
        loss_value = loss.item()
        running_loss += loss_value
        losses.append(loss_value)

        psnr = compute_psnr(loss_value)
        psnrs.append(psnr)

        #logging information and validation
        if (iteration + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            avg_psnr = np.mean(psnrs[-log_interval:])
            running_loss = 0.0
            
            print(f"\nIteration {iteration+1}/{num_iterations}")
            print(f"  Train Loss: {avg_loss:.6f} | Train PSNR: {avg_psnr:.2f} dB")
            
            # Validation
            if images_val is not None and c2ws_val is not None:
                print("  Running validation...")
                val_psnr_mean, val_psnr_list = evaluate_validation(
                    model, K, images_val, c2ws_val, device, near, far, N_samples=64
                )
                val_psnrs.append(val_psnr_mean)
                val_iterations.append(iteration + 1)
                print(f"  Val PSNR: {val_psnr_mean:.2f} dB (on {len(images_val)} images)")
            
            # Render test camera view at specific checkpoints (for progressive visualization)
            if c2ws_test is not None and (iteration + 1) in test_render_iterations:
                print("  Rendering test camera view...")
                test_image = render_image_from_camera( #using our inference script. 
                    model, K, c2ws_test[test_camera_idx], H, W, device, near, far, N_samples=32
                )
                test_render_checkpoints.append((iteration + 1, test_image))
                
                # Save individual test render
                test_render_path = os.path.join(save_dir, f"test_render_iter_{iteration+1}.jpg")
                plt.imsave(test_render_path, np.clip(test_image, 0, 1), format='jpg', dpi=150)
                print(f"  Test render saved: {test_render_path}")
            
            # Visualize rays and samples (only once, at first log interval)
            if iteration == 0 or (iteration + 1) == log_interval:
                print("  Creating ray/sample visualization...")
                # Use the current batch's rays and points
                ray_o_viz = ray_o.cpu().numpy()
                ray_d_viz = ray_d.cpu().numpy()
                points_viz = points  # Already numpy from sample_points_along_rays
                
                viz_path = os.path.join(save_dir, f"rays_samples_iter_{iteration+1}.png")
                visualize_rays_and_samples(
                    ray_o_viz, ray_d_viz, points_viz, c2ws_train, K, H, W, viz_path, max_rays=100
                )
            
            # Save checkpoint
            if (iteration + 1) in checkpoint_iters or (iteration + 1) == num_iterations:
                checkpoint_path = os.path.join(save_dir, f"model_iter_{iteration+1}.pt")
                torch.save({
                    'iteration': iteration+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_value,
                    'val_psnr': val_psnrs[-1] if val_psnrs else None,
                }, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")
            
            print()  # Empty line for readability
    
    # Final visualizations after training
    print("\n" + "=" * 60)
    print("Creating final visualizations...")
    print("=" * 60)
    
    # 1. Plot PSNR curve on validation set
    if len(val_psnrs) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(val_iterations, val_psnrs, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Validation PSNR (dB)', fontsize=12)
        plt.title('Validation PSNR Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        val_curve_path = os.path.join(save_dir, "validation_psnr_curve.png")
        plt.savefig(val_curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Validation PSNR curve saved: {val_curve_path}")
    
    # 2. Plot ONE test camera view across 4-5 training checkpoints (progressive reconstruction)
    if len(test_render_checkpoints) > 0:
        num_test_renders = len(test_render_checkpoints)
        cols = num_test_renders
        rows = 1
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, 4))
        if cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (iter_num, test_img) in enumerate(test_render_checkpoints):
            axes[i].imshow(np.clip(test_img, 0, 1))
            axes[i].set_title(f'Iteration {iter_num}', fontsize=12, fontweight='bold')
            axes[i].axis('off')
        
        plt.suptitle('Progressive Reconstruction: Test Camera View Across Training', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        test_progress_path = os.path.join(save_dir, "test_camera_progressive_reconstruction.jpg")
        plt.savefig(test_progress_path, dpi=150, bbox_inches='tight', format='jpg')
        plt.close()
        print(f"✓ Progressive test camera reconstruction saved: {test_progress_path}")
    
    print("\nTraining complete!")
    print(f"All outputs saved to: {save_dir}")

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
    L_x = 10,
    L_rd = 4,
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir = "nerf/part2/checkpoints/lego_200x200_L10_width256",
    log_interval = 100)