import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import importlib.util


spec = importlib.util.spec_from_file_location("d2d_nerf", "nerf/part1/2d_nerf.py")
d2d_nerf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d2d_nerf)
NeRF2D = d2d_nerf.NeRF2D

#inference is just passing all coords, queries model at every pixel in the image


def load_model(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only = False)
    
    #get model configuration
    L = checkpoint.get('L', 10) #default is 10 byt can set this (tune this in hyperparameter search)
    

    model = NeRF2D(L = L).to(device) #init model structure
    
    model.load_state_dict(checkpoint['model_state_dict']) #load the weights from the checkpoint, this is what using nn.Module allows us to do in the Nerf2d class
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"L (positional encoding levels): {L}")
    print(f"Final loss: {checkpoint.get('final_loss', 'N/A')}")
    print(f"Final PSNR: {checkpoint.get('final_psnr', 'N/A'):.2f} dB")
    
    return model, checkpoint
#load model gets our model from where it was saved

def reconstruct_image(model, width, height, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    
    #this is now our list of all possible pix coords to sample from (we will sample every single one, and this is whole image)
    y_coords, x_coords = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing='ij'
    )
    
    #normalize
    coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
    coords[:, 0] = coords[:, 0] / width  
    coords[:, 1] = coords[:, 1] / height  
    
  
    coords_tensor = torch.from_numpy(coords).float().to(device)
    
    batch_size = 10000
    reconstructed_colors = []
    
    with torch.no_grad():
        for i in range(0, len(coords_tensor), batch_size): #iterate over all coords in batches of 10k, avoid memory issues
            batch_coords = coords_tensor[i:i+batch_size]
            batch_colors = model(batch_coords)  #forward pass call (inference here)
            reconstructed_colors.append(batch_colors.cpu().numpy())
    
    #concatenate all batches
    reconstructed_colors = np.concatenate(reconstructed_colors, axis = 0) #essentially a reshape, putting the 10k pixel batches back togethrer
    

    reconstructed_image = reconstructed_colors.reshape(height, width, 3)
    
    #reconstructed_image = np.clip(reconstructed_image, 0, 1)
    
    return reconstructed_image


def visualize_reconstruction(original_image_path, model, save_path=None):

    original_img = Image.open(original_image_path).convert('RGB')
    original_array = np.array(original_img) / 255.0  
    height, width = original_array.shape[:2]
    
    print(f"Reconstructing image of size {width}x{height}...")
    reconstructed = reconstruct_image(model, width, height)
    
   
    mse = np.mean((original_array - reconstructed) ** 2)
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    

    axes[0].imshow(original_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(reconstructed)
    axes[1].set_title(f'Reconstructed Image\nPSNR: {psnr:.2f} dB')
    axes[1].axis('off')
    

    error = np.abs(original_array - reconstructed)
    axes[2].imshow(error, cmap='hot')
    axes[2].set_title('Absolute Error\n(MSE: {:.6f})'.format(mse))
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return mse, psnr


def visualize_training_progress(checkpoint_path, save_path=None):
  
  
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'losses' not in checkpoint or 'psnrs' not in checkpoint:
        print("Checkpoint does not contain training history")
        return
    
    losses = checkpoint['losses']
    psnrs = checkpoint['psnrs']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    

    ax1.plot(losses)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    ax1.set_yscale('log')  
    
    ax2.plot(psnrs)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Training PSNR')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_image_at_different_training_checkpoints(model, coords, save_path=None):
    import glob

    #find checkpoint files at steps of 500 (including 0 if available)
    checkpoint_iters = [0, 500, 1000, 1500, 2000]
    checkpoint_dir = "/Users/akshaanahuja/akshaanahuja.github.io/nerf/part1/checkpoints/my_image_L10_width256"
    image_path = "nerf/datasets/2dnerf_myimage.jpg"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the original image for shape
    original_img = Image.open(image_path).convert("RGB")
    original_array = np.array(original_img) / 255.0
    h, w = original_array.shape[:2]

    # Try to find the relevant checkpoint files
    reconstructions = []
    actual_iters = []
    for it in checkpoint_iters:
        # Look for file of the form "nerf2d_L10_iter{it}.pth" in the L10_width256 directory
        candidate = glob.glob(os.path.join(checkpoint_dir, f"nerf2d_L10_iter{it}.pth"))
        if not candidate:
            continue  # not found
        checkpoint_path = candidate[0]
        # Reload model for each checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        L = checkpoint.get('L', 10)
        # Dynamically get model class
        model_class = NeRF2D
        model_ = model_class(L=L).to(device)
        model_.load_state_dict(checkpoint['model_state_dict'])
        model_.eval()
        # Reconstruct image at this step
        y_coords, x_coords = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing='ij'
        )
        coords_grid = np.stack([x_coords.flatten() / w, y_coords.flatten() / h], axis=1)
        coords_tensor = torch.from_numpy(coords_grid).float().to(device)
        reconstructed_colors = []
        batch_size = 10000
        with torch.no_grad():
            for i in range(0, len(coords_tensor), batch_size):
                batch = coords_tensor[i:i+batch_size]
                rgb = model_(batch)
                reconstructed_colors.append(rgb.cpu().numpy())
        recon = np.concatenate(reconstructed_colors, axis=0).reshape(h, w, 3)
        reconstructions.append(recon)
        actual_iters.append(it)

    # Plot
    n = len(reconstructions)
    fig, axes = plt.subplots(1, n, figsize=(4.5*n, 5))
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].imshow(np.clip(reconstructions[i], 0, 1))
        axes[i].set_title(f"Iter {actual_iters[i]}")
        axes[i].axis('off')
    plt.suptitle("Reconstruction at Different Training Iterations", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Image evolution figure saved to {save_path}")
    else:
        plt.show()
    plt.close()
#THis function is just an extra function I vibecoded to see the inside of the model (interpretability)
def visualize_intermediate_layers(model, coords, save_dir=None):
    """
    Visualize intermediate activations in the model.
    This requires registering forward hooks to capture layer outputs.
    
    Args:
        model: Trained NeRF2D model
        coords: Input coordinates, shape (N, 2)
        save_dir: Directory to save visualizations
    """
  
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    hooks = []
    layer_names = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Linear):
            layer_name = f'linear_{i}'
            layer_names.append(layer_name)
            hooks.append(layer.register_forward_hook(get_activation(layer_name)))
    
    #forward pass
    model.eval()
    with torch.no_grad():
        output = model(coords)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize activations
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    num_layers = len(layer_names)
    fig, axes = plt.subplots(1, num_layers, figsize=(4*num_layers, 4))
    
    if num_layers == 1:
        axes = [axes]
    
    for i, layer_name in enumerate(layer_names):
        activation = activations[layer_name]
        
        # For visualization, take mean across batch dimension
        # and show distribution or statistics
        if len(activation.shape) == 2:
            # Show activation statistics
            ax = axes[i]
            ax.hist(activation.cpu().numpy().flatten(), bins=50, alpha=0.7)
            ax.set_title(f'{layer_name}\nShape: {activation.shape}')
            ax.set_xlabel('Activation Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'intermediate_activations.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Intermediate activations saved to {save_path}")
    else:
        plt.show()
    
    plt.close()



if __name__ == "__main__":
    # Configuration
    checkpoint_path = "/Users/akshaanahuja/akshaanahuja.github.io/nerf/part1/checkpoints/my_image_L10_width256/nerf2d_L10_iter2000.pth"
    original_image_path = "nerf/datasets/2dnerf_myimage.jpg"
    output_dir = "nerf/part1/results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    #load model
    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, checkpoint = load_model(checkpoint_path, device)
    
    # Visualize reconstruction
    print("\n" + "="*60)
    print("Reconstructing Image")
    print("="*60)
    mse, psnr = visualize_reconstruction(
        original_image_path,
        model,
        save_path=os.path.join(output_dir, "reconstruction.png")
    )
    print(f"\nReconstruction Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    
    # Visualize training progress
    print("\n" + "="*60)
    print("Visualizing Training Progress")
    print("="*60)
    visualize_training_progress(
        checkpoint_path,
        save_path=os.path.join(output_dir, "training_curves.png")
    )
    
    # Visualize intermediate layers (optional)
    print("\n" + "="*60)
    print("Visualizing Intermediate Layers")
    print("="*60)
    # Sample some coordinates for visualization
    test_coords = torch.tensor([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0]
    ], dtype=torch.float32).to(device)
    
    visualize_intermediate_layers(
        model,
        test_coords,
        save_dir=os.path.join(output_dir, "intermediate_layers")
    )
    # Visualize image at different training checkpoints
    print("\n" + "="*60)
    print("Visualizing Image at Different Training Checkpoints")
    print("="*60)
    visualize_image_at_different_training_checkpoints(
        model,
        test_coords,
        save_path=os.path.join(output_dir, "image_at_different_checkpoints.png")
    )

