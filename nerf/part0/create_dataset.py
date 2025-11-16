import cv2
import numpy as np
import os
import glob

# Hardcode parameters
calibration_file = "nerf/camera_calibration.npz"
poses_file = "nerf/camera_poses.npz"
object_images_dir = "/Users/akshaanahuja/akshaanahuja.github.io/nerf/mouse_images"
output_file = "nerf/dataset.npz"

# Train/val/test split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Load camera calibration
print("Loading camera calibration...")
calibration_data = np.load(calibration_file)
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Load poses
print("Loading camera poses...")
poses_data = np.load(poses_file, allow_pickle=True)
c2w_matrices_3x4 = poses_data['c2w_matrices']  # Shape: (N, 3, 4)
valid_images = poses_data['valid_images']  # Array of image paths

# Convert valid_images to list of strings if it's a numpy array
if isinstance(valid_images, np.ndarray):
    valid_images = valid_images.tolist()

# Convert c2w matrices from 3x4 to 4x4 format
print("Converting c2w matrices to 4x4 format...")
c2w_matrices_4x4 = []
for c2w_3x4 in c2w_matrices_3x4:
    # Add bottom row [0, 0, 0, 1] to make it 4x4
    c2w_4x4 = np.vstack([c2w_3x4, np.array([0, 0, 0, 1])])
    c2w_matrices_4x4.append(c2w_4x4)
c2w_matrices_4x4 = np.array(c2w_matrices_4x4)

# Get image dimensions from first image
first_image = cv2.imread(valid_images[0])
h, w = first_image.shape[:2]

# Compute optimal new camera matrix to handle black boundaries
print("Computing optimal camera matrix...")
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), alpha=0, newImgSize=(w, h)
)
x, y, w_roi, h_roi = roi

# Update principal point to account for crop offset
new_camera_matrix[0, 2] -= x  # cx
new_camera_matrix[1, 2] -= y  # cy

# Extract focal length (assuming fx = fy)
focal = new_camera_matrix[0, 0]

print(f"Original image size: {w}x{h}")
print(f"ROI after undistortion: x={x}, y={y}, w={w_roi}, h={h_roi}")
print(f"New image size: {w_roi}x{h_roi}")
print(f"Focal length: {focal:.2f}")

# Undistort all images
print("\nUndistorting images...")
undistorted_images = []
valid_c2w_matrices = []

for i, image_path in enumerate(valid_images):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read {image_path}, skipping...")
        continue
    
    # Undistort image
    undistorted_img = cv2.undistort(
        img, camera_matrix, dist_coeffs, None, new_camera_matrix
    )
    
    # Crop to ROI
    undistorted_img = undistorted_img[y:y+h_roi, x:x+w_roi]
    
    # Convert BGR to RGB
    undistorted_img_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
    
    # Store undistorted image and corresponding pose
    undistorted_images.append(undistorted_img_rgb)
    valid_c2w_matrices.append(c2w_matrices_4x4[i])
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(valid_images)} images")

# Convert to numpy arrays
undistorted_images = np.array(undistorted_images, dtype=np.uint8)  # (N, H, W, 3)
valid_c2w_matrices = np.array(valid_c2w_matrices, dtype=np.float32)  # (N, 4, 4)

print(f"\nTotal images processed: {len(undistorted_images)}")
print(f"Image shape: {undistorted_images.shape}")

# Split into train/val/test sets
n_total = len(undistorted_images)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)
n_test = n_total - n_train - n_val

print(f"\nSplitting dataset:")
print(f"  Train: {n_train} images")
print(f"  Val: {n_val} images")
print(f"  Test: {n_test} images")

# Shuffle indices for random split
indices = np.arange(n_total)
np.random.seed(42)  # For reproducibility
np.random.shuffle(indices)

train_indices = indices[:n_train]
val_indices = indices[n_train:n_train+n_val]
test_indices = indices[n_train+n_val:]

# Create splits
images_train = undistorted_images[train_indices]
c2ws_train = valid_c2w_matrices[train_indices]

images_val = undistorted_images[val_indices]
c2ws_val = valid_c2w_matrices[val_indices]

c2ws_test = valid_c2w_matrices[test_indices]

# Save dataset
print(f"\nSaving dataset to {output_file}...")
np.savez(
    output_file,
    images_train=images_train,    # (N_train, H, W, 3)
    c2ws_train=c2ws_train,        # (N_train, 4, 4)
    images_val=images_val,        # (N_val, H, W, 3)
    c2ws_val=c2ws_val,            # (N_val, 4, 4)
    c2ws_test=c2ws_test,          # (N_test, 4, 4)
    focal=focal                   # float
)

print(f"Dataset saved successfully!")
print(f"  Training images: {images_train.shape}")
print(f"  Validation images: {images_val.shape}")
print(f"  Test poses: {c2ws_test.shape}")
print(f"  Focal length: {focal:.2f}")

