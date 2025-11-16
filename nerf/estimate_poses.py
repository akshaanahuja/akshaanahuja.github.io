import cv2
import numpy as np
import os
import glob
import time

# Hardcode parameters (modify as needed)
tag_size = 0.1  # Size of ArUco tag in meters (0.06m = 60mm)
calibration_file = "nerf/camera_calibration.npz"
object_images_dir = "/Users/akshaanahuja/akshaanahuja.github.io/nerf/mouse_images"

# Load camera calibration results
calibration_data = np.load(calibration_file)
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Create ArUco dictionary and detector parameters (4x4 tags)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Define 3D world coordinates for ArUco tag corners
# Tag corners relative to tag origin: [(0,0,0), (tag_size,0,0), (tag_size,tag_size,0), (0,tag_size,0)]
objp = np.array([
    [0, 0, 0],
    [tag_size, 0, 0],
    [tag_size, tag_size, 0],
    [0, tag_size, 0]
], dtype=np.float32)

# Get all image files from the object images directory
image_files = glob.glob(os.path.join(object_images_dir, "*.jpg"))
image_files.extend(glob.glob(os.path.join(object_images_dir, "*.JPG")))
image_files = sorted(image_files)

# Store camera poses (c2w matrices)
c2w_matrices = []
valid_images = []

# Loop through all object images
for image_path in image_files:
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        continue
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers in the image
    corners, ids, _ = detector.detectMarkers(gray)
    
    # Check if any markers were detected
    if ids is not None and len(ids) > 0:
        # Extract corner coordinates from the first detected tag
        # corners[i] has shape (1, 4, 2), we need (4, 2)
        tag_corners = corners[0][0]  # Shape: (4, 2)
        
        # Reshape for solvePnP: objectPoints should be (N, 3) and imagePoints should be (N, 2)
        object_points = objp.reshape(-1, 3)  # Shape: (4, 3)
        image_points = tag_corners.reshape(-1, 2)  # Shape: (4, 2)
        
        # Use cv2.solvePnP() to estimate camera pose
        ret, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )
        
        if ret:
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)  # R is (3, 3)
            
            # OpenCV's solvePnP returns world-to-camera transformation
            # We need to invert it to get camera-to-world (c2w)
            # w2c = [R | t], so c2w = [R^T | -R^T * t]
            R_w2c = R
            t_w2c = tvec.reshape(3, 1)
            
            # Compute c2w transformation
            R_c2w = R_w2c.T
            t_c2w = -R_c2w @ t_w2c
            
            # Build c2w matrix (3x4)
            c2w = np.hstack([R_c2w, t_c2w])
            
            # Store results
            c2w_matrices.append(c2w)
            valid_images.append(image_path)
            
            print(f"Pose estimated for {os.path.basename(image_path)}")
        else:
            print(f"Failed to estimate pose for {os.path.basename(image_path)}")
    else:
        print(f"No tag detected in {os.path.basename(image_path)}, skipping...")

print(f"\nSuccessfully estimated poses for {len(c2w_matrices)} images")

# Save poses and image paths
poses_file = "nerf/camera_poses.npz"
np.savez(poses_file,
         c2w_matrices=np.array(c2w_matrices),
         valid_images=np.array(valid_images))
print(f"Poses saved to {poses_file}")


try:
    import viser
    
    server = viser.ViserServer(share=True)
    
    # Get image dimensions from first valid image
    first_image = cv2.imread(valid_images[0])
    H, W = first_image.shape[:2]
    
    for i, (c2w, img_path) in enumerate(zip(c2w_matrices, valid_images)):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Calculate field of view
        fov = 2 * np.arctan2(H / 2, camera_matrix[0, 0])
        aspect = W / H
        
        # Convert rotation matrix to quaternion
        wxyz = viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz
        
        # Add camera frustum
        server.scene.add_camera_frustum(
            f"/cameras/{i}",
            fov=fov,
            aspect=aspect,
            scale=0.02,
            wxyz=wxyz,
            position=c2w[:3, 3],
            image=img_rgb
        )
    
    print("Visualization server started. Press Ctrl+C to stop.")
    while True:
        time.sleep(0.1)
        
except ImportError:
    print("viser not installed. Skipping visualization.")
    print("Install with: pip install viser")
except KeyboardInterrupt:
    print("\nVisualization stopped.")

