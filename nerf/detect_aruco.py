import cv2
import numpy as np
import os
import glob

# Hardcode parameters (modify as needed)
tag_size = 0.06  # Size of ArUco tag in meters (0.06m = 60mm)

# Create ArUco dictionary and detector parameters (4x4 tags)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Define 3D world coordinates for ArUco tag corners
# Tag corners relative to tag origin: [(0,0,0), (0.02,0,0), (0.02,0.02,0), (0,0.02,0)]
objp = np.array([
    [0, 0, 0],
    [tag_size, 0, 0],
    [tag_size, tag_size, 0],
    [0, tag_size, 0]
], dtype=np.float32)

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in world space
imgpoints = []  # 2D points in image plane

# Get all image files from the directory
images_dir = "/Users/akshaanahuja/akshaanahuja.github.io/nerf/calibration_images/images"

# Get all JPG files from the directory
image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
image_files.extend(glob.glob(os.path.join(images_dir, "*.JPG")))
image_files = sorted(image_files)

# Store image shape (will be set from first successfully processed image)
image_shape = None

# Loop through all calibration images by explicit index
for image_path in image_files:
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Store image shape from first image
    if image_shape is None:
        image_shape = gray.shape[::-1]  # (width, height)
    
    # Detect ArUco markers in the image
    corners, ids, _ = detector.detectMarkers(gray)
    
    # Check if any markers were detected
    if ids is not None:
        # Process each detected tag
        image_corners = []
        object_points = []
        
        for i in range(len(ids)):
            # Extract the 4 corner coordinates from the detected tag
            tag_corners = corners[i][0]  # Shape: (4, 2)
            
            # Collect detected corners
            image_corners.extend(tag_corners)
            
            # Collect corresponding 3D world coordinates
            object_points.extend(objp)
        
        # Convert to numpy arrays
        image_corners = np.array(image_corners, dtype=np.float32)
        object_points = np.array(object_points, dtype=np.float32)
        
        # Store for calibration
        objpoints.append(object_points)
        imgpoints.append(image_corners)
        
        print(f"Detected {len(ids)} tag(s) with {len(image_corners)} corners in {os.path.basename(image_path)}")

# Use cv2.calibrateCamera() to compute camera intrinsics and distortion coefficients
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_shape, None, None
)

# Save calibration results
calibration_file = "camera_calibration.npz"
np.savez(calibration_file, 
         camera_matrix=camera_matrix, 
         dist_coeffs=dist_coeffs)
print(f"Calibration saved to {calibration_file}")
