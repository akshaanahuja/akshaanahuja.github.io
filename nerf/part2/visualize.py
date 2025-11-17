# import viser, time  # pip install viser
# import numpy as np
# from helper_funcs import sample_rays_from_images, sample_points_along_rays


# # Load data
# data = np.load(f"nerf/part2/datasets/lego_200x200.npz")

# # Training images: [100, 200, 200, 3]
# images_train = data["images_train"] / 255.0

# # Cameras for the training images 
# # (camera-to-world transformation matrix): [100, 4, 4]
# c2ws_train = data["c2ws_train"]

# # Camera focal length
# focal = data["focal"]  # float

# # Get image dimensions
# H, W = images_train.shape[1:3]

# # Construct camera intrinsic matrix K from focal length
# K = np.array([
#     [focal, 0, W / 2],
#     [0, focal, H / 2],
#     [0, 0, 1]
# ])

# # Sample rays from images
# N_rays = 100  # Number of rays to visualize
# rays_o, rays_d, ray_colors = sample_rays_from_images(
#     K, c2ws_train, images_train, N_rays
# )

# # Sample points along rays
# points = sample_points_along_rays(
#     rays_o, rays_d, 
#     N_samples=32, 
#     near=2.0, 
#     far=6.0, 
#     during_training=True
# )

# # Create viser server
# server = viser.ViserServer(share=True)

# # Add camera frustums for all training images
# for i, (image, c2w) in enumerate(zip(images_train, c2ws_train)):
#     server.add_camera_frustum(
#         f"/cameras/{i}",
#         fov=2 * np.arctan2(H / 2, focal),
#         aspect=W / H,
#         scale=0.15,
#         wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
#         position=c2w[:3, 3],
#         image=image
#     )

# # Add rays (visualize as lines from origin to far point)
# for i, (o, d) in enumerate(zip(rays_o, rays_d)):
#     # Create a line from ray origin to a point 6 units away
#     far_point = o + d * 6.0
#     server.add_spline_catmull_rom(
#         f"/rays/{i}", 
#         positions=np.stack((o, far_point)),
#     )

# # Add sampled points along rays as point cloud
# # points shape is (N_rays, N_samples, 3), need to flatten to (N_rays * N_samples, 3)
# server.add_point_cloud(
#     f"/samples",
#     colors=np.zeros_like(points.reshape(-1, 3)),  # Gray points
#     points=points.reshape(-1, 3),  # Flatten to (N_rays * N_samples, 3)
#     point_size=0.02,
# )

# print(f"Visualization server started!")
# print(f"Visualizing {N_rays} rays with {points.shape[1]} samples per ray")
# print(f"Total points: {points.reshape(-1, 3).shape[0]}")

# while True:
#     time.sleep(0.1)  # Wait to allow visualization to run

# Visualize Cameras, Rays and Samples
import viser, time
import numpy as np
from helper_funcs import sample_rays_from_images, sample_points_along_rays, pixel_to_ray
from dataloader import create_dataloader

# Load data
data = np.load(f"nerf/part2/datasets/lego_200x200.npz")

# Training images: [100, 200, 200, 3]
images_train = data["images_train"] / 255.0

# Cameras for the training images 
# (camera-to-world transformation matrix): [100, 4, 4]
c2ws_train = data["c2ws_train"]

# Camera focal length
focal = data["focal"]  # float

# Get image dimensions
H, W = images_train.shape[1:3]

# Construct camera intrinsic matrix K from focal length
K = np.array([
    [focal, 0, W / 2],
    [0, focal, H / 2],
    [0, 0, 1]
])

# Helper function to get all rays/UVs/pixels from first image for testing
def get_all_rays_from_first_image(images_train, K, c2ws_train):
    """Get all rays, UVs, and pixels from the first image for testing"""
    # Get first image and its camera
    image = images_train[0]  # (H, W, 3)
    c2w = c2ws_train[0]  # (4, 4)
    
    # Create coordinate grid for all pixels
    y_coords, x_coords = np.meshgrid(
        np.arange(image.shape[0]), 
        np.arange(image.shape[1]), 
        indexing='ij'
    )
    
    # Flatten to get all pixel coordinates
    x_flat = x_coords.flatten()  # (H*W,)
    y_flat = y_coords.flatten()  # (H*W,)
    
    # Store UVs as integer coordinates (xy format)
    uvs = np.stack([x_flat, y_flat], axis=1)  # (H*W, 2) - xy coordinates
    
    # Get pixel colors
    pixels = image[y_flat, x_flat]  # (H*W, 3)
    
    # Convert to rays (add 0.5 offset for pixel center)
    uv_with_offset = uvs + 0.5
    rays_o, rays_d = pixel_to_ray(K, c2w, uv_with_offset)
    
    return uvs, pixels, rays_o, rays_d

# Get all rays from first image for testing
dataset_uvs, dataset_pixels, dataset_rays_o, dataset_rays_d = get_all_rays_from_first_image(images_train, K, c2ws_train)

# Create a simple namespace object for the checks (matching expected structure)
class DatasetForChecks:
    def __init__(self, uvs, pixels, rays_o, rays_d):
        self.uvs = uvs
        self.pixels = pixels
        self.rays_o = rays_o
        self.rays_d = rays_d

dataset = DatasetForChecks(dataset_uvs, dataset_pixels, dataset_rays_o, dataset_rays_d)

# This will check that your uvs aren't flipped
uvs_start = 0
uvs_end = 40_000
sample_uvs = dataset.uvs[uvs_start:uvs_end]  # These are integer coordinates of widths / heights (xy not yx) of all the pixels in an image
# uvs are array of xy coordinates, so we need to index into the 0th image tensor with [0, height, width], so we need to index with uv[:,1] and then uv[:,0]
assert np.all(images_train[0, sample_uvs[:,1], sample_uvs[:,0]] == dataset.pixels[uvs_start:uvs_end])
print("✓ UV check passed: UVs are correctly oriented")

# Use dataloader to get rays for visualization
N_rays = 100  # Number of rays to visualize
dataloader = create_dataloader(K, c2ws_train, images_train, N_rays, batch_size=1, shuffle=True)

# Get one batch from dataloader
for batch in dataloader:
    ray_origins, ray_directions, ray_colors = batch
    # Convert from torch tensors to numpy
    rays_o = ray_origins.squeeze(0).numpy()  # Remove batch dim: (N, 3)
    rays_d = ray_directions.squeeze(0).numpy()  # (N, 3)
    break

# For visualization, also get some rays from the first image using the dataset
# Uncomment this to display random rays from the first image
indices = np.random.randint(low=0, high=40_000, size=100)

# Uncomment this to display random rays from the top left corner of the image
# indices_x = np.random.randint(low=100, high=200, size=100)
# indices_y = np.random.randint(low=0, high=100, size=100)
# indices = indices_x + (indices_y * 200)

# Use rays from the dataset for visualization
data = {"rays_o": dataset.rays_o[indices], "rays_d": dataset.rays_d[indices]}

# Sample points along rays
points = sample_points_along_rays(
    data["rays_o"], data["rays_d"], 
    N_samples=32, 
    near=2.0, 
    far=6.0, 
    during_training=True
)

# Create viser server
server = viser.ViserServer(share=True)

# Add camera frustum for only the first image
server.add_camera_frustum(
    f"/cameras/0",
    fov=2 * np.arctan2(H / 2, focal),
    aspect=W / H,
    scale=0.15,
    wxyz=viser.transforms.SO3.from_matrix(c2ws_train[0][:3, :3]).wxyz,
    position=c2ws_train[0][:3, 3],
    image=images_train[0]
)

# Add rays from first image (visualize as lines from origin to far point)
for i, (o, d) in enumerate(zip(data["rays_o"], data["rays_d"])):
    positions = np.stack((o, o + d * 6.0))
    server.add_spline_catmull_rom(
        f"/rays/{i}", 
        positions=positions,
    )

# Add sampled points along rays as point cloud
server.add_point_cloud(
    f"/samples",
    colors=np.zeros_like(points.reshape(-1, 3)),  # Gray points
    points=points.reshape(-1, 3),  # Flatten to (N_rays * N_samples, 3)
    point_size=0.03,
)

print(f"Visualization server started!")
print(f"Visualizing {len(indices)} rays from first image with {points.shape[1]} samples per ray")
print(f"Total points: {points.reshape(-1, 3).shape[0]}")

while True:
    time.sleep(0.1)  # Wait to allow visualization to run