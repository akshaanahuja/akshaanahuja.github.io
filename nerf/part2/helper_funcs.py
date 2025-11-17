import numpy as np



#camera to world coordinate conversion
    #camera_coords is N, 3
def camera_to_world_transform(c2w_matrix, camera_coords):
    #batching just will apply c2w matrix on camera coords all coords at once
    camera_coords_homogeneous = np.concatenate([camera_coords, np.ones((camera_coords.shape[0], 1))], axis = 1)

    world_coords = c2w_matrix @ camera_coords_homogeneous.T #(4, 4) @ (4, N) = (4, N)

    #remove homogeneous coordinate
    world_coords = world_coords.T[:, :3]
    return world_coords


#pixel to camera coordinate conversion
#invert the camera proj -> convert 2d pixel coords u, v back to 3d camera coords. 
#first step in ray casting pipeline
def pixel_to_camera(K, uv, s):
    #K is 3, 3 camera intrinsict containing focal len and principal point
    #uv is batched pixel coords
    #s is depth factor (s = zc because zc is depth along optical axis (for ray casting this is set to 1 because we want normalized ray direction))
    
    K_inv = np.linalg.inv(K)
    uv_homogeneous = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis = 1)
    camera_coords = s * uv_homogeneous.T #(N, 3)
    camera_coords = K_inv @ camera_coords #(3, 3) @ (3, N) = (3, N)
    return camera_coords.T[:, :3] #(N, 3)


#pixel to ray
def pixel_to_ray(K, c2w_matrix, uv):
    r_o_single = c2w_matrix[:3, 3] 
    #start with a pixel: 
    x_c = pixel_to_camera(K, uv, 1)  
    X_w = camera_to_world_transform(c2w_matrix, x_c)  

    #need this to broadcast r_o to match batch size
    r_o = np.tile(r_o_single, (uv.shape[0], 1)) 
    
    r_d = X_w - r_o  
    r_d = r_d / np.linalg.norm(r_d, axis = 1, keepdims = True)  

    return r_o, r_d



#SAMPLING:

#sample rays from images:
def sample_rays_from_images(K, c2w_matrices, images, N):
    ray_origins = []
    ray_directions = []
    ray_colors = []
    M = len(images)
    #pipeline:
    #for each image
        #for N//M range:
            #pixel_to_ray((K, c2w_matrix, pixel))
            #save to running list of rays (ro, rd) pairs
    for i in range(len(images)):
        image = images[i] #h, w, 3
        c2w_matrix = c2w_matrices[i] #4, 4
        #create coordinate grid for image
        y_coords, x_coords = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing = 'ij')
        #sample N//M pixels from coordinate grid
        indices = np.random.choice(x_coords.size, size = N//M, replace = False)
        x_sampled = x_coords.flatten()[indices]
        y_sampled = y_coords.flatten()[indices]


        colors = image[y_sampled.astype(int), x_sampled.astype(int)]  #(N//M, 3)

        uv = np.stack([x_sampled, y_sampled], axis = 1) #batched ops
        uv = uv + 0.5
        #convert to ray origin and direction
        ro, rd = pixel_to_ray(K, c2w_matrix, uv)

        ray_origins.append(ro)
        ray_directions.append(rd)
        ray_colors.append(colors)
    
    ray_origins = np.concatenate(ray_origins, axis=0)  # (N, 3)
    ray_directions = np.concatenate(ray_directions, axis=0)  # (N, 3)
    ray_colors = np.concatenate(ray_colors, axis=0)  # (N, 3)
    
    return ray_origins, ray_directions, ray_colors


def sample_points_along_rays(r_o, r_d, N_samples = 32, near = 2.0, far = 6.0, during_training = True):
    t = np.linspace(near, far, N_samples)  # Shape: (N_samples,)
    if during_training:
        t_width = (far - near) / (N_samples - 1)
        t = t + (np.random.rand(N_samples)) * t_width
    
    # Reshape for broadcasting: r_o (N_rays, 3), r_d (N_rays, 3), t (N_samples,)
    # We want: (N_rays, 3, 1) + (1, 1, N_samples) * (N_rays, 3, 1) -> (N_rays, 3, N_samples)
    X = r_o[:, :, np.newaxis] + t[np.newaxis, np.newaxis, :] * r_d[:, :, np.newaxis]
    X = X.transpose(0, 2, 1)  # Transpose to (N_rays, N_samples, 3)
    return X 
