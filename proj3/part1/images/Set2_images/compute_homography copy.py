import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
# matplotlib.use('TkAgg')
# import _tkinter


def to_rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def display_correspondences(im1_rgb, im2_rgb, im1_pts, im2_pts):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(im1_rgb); axs[0].set_title("Image 1"); axs[0].set_axis_off()
    axs[0].scatter(im1_pts[:,0], im1_pts[:,1], c='r', s=40)
    axs[1].imshow(im2_rgb); axs[1].set_title("Image 2"); axs[1].set_axis_off()
    axs[1].scatter(im2_pts[:,0], im2_pts[:,1], c='r', s=40)
    plt.tight_layout()
    plt.show(block=True)
    plt.close(fig)

def get_points_from_gui(im_rgb, n):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(im_rgb)

    # plt.title('Click on four points in the image')
    ax.set_title(f"Click points.\nExpected: {n} points.")
    ax.set_axis_off()

    pts = plt.ginput(n=n, timeout=0, show_clicks=True)

    pts = np.array(pts, dtype=float)
    ax.scatter(pts[:,0], pts[:,1], c= 'r', s = 40)
    plt.show(block = True) 
    plt.close(fig)
    return pts



def compute_H(im1_pts, im2_pts):
    #in this function, I need to show correspondences on both images, system of equations, and then solve for the homography and display the homography matrix
    #for every point in my correspondences, I need to create a system of equations and stack these into the A matrix which will become my overdetermined system of equations
    #then solve A using least squares, where solution is vector of H matrix entries 
    #I then reshape the vector into the H matrix
    #then return the H matrix
    # print(im1_pts)
    # print(im2_pts)
    A = []
    b = []
    for i in range(len(im1_pts)): 
        x, y = im1_pts[i]
        x_prime, y_prime = im2_pts[i]

        A.append([x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y])
        A.append([0, 0, 0, x, y, 1, -y_prime * x, -y_prime * y])
        b.append(x_prime)
        b.append(y_prime)
    A = np.array(A)
    b = np.array(b)
    #our b vector is now our vector of x_prime and y_prime values
    h_vec, residuals, rank, s = np.linalg.lstsq(A, b)

    h_vec = np.append(h_vec, 1)
    H = h_vec.reshape(3, 3)
    return H, A, b



# im1_bgr = cv2.imread("/Users/akshaanahuja/akshaanahuja.github.io/proj3/part1/images/Set1_images/Set1_img1.jpg") # cv reads in images as BGR
# im2_bgr = cv2.imread("/Users/akshaanahuja/akshaanahuja.github.io/proj3/part1/images/Set1_images/Set1_img2.jpg")
# im1 = to_rgb(im1_bgr) #convert to rgb for matplotlib
# im2 = to_rgb(im2_bgr)

# im1_pts = get_points_from_gui(im1, 6)
# im2_pts = get_points_from_gui(im2, 6)
# display_correspondences(im1, im2, im1_pts, im2_pts) #h x w x 3 images overlayed with the 2d coords we selected
# homography, A, b = compute_H(im1_pts, im2_pts)

# print("Homography matrix: ", homography)
# print("Homography matrix shape: ", homography.shape)
# print("A matrix: ",A)
# print("A shape: ", A.shape)
# print("b vector: ", b)
# print("b shape: ", b.shape)




