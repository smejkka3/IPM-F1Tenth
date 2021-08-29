import cv2
import numpy as np
import matplotlib.image as mpimg
import glob
import os

# The function cal_undistort takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

# Read in and make a list of calibration images
images = glob.glob('Advanced_Lanes_Detection-master/camera_cal/calibration*.jpg')

# Arrays to store object points and image points from all the images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane 

# Prepare object points 
objp = np.zeros((6*9,3),np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x,y coordinates

# Create the undistorted_images directory within the camera_cal directory
if not os.path.exists("Advanced_Lanes_Detection-master/camera_cal/undistorted_images"):
    os.makedirs("Advanced_Lanes_Detection-master/camera_cal/undistorted_images")

for fname in images:
    # read in each image
    img = mpimg.imread(fname)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
    # If corners are found, add object and image points 
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        
        # draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        
        # get the undistorted version of the calibration image
        undistorted = cal_undistort(img, objpoints, imgpoints)


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    # Apply threshold
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(warped, window_width, window_height, margin):
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    past_centroids = []
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))
        past_centroids.append(np.average(window_centroids[-3:], axis = 0))

    return past_centroids

#Read in pictures
road_images = glob.glob('assets/*.jpg')

for fname in road_images:
    # read in each image
    img = cv2.imread(fname)
    undistorted = cal_undistort(img, objpoints, imgpoints)
    if fname != "None":
        #display image
        cv2.namedWindow('undistorted_img')
        cv2.imshow("undistorted_img", undistorted)
        cv2.waitKey(0)
   
    s_thresh=(90, 255) 
    v_thresh=(200, 255) 
    sx_thresh=(20, 255) 
    sy_thresh=(10, 255) 
    
    gradx = abs_sobel_thresh(img, orient='x', thresh=sx_thresh)
    grady = abs_sobel_thresh(img, orient='y', thresh=sy_thresh)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(float)
    s_channel = hls[:,:,2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(float)
    v_channel = hsv[:,:,2]
    
    # Threshold color channel
    s_binary = np.zeros_like(img[:,:,0])
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1]) & (v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

    binary_final = np.zeros_like(s_channel)
    binary_final[((gradx == 1) & (grady == 1) | (s_binary == 1))] = 255
    if fname != "None":
        #display image
        cv2.namedWindow('binary_img')
        cv2.imshow("binary_img", binary_final)
        cv2.waitKey(0)

    # Apply a birds-eye view's perspective transform
    src = np.float32([[264, 678],[1042, 678],[686, 452],[596, 452]])
    dst = np.float32([[320, 720],[960, 720],[960, 0],[320, 0]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (binary_final.shape[1],binary_final.shape[0])
    warped = cv2.warpPerspective(binary_final, M, img_size, flags=cv2.INTER_LINEAR)
    if fname != "None":
        #display image
        cv2.namedWindow('bev_img')
        cv2.imshow("bev_img", warped)
        cv2.waitKey(0)

    # Apply a sliding window search
    # window settings
    window_width = 30 
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 30 # How much to slide left and right for searching
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)
    # If we found any window centers
    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channle 
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results 
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
    if fname != "None":
        #display image
        cv2.namedWindow('output_img')
        cv2.imshow("output_img", output)
        cv2.waitKey(0)
