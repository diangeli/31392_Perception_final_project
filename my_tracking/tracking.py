
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import glob
import time
import os
import sys
sys.path.insert(0,os.getcwd())
from rectification.rectifying import rectify



baseline = 0.54

def detect_object(frame, object_detector):
    
    # Initialize variable to return
    boundRect       = None
    bounding_box    = []
    detected_object = False
    
    # Create mask
    mask = object_detector.apply(frame)
    ### Erode it and dilate it to avoid small points
    # Performing closing on the mask, to remove noise with a 2x2 box filter
    kernel = np.ones((2,2), np.uint8) 
    mask   = cv2.erode(mask, kernel, iterations=1) 
    mask   = cv2.dilate(mask, kernel, iterations=1)

    # Performing operning on the mask, to close holes in the box,
    # as we only work with solid objects, with a 7x7 kernel
    kernel = np.ones((10,10), np.uint8) 
    mask   = cv2.dilate(mask, kernel, iterations=1)
    mask   = cv2.erode(mask, kernel, iterations=1)

        
    # Find contours in image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,)
    
    for cnt in contours:
        # Calculate the area of the specific contour
        area = cv2.contourArea(cnt)
        
        # Only look at contours with an area of over 2k pixels
        if area > 2000:
            # Detected object
            detected_object = True
            bounding_box.append(np.zeros(4))
            
            # Draw bounding rectangle
            boundRect = cv2.boundingRect(cnt)
            bounding_box[-1][0] = boundRect[0]
            bounding_box[-1][1] = boundRect[1]
            bounding_box[-1][2] = boundRect[2]
            bounding_box[-1][3] = boundRect[3]


    if len(bounding_box) > 1:
        x_pos = [calc_center_for_BB(BB)[0] for BB in bounding_box]
        areas = [BB[2] * BB[3] for BB in bounding_box]

        # go backwards thorugh the list, by skipping the last
        for i in list(range(len(x_pos)-2,-1,-1)):
            if abs(x_pos[i] - x_pos[i+1]) <= 400:                               # cnt for same obj
                bounding_box.pop(i + np.argmin((areas[i], areas[i+1])))      # Pop the obj with lowest area

    return frame, mask, bounding_box, detected_object

def calc_center_for_BB(BB):
    x = int(BB[0] + BB[2]/2)
    y = int(BB[1] + BB[3]/2)
    return(x,y)

def depth_map(imgL, imgR):

    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 17 # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
                                    minDisparity=-1,
                                    numDisparities=13*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
                                    blockSize=window_size,
                                    P1=8 * 11 * window_size,
                                    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
                                    P2=32 * 11 * window_size,
                                    disp12MaxDiff=12,
                                    uniquenessRatio=6,
                                    speckleWindowSize=200,
                                    speckleRange=2*16,
                                    preFilterCap=63,
                                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 3.0
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgR, None, displ) 

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=255, alpha=0, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg


def depth_of_pixel(x, y, img_L, img_R):
    #Lower the resolution
    img_L = cv2.resize(img_L, (img_L.shape[::-1][0]//2, img_L.shape[::-1][1]//2))
    img_R = cv2.resize(img_R, (img_R.shape[::-1][0]//2, img_R.shape[::-1][1]//2))
    
    # Create the Stereo map for the frame
    depth_img = depth_map(img_L[50:,:], img_R[50:,:])
    cv2.imshow('depth map', depth_img)

    return depth_img[y//2-50, x//2], depth_img

if __name__ == '__main__':

    # Creating a list with the image names of all the images
    w_occ_left   = list(glob.glob('videos/w_occ_left/*.png'))
    w_occ_right  = list(glob.glob('videos/w_occ_right/*.png'))
    wo_occ_left  = list(glob.glob('videos/wo_occ_left/*.png'))
    wo_occ_right = list(glob.glob('videos/wo_occ_right/*.png'))


    # Object detection from Stable camera. 
    object_detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold = 100)

    # Create the initial features from the first frame
    init_frame    = True
    draw_eplines  = False
    draw_rect_img = False

    # Creating the rectifying class, which will be used to rectify each image (Also contains the camera matrix for both lenses)
    frame_obj = rectify('rectification/camera_params.pkl')


    for (img_name_l, img_name_r) in zip(w_occ_left, w_occ_right):
        s_time = time.perf_counter()
        # Load the current frame
        current_frame_L = cv2.imread(img_name_l)
        current_frame_R = cv2.imread(img_name_r)

        # Create a black and white version of the frames
        cur_L_BW = cv2.cvtColor(current_frame_L, cv2.COLOR_BGR2GRAY)
        cur_R_BW = cv2.cvtColor(current_frame_R, cv2.COLOR_BGR2GRAY)

        # Rectify the frame #TODO
        rect_L_BW, rect_R_BW = frame_obj.rectify(cur_L_BW, cur_R_BW) 
        if draw_rect_img:
            f, axarr = plt.subplots(2,1)
            axarr[0].imshow(rect_L_BW, 'gray')
            axarr[1].imshow(rect_R_BW, 'gray')
            plt.show()
            plt.imshow(rect_R_BW,'gray')
            plt.show()
        
        # Epipolar lines
        if draw_eplines:
            frame_obj.draw_epipolar_lines(rect_L_BW, rect_R_BW)


        depth_of_pixel(100, 10, rect_L_BW, rect_R_BW)

        # Only run this for the first left and right frame.
        if init_frame:
            
            # Save the left and right images as the reference images 
            ref_img_L = current_frame_L
            ref_img_R = current_frame_R

            # It will no longer be the initial frame, and this if-statement is skipped.
            init_frame = False
            continue


        # Save the frames as reference frame for the next frames.
        ref_img_L = current_frame_L
        ref_img_R = current_frame_R


        # Track the objects
        frame_tracked, mask, init_BB, detected_new_object = detect_object(current_frame_R[100:-100,100:-100], object_detector)
        
        # Show the background mask and the frame upon which the contours are drawn upon.
        cv2.imshow('Frame', frame_tracked)
        cv2.imshow('BG Mask', mask)
        k = cv2.waitKey(1)
        if k==27:    # Esc key to stop
            break

        print(f'The frame took {time.perf_counter() - s_time}sec. to run')

        

    cv2.destroyAllWindows()



