import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle



criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def camera_matrix():
    # Loading in the images from the folder 'rs'
    # The number vertical and horizontal corners within the checkerboard.
    nb_vertical   = 6       # height
    nb_horizontal = 9       # width


    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nb_vertical * nb_horizontal,3), np.float32)
    objp[:,:2] = np.mgrid[0:nb_horizontal,0:nb_vertical].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    img_ptsL   = [] # 2d points in Left image plane.
    img_ptsR   = [] # 2d points in Right image plane.
    obj_points_L = [] # 3d point in real world space
    obj_points_R = [] # 3d point in real world space

    images = glob.glob('Stereo_calibration_images/*.png')
    assert images

    # Loop through most of the images ( the ones that are not blurry)
    blurry_imgs = [9, 11, 20, 36, 37, 38, 47, 48]
    for i in [x for x in range(len(images)//2) if not any(ele == x for ele in blurry_imgs) ]:
        imgL = cv2.imread(images[i])
        imgR = cv2.imread(images[i + len(images)//2])
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)        
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        
        # Implementation of the findChessboardCorners
        retL, cornersL = cv2.findChessboardCorners(grayL, patternSize = (nb_horizontal, nb_vertical))
        retR, cornersR = cv2.findChessboardCorners(grayR, patternSize = (nb_horizontal, nb_vertical))

        # If found, add object points, image points (after refining them)

        if retL:
            obj_points_L.append(objp)

            corners2 = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria_stereo)
            img_ptsL.append(corners2)

            # Draw and display the corners
            imgL = cv2.drawChessboardCorners(imgL, (nb_horizontal, nb_vertical), cornersL, retL)
            cv2.imshow(f'imgL {i}',imgL)
            cv2.waitKey(1)

        if retR:
            obj_points_R.append(objp)

            corners2 = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria_stereo)
            img_ptsR.append(corners2)

            # Draw and display the corners
            imgR = cv2.drawChessboardCorners(imgR, (nb_horizontal, nb_vertical), cornersR, retR)
            cv2.imshow(f'imgR {i}',imgR)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    
    # Alpha value for how much the pixels in the images are filtered away after calibration
    alpha = 0

    img_shape =  grayL.shape[::-1]
    # Obtain the camera matrix for the Left camera
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_points_L, img_ptsL, img_shape, None, None)


    # Obtain the camera matrix for the Right camera
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_points_R, img_ptsR, img_shape, None, None)


    flags = 0
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    flags |= cv2.CALIB_FIX_ASPECT_RATIO
    flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    flags |= cv2.CALIB_ZERO_TANGENT_DIST
    flags |= cv2.CALIB_RATIONAL_MODEL

    # This step is performed to transform between the two cameras and calculate Essential and Fundamenatl matrix
    retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = \
                    cv2.stereoCalibrate(obj_points_L, img_ptsL, img_ptsR, mtxL, distL, mtxR,
                                        distR, img_shape, criteria_stereo, flags)
    
    # Stereo rectify the images
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = \
                    cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR,
                                    img_shape, Rot, Trns, flags=cv2.CALIB_ZERO_DISPARITY, alpha = alpha)


    # Saving the objects:
    with open('rectification/camera_params.pkl', 'wb') as camera_matrix:
        pickle.dump([mtxL, new_mtxL, distL, rect_l, proj_mat_l, roiL, mtxR, new_mtxR, distR, rect_r, proj_mat_r, roiR, Fmat, Q, img_shape], camera_matrix)




class rectify:

    def __init__(self, params_path):
        # Load the camera paramters from the pickle file
        with open(params_path, 'rb') as camera_params:
            params = pickle.load(camera_params)
        
        # Assign variables for each parameter
        
        self.K_L     = params[0]             # Left camera matrix
        self.K_R     = params[6]             # Right camera matrix

        self.new_K_L = params[1]             # Left camera matrix after calibration
        self.new_K_R = params[7]             # Right camera matrix after calibration

        self.dist_L  = params[2]             # Left distortion matrix
        self.dist_R  = params[8]             # Right distortion matrix
        
        self.rect_L  = params[3]             # Left rectification matrix
        self.rect_R  = params[9]             # Right rectification

        self.proj_L  = params[4]             # Left projection matrix
        self.proj_R  = params[10]            # Right projection matrix

        self.roi_L   = params[5]             # Region of interest for the left camera
        self.roi_R   = params[11]             # Region of interest for the right camera

        self.F_mat   = params[12]            # The fundamental matrix

        self.Q       = params[13]            # The Q matrix for dispersion

        self.img_shape = params[14]          # The shape of the img

        # Undistor and rectify map
        self.stereo_map_L = cv2.initUndistortRectifyMap(self.K_L,self.dist_L,self.rect_L,self.new_K_L, self.img_shape,cv2.CV_32FC2)
        self.stereo_map_R = cv2.initUndistortRectifyMap(self.K_R, self.dist_R, self.rect_R, self.new_K_R, self.img_shape, cv2.CV_32FC2)



    def rectify(self, imgL, imgR):
        # Remap the images with the stereo maps
        dst_L = cv2.remap(imgL,self.stereo_map_L[0],self.stereo_map_L[1],cv2.INTER_LINEAR)
        dst_R = cv2.remap(imgR,self.stereo_map_R[0],self.stereo_map_R[1],cv2.INTER_LINEAR)

        # Crop the images
        x,y,w,h = self.roi_L
        dst_L = dst_L[y:y+h, x:x+w]
        
        x,y,w,h = self.roi_R
        dst_R = dst_R[y:y+h, x:x+w]


        return dst_L, dst_R

    
    def calc_ep_lines(self, pts1, pts2):
        
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, self.F_mat)
        lines1 = lines1.reshape(-1, 3)

        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, self.F_mat)
        lines2 = lines2.reshape(-1, 3)
        
        return lines1, lines2
        
    


    def drawlines(self, img1, lines):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r,c = img1.shape
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
        for r in lines:
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        return img1

    def extract_keypoints_sift(self, img1, img2):
        # Using SIFT to create keypoints between the left and right images
        # Create the SIFT object:
        sift = cv2.xfeatures2d.SIFT_create()
        
        # Find keypoints and descriptors directly
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        # Match the points in each image with FLANN
        # parameters
        FLANN_INDEX_KDTREE = 1
        index_params  = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        
        # Match the points
        flann   = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        
        # Need to take only the good matches
        match_points1 = list()
        match_points2 = list()
        for (m1,m2) in matches:
            if m1.distance < 0.7 * m2.distance:
                match_points1.append(kp1[m1.queryIdx].pt)
                match_points2.append(kp2[m1.trainIdx].pt)
        
        # Convert the matched points into numpy arrays
        self.p1 = np.array(match_points1).astype(float)
        self.p2 = np.array(match_points2).astype(float)


    def draw_epipolar_lines(self,img_L, img_R):
        self.extract_keypoints_sift(img_L, img_R)

        lines1, lines2 = self.calc_ep_lines(self.p1, self.p2)
        
        ep_img_L = self.drawlines(img_L, lines1[::15])
        ep_img_R = self.drawlines(img_R, lines2[::15])
        plt.subplot(121),plt.imshow(ep_img_L)
        plt.subplot(122),plt.imshow(ep_img_R)
        plt.show()

if __name__ == '__main__':
    camera_matrix()
    frame_obj = rectify('rectification/camera_params.pkl')

    images = glob.glob('Stereo_calibration_images/*.png')
    imgL = cv2.imread(images[0],0)
    imgR = cv2.imread(images[0 + len(images)//2],0)
    

    cur_L_BW, cur_R_BW = frame_obj.rectify(imgL, imgR)
    plt.subplot(211),plt.imshow(cur_L_BW, 'gray')
    plt.subplot(212),plt.imshow(cur_R_BW, 'gray')
    plt.show()

    cur_L_BW, cur_R_BW = frame_obj.rectify(imgL, imgR)
    plt.subplot(211),plt.imshow(cur_L_BW, 'gray')
    plt.subplot(212),plt.imshow(cur_R_BW, 'gray')
    plt.show()


    pass

