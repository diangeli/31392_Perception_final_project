# test 1
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import glob
import time
import keras.models

from classification.classification import classify_frame
from my_tracking.kallman_fiter import initialize_kalman, update, predict
from my_tracking.tracking import detect_object, depth_of_pixel
from visualize.show_frame import show_frame
from rectification.rectifying import rectify
from my_tracking.obj_class import obj_on_conveyr


if __name__ == '__main__':
    ####################################################################
    ##                         Initialisation                         ##
    ####################################################################
    # Creating a list with the image names of all the images
    w_occ_left   = list(glob.glob('videos/w_occ_left/*.png'))
    w_occ_right  = list(glob.glob('videos/w_occ_right/*.png'))
    wo_occ_left  = list(glob.glob('videos/wo_occ_left/*.png'))
    wo_occ_right = list(glob.glob('videos/wo_occ_right/*.png'))


    # Object detecter for stable camera. Will remove background.
    object_detector1 = cv2.createBackgroundSubtractorKNN(history=1000, dist2Threshold=1000, detectShadows=False)
    object_detector2 = cv2.createBackgroundSubtractorKNN(history=1000, dist2Threshold=1000, detectShadows=False)

    # Found object paramters
    new_count   = 0
    radius      = 2
    objects     = []                # List with all the objects found

    # Load the classification model
    model  = keras.models.load_model('classification/model_5.h5')


    # Creating the rectifying class, which will be used to rectify the frames
    frame_obj = rectify('rectification/camera_params.pkl')


    ####################################################################
    ##                           Main loop                            ##
    ####################################################################
    # Loop through each pair of images in the folder.
    for (img_name_L, img_name_R) in zip(w_occ_left, w_occ_right):
        s_time = time.perf_counter()
        # Load the current frame
        current_frame_L = cv2.imread(img_name_L)
        current_frame_R = cv2.imread(img_name_R)

        # Rectify the frame
        rect_L_RGB, rect_R_RGB = frame_obj.rectify(current_frame_L, current_frame_R) 

        # Regeion of interest of where the moving object will be placed
        start_roi   = (240, 940, 160, 180)        # (y, x, height, width)
        start_frame = rect_R_RGB[start_roi[0]:start_roi[0] + start_roi[2],start_roi[1]:start_roi[1] + start_roi[3]]
      
        ### Detect new object
        start_frame, bg_mask_start, init_BB , detected_new_object = detect_object(start_frame, object_detector1)

        # Bounding Box = (x, y, h, w)
        if detected_new_object:
            new_count += 1
        else: 
            new_count = 0


        # if new object on conveyor
        if detected_new_object and new_count == 9:

            objects.append(obj_on_conveyr(init_BB[-1], start_roi))    

            active_obj = [x.active for x in objects]

            obj = objects[-1]
            cv2.rectangle(rect_R_RGB, (int(obj.BBx), int(obj.BBy)), (int(obj.BBx+obj.w), int(obj.BBy+obj.h)), (255, 0, 255), 2)

            # Change counter so that it doesn't trigger the new object rule more than once for each object
            new_count = -1000
            
            # Initialize Kalman filter
            obj.kalman_init_params(initialize_kalman())
            continue

        
        if not detected_new_object:
            new_count = -1

        

        ### Detect moving object on conveyor 
        search_frame = rect_R_RGB[start_roi[0]:650,200:]
        search_frame, bg_mask, BB_list, detected = detect_object(search_frame, object_detector2)
        
        if detected and len(objects) > 0:
            # for loop through objects
            for obj in objects:
                # Use bounding box closest to the object
                if obj.center[0] >= 600:
                    BB = BB_list[-1]
                else:
                    BB = BB_list[0]

                if obj.area <= BB[2]*BB[3]:      # Object is not occluded
                    
                    # Update the objects parameters
                    obj.update_pos(BB)

                    # get BW images and calculate the depth of the object
                    rect_L_BW = cv2.cvtColor(rect_L_RGB, cv2.COLOR_BGR2GRAY)
                    rect_R_BW = cv2.cvtColor(rect_R_RGB, cv2.COLOR_BGR2GRAY)
                    depth, _  = depth_of_pixel(obj.center[0], obj.center[1], rect_L_BW, rect_R_BW)

                    # Before the occlusion
                    if obj.center[0] > 640:
                        obj.center[0] = int(obj.prevx + obj.BBx + obj.w - obj.prevRbound)
                        obj.center[1] = int(obj.prevy + obj.BBy + obj.h - obj.prevBbound)

                        obj.prev_pos()

                    z = np.array([[obj.center[0]], [obj.center[1]], [depth]])
                    obj.x, obj.P = update(obj.x, obj.P, z, obj.H, obj.R)
                    
                  
            
        if len(objects) > 0:
            for obj in objects:
            # Draw previous state
                if detected:
                    # Use bounding box closest to the object
                    if obj.center[0] >= 600:
                        BB = BB_list[-1]
                    else:
                        BB = BB_list[0]
                    
                    x1 = int(BB[0])
                    x2 = int(BB[0]+BB[2])
                    y1 = int(BB[1])
                    y2 = int(BB[1]+BB[3])

                    cv2.rectangle(search_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    
                    ########## Classify the object ##########
                    object_frame = search_frame[y1:y2,x1:x2]
                    label = classify_frame(object_frame, model)
                    cv2.putText(rect_R_RGB, label, (x1+300, y1+250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Predict next position
                obj.x, obj.P = predict(obj.x, obj.P, obj.F, obj.u)
                
                ### Draw the current tracked state 
                cv2.circle(search_frame, (int(obj.x[0]), int(obj.x[3])), int(radius), (0, 0, 255), 2)


        for obj in objects: 
            if obj.center[0] <= 100:
                objects.pop(0)
            
        

        # Show the frame
        if not show_frame(frames = [rect_R_RGB, start_frame, bg_mask_start, bg_mask],
                     frame_names = ['Frame', 'start_frame','BG mask start','BG mask']):
            break
        

        # Time it took for the frame to run
        print(f'The frame took {time.perf_counter() - s_time:.2f}sec. to run')

    # Destroy all the cv2.imshow windows
    cv2.destroyAllWindows()




    


