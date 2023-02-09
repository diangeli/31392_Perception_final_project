# Perseption_final_project
## Object tracker
This python module, will track objects on a conveyer belt, by using background subtraction to find the object, a Kalman filter to predict where it will go, and a classification neural network to predict what kind of object it is. 

## Prerequisites
* python:       3.8
* numpy:        1.21.5
* CV2:          4.5.5
* matplotlib:   3.5.1
* Tensorflow:   2.8
* Pandas:       1.4.2


## How to install and the run the program
When the project is downloaded you have to create a folder called 'videos', and in that folder put 4 new folders;
    - 'w_occ_left'   for all the images of the left camera With OCClusion
    - 'w_occ_right'  for all the images of the right camera With OCClusion
    - 'wo_occ_left'  for all the images of the left camera WithOut OCClusion
    - 'wo_occ_right' for all the images of the right camera WithOut OCClusion

The program will be run by the main.py file. 

In order to change the videos between occlussion and non occlusion go to line 49 in main.py and write:

 - Occlusion:    for (img_name_L, img_name_R) in zip(w_occ_left, w_occ_right):
 - NO occlusion: for (img_name_L, img_name_R) in zip(wo_occ_left, wo_occ_right):
