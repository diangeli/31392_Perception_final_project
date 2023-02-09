import cv2


def show_frame(frames, frame_names):
    """
    Takes as input a frame and frame name or a list of frames and a list of frame names, and shows them.
    """

    # turn into a list if only a single frame or name is given and make sure the types are correct
    if type(frames) != list:
        frames = [frames]

    if type(frame_names) != list:
        assert type(frame_names) == str, 'frame_names has to be a string or a list of strings.\n'
        frame_names = [frame_names]
    else:
        assert type(frame_names[0]) == str, 'frame_names has to be a string or a list of strings.\n'


    #Show the different frames
    for i, frame in enumerate(frames):
        cv2.imshow(frame_names[i], frame)
    
    # Make an exit key
    k = cv2.waitKey(1)
    if k==27:    # Esc key to stop
        return False
    else:
        return True