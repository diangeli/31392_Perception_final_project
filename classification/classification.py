
import tensorflow as tf
import cv2
import numpy as np



def get_label(input):
    labels = {0:'book',1:'box',2:'mug'}
    return labels[np.argmax(input)]

def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

def classify_frame(frame, model):
    frame = cv2.resize(frame, (256, 256))
    frame = tf.keras.preprocessing.image.img_to_array(frame)
    frame = np.array([frame])
    frame = tf.cast(frame/255. ,tf.float32)
    label = get_label(model.predict(frame))
        
    return label