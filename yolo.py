
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML

import keras # broken for keras >= 2.0, use 1.2.2
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape

from utils import load_weights, Box, yolo_net_out_to_car_boxes, draw_box

keras.backend.set_image_dim_ordering('th')

model = Sequential()
model.add(Convolution2D(16, 3, 3,input_shape=(3,448,448),border_mode='same',subsample=(1,1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(Convolution2D(64,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(Convolution2D(128,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(Convolution2D(256,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(Convolution2D(512,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(Convolution2D(1024,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Convolution2D(1024,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Convolution2D(1024,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(4096))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(1470))

model.summary()

load_weights(model,'./Load_files/yolo-tiny.weights')

def frame_func(imgcv):
    with open('./Load_files/calibparams.pickle', 'rb') as handle:
        K = pickle.load(handle)
    mtx = K['mtx']
    dist = K['dist']
    newcameramtx = K['newcameramtx']
    roi = K['roi']

    # Undistort image
    dst = cv2.undistort(imgcv, mtx, dist, None, newcameramtx)
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    undistorted_img = dst;
    imgcv = undistorted_img;

    # src= np.float32([[540, 400],
    #                  [210, 580.],
    #                  [1050, 580.],
    #                  [690, 400]])

    # dst = np.float32(   [[ 200. ,   0.],
    #                      [ 200. , 600.],
    #                      [ 1080. , 600.],
    #                      [ 1080. ,   0.]])

    src= np.float32([[540, 400],
                 [210, 580.],
                 [1050, 580.],
                 [690, 400]])

    dst = np.float32(   [[ 1800. ,   0.],
                         [ 1800. , 1000.],
                         [ 2600. , 1000.],
                         [ 2600. ,   0.]])

    M = cv2.getPerspectiveTransform(src, dst);
    Minv = cv2.getPerspectiveTransform(dst, src);
    # print(M);
    crop = imgcv[300:650,500:,:]
    resized = cv2.resize(crop,(448,448))
    batch = np.array([resized[:,:,0],resized[:,:,1],resized[:,:,2]])
    batch = 2*(batch/255.) - 1
    batch = np.expand_dims(batch, axis=0)
    out = model.predict(batch)
    boxes = yolo_net_out_to_car_boxes(out[0], threshold = 0.17)
    return draw_box(boxes,imgcv,[[500,1280],[300,650]],M)