from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications import VGG16

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
import os
import cv2
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  #disables GPU 
import matplotlib.pyplot as plt
import numpy as np
#tf.__version__
from tensorflow.python.client import device_lib

from sklearn import preprocessing

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image


class CNNFeatureExtractor:

    def __init__(self, cnn = 'VGG16', weights = 'imagenet'):

        self.model = VGG16(weights='imagenet',
                    include_top=True,
                    input_shape=(224, 224, 3))
        

    def load_img(self, img_path, img_shape, preprocess=True):
        img_shape=[224,224]
        img_rows=img_shape[0]
        img_cols=img_shape[1]
        #num_channel=img_shape[2]


        img = image.load_img(img_path , target_size=(img_rows, img_cols))    
        if preprocess:
            img = image.img_to_array(img) 
            img = np.expand_dims(img, axis=0)

            img =  preprocess_input(img)
            return img

        #data = np.array(data)
        #data = data.astype('float32')
        #data /= 255
        #labels=np.array(labels)
        #print('data shape',data.shape)
        #print('labels shape',labels.shape)
        return np.array(img)

    def lap_mag(self , channels):
        laps = []
        #print(channels.shape)
        for i in range(channels.shape[-1]):
            
            dst = cv2.Laplacian(channels[:,:,i], cv2.CV_32F , 3)
            laps.append(dst)
        mag = np.linalg.norm(laps, axis = 0)
        return mag

    def get_cnn_out(self, input, layer_num):
        get_layer_output = K.function([self.model.input], [l.output for l in self.model.layers][layer_num])
        layer_out = np.array(get_layer_output(input))
        return layer_out[0]




    def local_arg_max(self, mat, window_size):
        #Use a moving window to find local max/min in section. Determine coordinate of max pixel in image.
        idx = []

        k = int(np.floor(window_size/2))
        #print(k)
        for i in range(k, mat.shape[0]-k, window_size):
            for j in range(k, mat.shape[1]-k, window_size):

                window = mat[i-k:i+k+1, j-k:j+k+1]
                coords = np.argwhere(window==window.max())
                
                idx.extend(coords + [i-k, j-k])

        return  np.unique(idx, axis=0)  

    def get_keypoints_and_descriptors(self, image_path):

        img = self.load_img(image_path, [224,224])
        img_no_proc = self.load_img(image_path, [224,224], preprocess=False)
        #keypoint_coords = np.array([])



        layer_1 = self.get_cnn_out(img, 1)  
        layer_2 =  self.get_cnn_out(img, 2)
        block_1 = (self.lap_mag(layer_1) +  self.lap_mag(layer_2)) /2
        layer_output = np.stack([layer_1, layer_2])

        keypoint_coords = self.local_arg_max(block_1, 3)
        del layer_1, layer_2, block_1

        k=[]
        d=[]
        gray = cv2.cvtColor(img_no_proc,cv2.COLOR_BGR2GRAY)
        r = cv2.cornerHarris(gray,5,5,0.04, cv2.BORDER_ISOLATED)
        r = cv2.dilate(r,None)
        for row, col in keypoint_coords:
            #print(float(row), float(col))
            if r[row,col]  > 0.01*np.amax(r) : #if pixel is not on edge

                keypoint = cv2.KeyPoint()
                keypoint.pt = (float(col), float(row))
                keypoint.octave = 0
                keypoint.size = 0
                keypoint.response = 0
                k.append(keypoint)
                d.append(np.array(layer_output[:, row, col ]).flatten())

        return np.array(k), np.array(d)