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
        

    def load_img(self, img_path, preprocess=True):
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

    def _cnn_lap(self, image, layer_name):
        min_max_scaler = preprocessing.MinMaxScaler()
        layer = [l for l in self.model.layers if layer_name == l.name][0]
        #print(layers)
        mag = []
        if type(layer.output_shape) is list:
            num_channels = layer.output_shape[0][-1]
        else:
            num_channels = layer.output_shape[-1]
        #print(num_channels)

        imgs = []
        get_layer_output = K.function([self.model.input], layer.output)
        layer_out = np.array(get_layer_output(image))
        for i in range(0, num_channels):
            #plt.imshow(layer_out[0][:,:,2])
            imgs.append(layer_out[0][:,:,i])

        laps = []
        for i in range(0, num_channels):
            out = imgs[i]
            dst = cv2.Laplacian(out, cv2.CV_32F , 7)
            laps.append(dst)

        mag.append(np.linalg.norm(np.abs(laps), axis = 0))
        #mag.append(np.sum(laps, axis = 0))
        #scaled_mag = min_max_scaler.fit_transform(mag[0])

        #mag /= mag.max()/255.0' ''
        return mag[0] * 1.0/mag[0].max()  , imgs, np.array(laps)
        #return dst

    def _local_arg_exrtrema_2(self, mat1, mat2):

        #Use a moving window to find local max/min in section. Determine coordinate of max pixel in image.
        idx = []
        thresh = np.abs(np.amax(mat2) - np.abs(np.amin(mat2)))
        #print("Thresh = ", thresh)

        r = cv2.cornerHarris(mat1,5,5,0.04, cv2.BORDER_ISOLATED)
        r = cv2.dilate(r,None)
        for i in range(1, mat1.shape[0]-1):
            for j in range(1, mat1.shape[1]-1):
                #if r[i,j]  > 0.01*np.amax(r) : #if pixel is not on edge
                pixel_of_interest = mat2[i,j]
                #if pixel_of_interest>0.15:

                neighbours = mat2[i-1:i+2, j-1:j+2]
                neighbours[1,1] = np.NaN
                neighbours_below = mat1[i-1:i+2, j-1:j+2]
                if (pixel_of_interest > np.nanmax(neighbours) and pixel_of_interest > np.nanmax(neighbours_below)) or (pixel_of_interest < np.nanmin(neighbours) and pixel_of_interest < np.nanmin(neighbours_below)):
                    idx.append(np.array([j,i]))

        return np.unique(idx, axis=0)

    def _getDescriptors(self, layer_output, coords):
        #fig2 = plt.figure(figsize=(30, 30))

        layers = []
        '''for i in range(len(layer_output)):
            layer_output[i] = cv2.resize(layer_output[i], (224, 224), interpolation = cv2.INTER_LINEAR )'''
        layer_output = np.stack(layer_output)
        '''for i in range(0,64):
            layers.append(layer_output[:,:,i])'''
        #layers = np.stack(layers)

        descriptors = []
        for col, row in  coords:
            #d_vec = np.array(layer_output[:,row, col ]).flatten()
            d_vec = np.array(layer_output[:,row-1:row+2, col-1:col+2 ]).flatten()
            d_vec = np.abs(d_vec)
            d_vec *= 1.0/d_vec.max()
            descriptors.append(d_vec)
        return descriptors

    def get_keypoints_and_descriptors(self, img):
        keypoint_coords = []

        mag_1,_,_ = self._cnn_lap(img, layer_name=self.model.layers[0].name)
        mag_2 ,layer_out,lap_out_2 = self._cnn_lap(img, layer_name=self.model.layers[1].name)
        mag_3,out, lap_out_3 = self._cnn_lap(img, layer_name=self.model.layers[2].name)
        #_,layer_out,_ = cnn_lap(img, layer_name=model.layers[7].name)
        
        keypoint_coords = np.unique(self._local_arg_exrtrema_2(mag_1, mag_2), axis=0)

        desc = self._getDescriptors(np.concatenate([lap_out_2, lap_out_3]), keypoint_coords)

        k=[]
        for row, col in keypoint_coords:
            #print(float(row), float(col))
            keypoint = cv2.KeyPoint()
            keypoint.pt = (float(row), float(col))
            keypoint.octave = 0
            keypoint.size = 0
            keypoint.response = 0
            k.append(keypoint)


        return np.array(k), np.array(desc)