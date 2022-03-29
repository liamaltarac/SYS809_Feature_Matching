from pprint import pprint
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
from sklearn.decomposition import PCA

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

    '''def lap_mag(self , channels):
        laps = []
        #print(channels.shape)
        for i in range(channels.shape[-1]):
            
            dst = cv2.Laplacian(channels[:,:,i], cv2.CV_32F , 3)
            laps.append(dst)
        mag = np.linalg.norm(laps, axis = 0)
        return mag'''

    def get_cnn_out(self, input, layer_num):
        get_layer_output = K.function([self.model.input], [l.output for l in self.model.layers][layer_num])
        layer_out = np.array(get_layer_output(input))
        return layer_out[0]




    '''def local_arg_max(self, mat, window_size):
        #Use a moving window to find local max/min in section. Determine coordinate of max pixel in image.
        idx = []

        k = int(np.floor(window_size/2))
        #print(k)
        for i in range(k, mat.shape[0]-k, window_size):
            for j in range(k, mat.shape[1]-k, window_size):

                window = mat[i-k:i+k+1, j-k:j+k+1]
                coords = np.argwhere(window==window.max())
                
                idx.extend(coords + [i-k, j-k])

        return  np.unique(idx, axis=0)  '''

    '''def get_keypoints_and_descriptors(self, image_path):

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

        return np.array(k), np.array(d)'''

    def get_keypoints_and_descriptors(self, image_path):

        img = self.load_img(image_path, [224,224])
        img_no_proc = self.load_img(image_path, [224,224], preprocess=False)
        #keypoint_coords = np.array([])

        pool_layers = [3,4]

        layer_pool_1 = np.stack(self.get_cnn_out(img, 3))
        layer_pool_2 = np.stack(self.get_cnn_out(img, 6)) 
        layer_pool_3 = np.stack(self.get_cnn_out(img, 10)) 
        layer_pool_4 = np.stack(self.get_cnn_out(img, 14))
        layer_pool_5 = np.stack(self.get_cnn_out(img, 18))

        layers = [layer_pool_1,layer_pool_2,layer_pool_3,layer_pool_4,layer_pool_5]

        k=[]
        d = []
        kp_desc = dict()
        #Use a moving window to find local max/min in section. Determine coordinate of max pixel in image.
        idx = []

        #for l in pool_layers:
            #d_temp = []
        for i in range(3, img[0].shape[0]-4, 8):
            for j in range(3, img[0].shape[1]-4, 8):
                #window = img[i-k:i+k+1, j-k:j+k+1]
                #coords = np.argwhere(window==window.max())
                row = i# ((i+.5)) * (2**l) 
                col = j #((j+.5)) * (2**l) 
                #d = layers[l-1][ i, j, : ]
                keypoint = cv2.KeyPoint()
                keypoint.pt = (float(col), float(row))
                keypoint.octave = 0
                keypoint.size = 0
                keypoint.response = 0
                k.append(keypoint)
                #print(np.floor(i/8), np.floor(j/8))
                d_vec_3 = layer_pool_3[ int(np.floor(i/8)), int(np.floor(j/8)), : ].flatten()
                #d_vec_3 = np.abs(d_vec_3)
                d_vec_3 *= 1/d_vec_3.std()
                d_vec_3 *= 1.414
                d_vec_4 = layer_pool_4[ int(np.floor(i/16)), int(np.floor(j/16)), : ].flatten()
                #d_vec_4 = np.abs(d_vec_4)
                d_vec_4 *= 1.0/d_vec_4.std()

                d_vec_5 = layer_pool_5[ int(np.floor(i/32)), int(np.floor(j/32)), : ].flatten()
                #d_vec_5 = np.abs(d_vec_5)
                d_vec_5 *= 1/d_vec_5.std()

                d.append(np.concatenate([d_vec_3 , d_vec_4, d_vec_5]))
                #d[-1] = np.abs(d[-1])
                #d[-1] *= 1.0/d[-1].max()

            #print(d[0])
            #d_temp = np.array(d_temp,  dtype=object)
        #pca = PCA(n_components=300)
        #d = pca.fit_transform(d)
        d = np.array(d, dtype=np.float32)
        #print(d)
        #print(d.shape)

        return np.array(k), d