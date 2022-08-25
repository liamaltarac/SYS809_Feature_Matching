import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.image import flip_up_down, flip_left_right, rot90
from tensorflow.linalg import normalize

from tensorflow.compat.v1 import extract_image_patches
import numpy as np

from keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from keras.models import Sequential, load_model

import matplotlib.pyplot as plt
import cv2
from skimage.transform import warp_polar, rotate, rescale

import gc
class SADecompLayer(Layer):
    def __init__(self, w_size=8,  **kwargs):
        super(SADecompLayer, self).__init__( **kwargs)
        self.k = w_size
    
    @tf.function
    def call(self, inputs):
        inputs_shape = tf.shape(inputs)  

        #print("Input shape : ", inputs.shape)
        #print("k", self.k)

        #batch_size, height, width, n_filters = inputs_shape[0], inputs_shape[1], inputs_shape[2], inputs_shape[3]

        #exp_data  = tf.expand_dims(inputs[0], 0)


        patches = self.extract_patches(inputs) #, [1, self.k, self.k, 1],  [1, self.k, self.k, 1], rates = [1,1,1,1] , padding = 'VALID')
        print(patches.shape)
        #patches = tf.reshape(patches, [1, self.k , self.k, 1])
        #patches = tf.make_ndarray(patches.numpy())

        r = np.sqrt(((patches.shape[1]/2.0)**2.0)+((patches.shape[2]/2.0)**2.0))
        out =  warp_polar(patches, radius=r, scaling="log")
        print(out.shape)
        out  = tf.convert_to_tensor(out)

        out = self.extract_patches_inverse(inputs, out)
        #print("PS : ", patches.shape)
        '''for patch in patches:
            print(patch.shape)
            plt.figure()
            plt.imshow(patch)'''


        return out #tf.concat([sym, anti], -1) #, inputs - sym # tf.reshape((sum + mat_sum_rot_90) / 8, (batch_size, height, width, n_filters))


    def extract_patches(self, x):
        patches =  extract_image_patches(x, [1, self.k, self.k, 1],  [1, self.k, self.k, 1],  rates = [1,1,1,1] , padding = 'SAME')
        return tf.reshape(patches, [-1, self.k, self.k, x.shape[-1]])
    # Thanks to https://stackoverflow.com/questions/44047753/reconstructing-an-image-after-using-extract-image-patches
    def extract_patches_inverse(self, x, y):
        _x = tf.zeros_like(x)
        _y = self.extract_patches(_x)
        grad = tf.gradients(_y, _x)[0]
        # Divide by grad, to "average" together the overlapping patches
        # otherwise they would simply sum up
        return tf.gradients(_y, _x, grad_ys=y)[0] / grad




if __name__ == "__main__":
    model = Sequential()
    '''model.add(Conv2D(input_shape=(16, 16, 1), filters=64, kernel_size=(3, 3),
                    padding='same', activation='relu'))'''
    
    model.add(SADecompLayer( w_size=4, input_shape=(16, 16, 1), name='sad2_1'))
    

    # Call model on a test input
    x =np.array([np.arange(16*16).reshape([16,16])])
    print(x.shape)

    y = model(x)

    print(y)