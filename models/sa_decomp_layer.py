import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.image import flip_up_down, flip_left_right, rot90
import numpy as np

from keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from keras.models import Sequential, load_model

class SADecompLayer(Layer):
    def __init__(self, w_size=8,  **kwargs):
        super(SADecompLayer, self).__init__( **kwargs)
        self.k = w_size

    #@tf.function
    def call(self, inputs):
        mat = inputs
        print("MS : ", mat.shape)
        mat_unpacked = tf.unstack(mat[0,:,:]) 
        #mat = np.array(inputs)
        #mat = list(mat)
        sym = [] #np.empty(mat.shape)
        #print(sym.shape[-2:])
        
        for n, m in enumerate(mat_unpacked):
            #out = np.empty(mat.shape[-2:])
            temp = tf.Variable(np.zeros(mat.shape[1:3]), dtype=tf.float32)
            for i in range(0, m.shape[0], self.k):
                for j in range(0, m.shape[1], self.k):               
                    w = [m[i:i+self.k, j:j+self.k]]
                    #w = w.numpy()
                
                    #print(i,j)

                    mat_flip_x = flip_left_right(w)

                    mat_flip_y = flip_up_down(w)
                    mat_flip_xy = flip_left_right(flip_up_down(w))
                    sum = w + mat_flip_x + mat_flip_y + mat_flip_xy

                    mat_sum_rot_90 = rot90(sum)
                    #print((sum + mat_sum_rot_90) / 8)
                    #sym = (sum + mat_sum_rot_90) / 8
                    #anti_sym = mat - sym
                    temp = temp[i:i+self.k, j:j+self.k].assign(((sum + mat_sum_rot_90) / 8))
            sym.append(temp)

        return tf.stack(sym) #, anti_sym

        #return inputs * self.scale

if __name__ == "__main__":
    model = Sequential()
    model.add(Conv2D(input_shape=(16, 16, 1), filters=64, kernel_size=(3, 3),
                    padding='same', activation='relu'))
    
    model.add(SADecompLayer( w_size=8, name='sad2_1'))
    

    # Call model on a test input
    x =np.array([np.arange(16*16).reshape([16,16])])
    print(x.shape)

    y = model(x)

    print(y)