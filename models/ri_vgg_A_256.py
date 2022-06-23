
from tabnanny import verbose
from sa_decomp_layer import SADecompLayer

#source : https://github.com/ashushekar/VGG16

import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.compat.v1.keras.backend import set_session


def ri_vgg():
    # Generate the model
    model = Sequential()

    #BLOCK 1   (in 240)
    # Layer 1: Convolutional
    model.add(Conv2D(input_shape=(240, 240, 3), filters=64, kernel_size=(3, 3),
                    padding='same', activation='relu'))
    # Layer 2: Convolutional
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 3: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))


    # BLOCK 2 (in 120)
    # Layer 4: Convolutional
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 5: Convolutional
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 6: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(SADecompLayer( w_size=8, name='sad2_1'))

    # BLOCK 3 (in 60)
    # Layer 7: Convolutional
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 8: Convolutional
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 9: Convolutional
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 10: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(SADecompLayer( w_size=3, name='sad3_1'))

    # BLOCK 4 (in 30)
    # Layer 11: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 12: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 13: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 14: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(SADecompLayer( w_size=3, name='sad4_1'))


    # BLOCK 5 (in 15)
    # Layer 15: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 16: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 17: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 18: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 19: Flatten
    model.add(Flatten())
    # Layer 20: Fully Connected Layer
    model.add(Dense(units=4096, activation='relu'))
    # Layer 21: Fully Connected Layer
    model.add(Dense(units=4096, activation='relu'))
    # Layer 22: Softmax Layer
    model.add(Dense(units=256, activation='softmax'))

    return model

if __name__ == "__main__":

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)

    trdata = ImageDataGenerator()
    traindata = trdata.flow_from_directory(directory="Caltech256/train/",target_size=(240,240))
    tsdata = ImageDataGenerator()
    testdata = tsdata.flow_from_directory(directory="Caltech256/val/", target_size=(240,240))
    
    print("Got Data")


    # Generate the model
    model = Sequential()
    print("S")
    #BLOCK 1   (in 240)
    # Layer 1: Convolutional
    model.add(Conv2D(input_shape=(240, 240, 3), filters=64, kernel_size=(3, 3),
                    padding='same', activation='relu'))
    print("1")

    # Layer 2: Convolutional
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    print("2")

    # Layer 3: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    print("3")


    # BLOCK 2 (in 120)
    # Layer 4: Convolutional
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    print("4")

    # Layer 5: Convolutional
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    print("5")

    # Layer 6: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    print("6")
    
    model.add(SADecompLayer( w_size=8, name='sad2_1'))
    print("7")

    # BLOCK 3 (in 60)
    # Layer 7: Convolutional
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 8: Convolutional
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 9: Convolutional
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 10: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(SADecompLayer( w_size=3, name='sad3_1'))

    # BLOCK 4 (in 30)
    # Layer 11: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 12: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 13: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 14: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(SADecompLayer( w_size=3, name='sad4_1'))


    # BLOCK 5 (in 15)
    # Layer 15: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 16: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 17: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 18: MaxPooling
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 19: Flatten
    model.add(Flatten())
    # Layer 20: Fully Connected Layer
    model.add(Dense(units=4096, activation='relu'))
    # Layer 21: Fully Connected Layer
    model.add(Dense(units=4096, activation='relu'))
    # Layer 22: Softmax Layer
    model.add(Dense(units=256, activation='softmax'))
    
    print("Got model")
    


    # Add Optimizer
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"], run_eagerly=True)
    '''model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy,
                metrics=['accuracy'], run_eagerly=True)'''
    # Check model summary
    print(model.summary())

    checkpoint = ModelCheckpoint("ri_vgg_A_256.h5", monitor='val_acc', 
                             verbose=2, save_best_only=True, 
                             save_weights_only=False, mode='auto', period=1)

    earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=2, mode='auto')
    print("HERE")
    hist = model.fit_generator(steps_per_epoch=100, generator=traindata, validation_data=testdata,
                           validation_steps=10, epochs=100,
                           callbacks=[checkpoint, earlystop], verbose=2)


    plt.plot(hist.history["acc"])
    plt.plot(hist.history['val_acc'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
    plt.show(block=True)
