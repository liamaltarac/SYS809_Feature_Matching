
from tabnanny import verbose
#from sa_decomp_layer import SADecompLayer

#source : https://github.com/ashushekar/VGG16

import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, BatchNormalization, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.compat.v1.keras.backend import set_session




if __name__ == "__main__":

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)

    trdata = ImageDataGenerator( rescale=1./255)
    traindata = trdata.flow_from_directory(directory="Caltech256/train/",target_size=(240,240), batch_size=32)
    print(traindata.batch_size)
    tsdata = ImageDataGenerator( rescale=1./255)
    testdata = tsdata.flow_from_directory(directory="Caltech256/val/", target_size=(240,240), batch_size=32)
    
    print("Got Data")


    # Generate the model
    model = keras.Sequential()
    model.add(Conv2D(filters=96, kernel_size=(11, 11), 
                            strides=(4, 4), activation="relu", 
                            input_shape=(240, 240, 3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3), strides= (2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), 
                            strides=(1, 1), activation="relu", 
                            padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), 
                            strides=(1, 1), activation="relu", 
                            padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=384, kernel_size=(3, 3), 
                            strides=(1, 1), activation="relu", 
                            padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3, 3), 
                            strides=(1, 1), activation="relu", 
                            padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="softmax"))
    model.compile(loss='categorical_crossentropy', 
                optimizer=tf.optimizers.SGD(lr=0.001), 
                metrics=['accuracy'])
    

    print(model.summary())

    checkpoint = ModelCheckpoint("ri_vgg_A_256.h5", monitor='val_accuracy', 
                             verbose=2, save_best_only=True, 
                             save_weights_only=False, mode='auto', period=1)

    earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=2, mode='auto')
    print("HERE")
    hist = model.fit_generator(steps_per_epoch=100, generator=traindata, validation_data=testdata,
                           validation_steps=10, epochs=100,
                           callbacks=[earlystop], verbose=2)


    plt.plot(hist.history["acc"])
    plt.plot(hist.history['val_acc'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
    plt.show(block=True)
