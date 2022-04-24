import numpy as np 
import os
import PIL
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras import backend as keras

from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Input,
    MaxPooling2D, concatenate)

def identity_function(x):
    return x

def get_model_definition(self, image_index, data_x, data_plate_y):
    inputs = Input((height, width,data_x))
    processing= Lambda(lambda x:x/388)(inputs)
    
c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(processing)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(drop_p)(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c1)
    c1 = BatchNormalization()(c1)

    c2 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(drop_p)(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c2)
    c2 = BatchNormalization()(c2)

    c3 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(drop_p)(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c3)
    c3 = BatchNormalization()(c3)

    c4 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(drop_p)(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c4)
    c4 = BatchNormalization()(c4)

    c5 = MaxPooling2D(pool_size=(2, 2))(c4)
    c5 = Conv2D(1024, (3, 3), activation='relu',
                kernel_initializer='he_normal',
                padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(drop_p)(c5)
    c5 = Conv2D(1024, (3, 3), activation='relu',
                kernel_initializer='he_normal',
                padding='same')(c5)
    c5 = BatchNormalization()(c5)

    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(drop_p)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c6)
    c6 = BatchNormalization()(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(drop_p)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c7)
    c7 = BatchNormalization()(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(drop_p)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c8)
    c8 = BatchNormalization()(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(drop_p)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c9)
    c9 = BatchNormalization()(c9)

    outputs = Conv2D(data_plate_y, (1, 1), activation='sigmoid')(c9)
    
    
    model = Model(input = inputs, output = conv10)
     model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    process_input = identity_function
    return model, process_input


