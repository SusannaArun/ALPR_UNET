# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 12:07:13 2022

@author: bentw
"""

import os
import numpy as np
import PIL
import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import datetime, os
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import random
from shapely.geometry import Point
from shapely.geometry import Polygon

from datetime import date, datetime

#tf.compat.v1.disable_eager_execution()

today = date.today()
now = datetime.now()

data_path = r'C:/RodoSol-ALPR/images/cars-br'
IMG_SIZE = 512
BUFFER_SIZE = 4
BATCH_SIZE = 4
SEED = 42
N_CHANNELS = 3
N_CLASSES = 2
#TODO: Grayscale <-may not be needed
#TODO: Normalize<-done
#TODO: shuffle <-done
#TODO: Resolve labels <-done
    #TODO: new label parser <- done
    #TODO: new image cutter <- done

class DataLoader:
    '''
    indexing for labels is as follows:
        x, y, width, hieght
        
        so data_x[index][y][x][channel] gives you the top left corner
    
    '''
    
    global_set = []
    
    TRAINING = .7
    TESTING = .2
    VAL = .1
    data = []
    data_x = []
    data_y = []
    data_plate_y = []
    data_plate_text_y = []
    data_dir_train = "/data_train/"
    data_dir_val = "/data_val/"
    data_dir_test = "/data_test/"
    label_dir_train = "/label_train/"
    label_dir_val = "/label_val/"
    label_dir_test = "/label_test/"
    INPUT_SIZE = -1#264
    OUTPUT_SIZE = -1
    RATIO = -1 #cast to int when using ratio
    HEIGHT = 720
    WIDTH = 1280
    dataset_test = None
    dataset_train = None
    dataset_val = None

    def __init__(self, root = r'C:\data subset', input_size = 512, output_size = 512, val_split = .1, grayscale = True):
        self.INPUT_SIZE = input_size
        self.OUTPUT_SIZE = output_size
        self.RATIO = self.OUTPUT_SIZE/self.INPUT_SIZE
        self.data_x = self.__get_filepath(r'C:/RodoSol-ALPR/images/cars-br') 
        self.data_y = self.__get_filepath(r'C:\RodoSol-ALPR\images\cars-br-label')
        self.data_plate_y, self.data_plate_text_y, = self.__parse_y(self.data_y, 
                                                                    self.data_plate_y, 
                                                                    self.data_plate_text_y)
        

        if os.path.isdir(self.data_dir_train)==False:
            os.mkdir(self.data_dir_train)
        if os.path.isdir(self.data_dir_val)==False:
            os.mkdir(self.data_dir_val)
        if os.path.isdir(self.data_dir_test)==False:
            os.mkdir(self.data_dir_test)
                
        if os.path.isdir(self.label_dir_train)==False:
            os.mkdir(self.label_dir_train)
        if os.path.isdir(self.label_dir_val)==False:
            os.mkdir(self.label_dir_val)
        if os.path.isdir(self.label_dir_test)==False:
            os.mkdir(self.label_dir_test)
        self.__crop_images(self.data_x, self.data_plate_y)
        print('debug stop point')
        self.dataset_train = tf.data.Dataset.list_files(self.data_dir_train + "*.png", seed = SEED)
        self.dataset_train = self.dataset_train.map(self.parse_image)
        self.dataset_val = tf.data.Dataset.list_files(self.data_dir_val + "*.png")
        self.dataset_val = self.dataset_val.map(self.parse_image)        
        self.dataset_test = tf.data.Dataset.list_files(self.data_dir_test + "*.png")        
        self.dataset_test = self.dataset_test.map(self.parse_image)
        
        #self.dataset = tf.keras.utils.image_dataset_from_directory(directory = self.data_dir, 
        #                                                           labels = None,
        #                                                           label_mode = None,
        #                                                           
        #                                                           batch_size = 32,
        #                                                           image_size = (self.INPUT_SIZE, self.INPUT_SIZE))
        #will yeild a tensor for shape (32, input_size, input_size, channels)
        print('here')

    def parse_image(self, img_path: str) -> dict:
        """Load an image and its annotation (mask) and returning
        a dictionary.
        
        Parameters
        ----------
        img_path : str
        Image (not the mask) location.

        Returns
        -------
        dict
        Dictionary mapping an image and its annotation.
        """
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        # For one Image path:
            # .../trainset/images/training/ADE_train_00000001.jpg
            # Its corresponding annotation path is:
                # .../trainset/annotations/training/ADE_train_00000001.png
        mask_path = tf.strings.regex_replace(img_path, "data", "label")

        mask = tf.io.read_file(mask_path)
        # The masks contain a class index for each pixels
        mask = tf.image.decode_png(mask, channels=1)
        # In scene parsing, "not labeled" = 255
        # But it will mess up with our N_CLASS = 150
        # Since 255 means the 255th class
        # Which doesn't exist
        mask = tf.where(mask == 255, np.dtype('uint8').type(1), mask)
        # Note that we have to convert the new value (0)
        # With the same dtype than the tensor itself

        return {'image': image, 'segmentation_mask': mask}
#Y is sorted. Do not shuffle until tf.dataset has been created
#this also rectifies the labels by the resizing factor of the shrunken image
    def __parse_y(self, data_y, data_plate_y, data_plate_text_y):
        
        plate_text = 1
        plate_pos = 3
        
        for x in range(0, len(data_y)):
            f = open(data_y[x])
            lines = f.readlines()
            data_plate_y.append(lines[plate_pos][8:])
            data_plate_y[x] = data_plate_y[x].split(',') #parsing happens here
            buffer = []
            for datum in data_plate_y[x]:
                datum = datum.split()
                for y in range(len(datum)):
                    
                    datum[y] = int(datum[y])
                    buffer.append(datum[y])
            data_plate_y[x] = np.array(buffer)
            data_plate_text_y.append(lines[plate_text][6:])
            
            f.close()
        
            
        return np.array(data_plate_y), np.array(data_plate_text_y)

    
    #this method requires a lot of memory. just don't freak out if you run out of mem. also it takes forever. sorry its just a lot of data
    
        
        
    
    #needs testing <- works as far as I can tell
    
    def __check_bounds(self, left, top, right, bottom, plate):
        '''Makes sure that we don't crop out the plate or outside of the image'''    
        if(left>plate[0]):
            left = plate[0]
            right = left+self.INPUT_SIZE
        
        if(right<(max(plate[2], plate[4]))):
            right = (max(plate[2], plate[4]))
            left = right-self.INPUT_SIZE
        
        if(top>plate[1]):
            top = plate[1]
            bottom = top+self.INPUT_SIZE
        
        if(bottom<(max(plate[7], plate[5]))):
            bottom = plate[1]+plate[3]
            top = bottom - self.INPUT_SIZE
            
        if(left<0):
            left = 0
            right = self.INPUT_SIZE
        
        if(right>self.WIDTH):
            right = self.WIDTH
            left = right-self.INPUT_SIZE
        
        if(top<0):
            top = 0
            bottom = self.INPUT_SIZE
        
        if(bottom>self.HEIGHT):
            bottom = self.HEIGHT
            top = bottom-self.INPUT_SIZE
            
        return left, top, right, bottom
    
    def __adjust_label(self, plate, left, top):
        
        buffer = plate
        for x in range(0, len(buffer)):
            if (x%2==0):
                buffer[x] = buffer[x]-left
            else:
                buffer[x] = buffer[x]-top
        
        return buffer
    
    def __crop_and_save(self, image, index, left, top, right, bottom, plate, normalize):
        #left, top, right, bottom = self.__check_bounds(left, top, right, bottom, plate)
        crop = image.crop((left, top, right, bottom))
        if(normalize):
            crop = np.array(crop, dtype = np.float32)
            crop = crop/255
            crop = PIL.Image.fromarray(crop.astype(np.uint8))
        #crop.show()
        
        train_size = int(5000*self.TRAINING)
        val_size = int(5000*self.VAL)
        test_size = int(5000*self.TESTING)
        directory = None
        
        randomizer = index%10
        
        
        if(randomizer in range(0, 7)):
            directory = self.data_dir_train
        elif(randomizer in range(7, 8)):
            directory = self.data_dir_val
        elif(randomizer in range(8, 10)):
            directory = self.data_dir_test
        
        crop.save(directory+str(index)+'.png')
        return crop
    
    def __adjust_input_to_output(self, bbox, left, top): 
        
        self.__adjust_label(bbox, left, top)
        for x in range(len(bbox)):
            bbox[x]= int(bbox[x]*self.RATIO)
        return bbox
    
    
    #TEST THE MASK CREATION AND MAKE THE LABELS IN THE CROP IMAGES FUNCTION
    def __make_seg_mask(self, x, data_plate_y, left, top):
        index = x%(len(data_plate_y))
        buffer = np.copy(data_plate_y[index])
        mask = np.zeros((self.OUTPUT_SIZE, self.OUTPUT_SIZE), dtype = np.float32)
        buffer = self.__adjust_input_to_output(buffer, left, top)
        polygon = Polygon([(buffer[0], buffer[1]), (buffer[2], buffer[3]), (buffer[4], buffer[5]), (buffer[6], buffer[7])])
        
        minx = min(buffer[0], buffer[2], buffer[4], buffer[6])
        maxx = max(buffer[0], buffer[2], buffer[4], buffer[6])
        miny = min(buffer[1], buffer[3], buffer[5], buffer[7])
        maxy = max(buffer[1], buffer[3], buffer[5], buffer[7])
        
        for x in range(minx-1, maxx+1):
            for y in range(miny-1, maxy+1):
            #buffer[x] = self.__adjust_label(buffer[x], left, top)
            #buffer[x] = self.__adjust_input_to_output(buffer[x])
                point = Point(x, y)
                if(polygon.contains(point)):
                    mask[y][x] = 255
            
        return mask
    

    
    def __masks_only(self, data_plate_y):
        output_labels = []
        for x in range(len(data_plate_y)):
            counter = 0
            
            offset = random.randint(30, int(self.INPUT_SIZE/1.5))
            local_plate = np.copy(data_plate_y[x])
            
            left = local_plate[0]-offset # relative to plate y
            right = left+self.INPUT_SIZE
            top = local_plate[1]-offset
            bottom = top+self.INPUT_SIZE
            label_buffer = []
            #anchor around top left corner
            
            left, top, right, bottom = self.__check_bounds(left, top, right, bottom, local_plate)
            adjusted_label = self.__make_seg_mask(x, data_plate_y, left, top)
            self.__save_segment_label(adjusted_label, x, counter)
            output_labels.append(adjusted_label)
            counter+=1
                      
            
            
        return np.array(output_labels, dtype = np.uint8)
        
    def __crop_images(self, data_x, data_plate_y, normalize = False):
    
        '''
        If the images aren't already preprocessed, that happens here. Cropped images and their corresponding segmentation
        mask are saved here in a file in the root directory called 'data. it makes and saves 18000 images, so this
        takes a long time to run the first time. 
        '''
        #output_labels = []
        for x in range(0, len(data_x)):
            
            '''
            local_plate[0], local_plate[1] -------------------local_plate[2], local_plate[3]
            |                                                                               |
            |                                                                               |
            |                                License Plate                                  |
            |                                                                               |
            |                                                                               |
            local_plate[6], local_palte[7]--------------------local_plate[4], local_plate[5]
            
            
            
            
            '''
            index = x%(len(data_x))
            image = PIL.Image.open(data_x[index])
            
            counter = 0
            offset = random.randint(30, int(self.INPUT_SIZE/1.5))
            local_plate = np.copy(data_plate_y[index])
            
            left = local_plate[0]-offset # relative to plate y
            right = left+self.INPUT_SIZE
            top = local_plate[1]-offset
            bottom = top+self.INPUT_SIZE
            
        
            #anchor around top left corner
            
            left, top, right, bottom = self.__check_bounds(left, top, right, bottom, local_plate)
            crop1 = self.__crop_and_save(image, x, left, top, right, bottom, local_plate, normalize)
            #adjusted_label = self.__adjust_label(local_plate, left, top)
            adjusted_label = self.__make_seg_mask(index, data_plate_y, left, top)
            self.__save_segment_label(adjusted_label, x)
            #crop1 = self.__show_image_plate_granular(adjusted_label, crop1, index, counter)
            #output_labels.append(adjusted_label)
            counter+=1
            #crop1.show()
        
        
           
            if(index%200==0):
                print(x)
            
            
            
            image.close()
        
        
        
    def __show_image_plate_granular(self, label, image, index, counter):
        image = np.array(image)
        image[label[1]:(label[1]+label[3]), label[0]:(label[0]+label[2]), 1] = 0
        image = PIL.Image.fromarray(image.astype(np.uint8))
        image.save(r'/data/'+str(index)+'-'+str(counter)+'labeled'+'.png')
        return image
    
    def __save_segment_label(self, image, index):
        train_size = int(5000*self.TRAINING)
        val_size = int(5000*self.VAL)
        test_size = int(5000*self.TESTING)
        directory = None
        
        randomizer = index%10
        
        if(randomizer in range(0, 7)):
            directory = self.label_dir_train
        elif(randomizer in range(7, 8)):
            directory = self.label_dir_val
        elif(randomizer in range(8, 10)):
            directory = self.label_dir_test
            
        image = np.array(image)
        image = PIL.Image.fromarray(image.astype(np.uint8))
        image.save(directory+str(index)+'.png')
    
    def __array_to_image(self, image_index, data_x):
        image = data_x[image_index]*255
        image = PIL.Image.fromarray(image.astype(np.uint8))
        return image
    
    def show_image_base(self, image_index, data_x):
        image = self.__array_to_image(image_index, data_x)
        image.show()
    
    def show_image_plate_box(self, image_index, data_x, data_plate_y):
        image = data_x[image_index]
        anchor_x = data_plate_y[image_index][0]
        width = anchor_x+data_plate_y[image_index][2]
        anchor_y = data_plate_y[image_index][1]
        height = anchor_y + data_plate_y[image_index][3]
        
        image[anchor_y:height, anchor_x:width, 1] = 0
        image = self.__array_to_image(image_index, data_x)
        image.show()
        

    def __get_filepath(self, root):        
        #this is a really convoluted way to get the file paths of each image and label
        directory = os.listdir(root)
        for x in range(len(directory)):
            directory[x] = os.path.join(root, directory[x])
            
        return sorted(directory)



@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].
    input_mask : tf.Tensor
        Tensorflow tensor containing an annotation of size [SIZE,SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its annotation.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

@tf.function
def load_image_test(datapoint: dict) -> tuple:
    """Normalize and resize a test image and its annotation.

    Notes
    -----
    Since this is for the test set, we don't need to apply
    any data augmentation technique.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask



        
data = DataLoader(root = data_path, output_size = 512)
#test.show_image_plate_box(0, test.data_x, test.data_plate_y)
print('stop')
    
train = data.dataset_train
train = train.map(load_image_train)
train = train.shuffle(buffer_size = BUFFER_SIZE, seed = SEED)
#train = train.repeat()
train = train.batch(BATCH_SIZE)
train = train.prefetch(buffer_size = BUFFER_SIZE)

val = data.dataset_val
val = val.map(load_image_test)
#val = val.repeat()
val = val.batch(BATCH_SIZE)
val = val.prefetch(buffer_size = BUFFER_SIZE)

test = data.dataset_test
test = test.map(load_image_test)
#test = test.repeat()
test = test.batch(1)
test = test.prefetch(buffer_size = BUFFER_SIZE)




dropout_rate = 0.5
input_size = (IMG_SIZE, IMG_SIZE, N_CHANNELS)


initializer = 'he_normal'
            

# -- Encoder -- #
# Block encoder 1
inputs = Input(shape=input_size)
conv_enc_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
conv_enc_1 = Conv2D(64, 3, activation = 'relu', padding='same', kernel_initializer=initializer)(conv_enc_1)

# Block encoder 2
max_pool_enc_2 = MaxPooling2D(pool_size=(2, 2))(conv_enc_1)
conv_enc_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_2)
conv_enc_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_2)

# Block  encoder 3
max_pool_enc_3 = MaxPooling2D(pool_size=(2, 2))(conv_enc_2)
conv_enc_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_3)
conv_enc_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_3)
        
# Block  encoder 4
max_pool_enc_4 = MaxPooling2D(pool_size=(2, 2))(conv_enc_3)
conv_enc_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_4)
conv_enc_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_4)
# -- Encoder -- #

# ----------- #
maxpool = MaxPooling2D(pool_size=(2, 2))(conv_enc_4)
conv = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(maxpool)
conv = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv)
# ----------- #

# -- Decoder -- #
# Block decoder 1
up_dec_1 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv))
#conv_enc_4 = Resizing(height = 70, width = 70)(conv_enc_4)
merge_dec_1 = concatenate([conv_enc_4, up_dec_1], axis = 3)
conv_dec_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_1)
conv_dec_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_1)

# Block decoder 2
up_dec_2 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_1))
#conv_enc_3 = Resizing(height = 140, width = 140)(conv_enc_3)
merge_dec_2 = concatenate([conv_enc_3, up_dec_2], axis = 3)
conv_dec_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_2)
conv_dec_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_2)

# Block decoder 3
up_dec_3 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_2))
#conv_enc_2 = Resizing(height = 280, width = 280)(conv_enc_2)
merge_dec_3 = concatenate([conv_enc_2, up_dec_3], axis = 3)
conv_dec_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_3)
conv_dec_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_3)

# Block decoder 4
up_dec_4 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_3))
#conv_enc_1 = Resizing(height = 560, width = 560)(conv_enc_1)
merge_dec_4 = concatenate([conv_enc_1, up_dec_4], axis = 3)
conv_dec_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_4)
conv_dec_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
conv_dec_4 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
# -- Dencoder -- #

output = Conv2D(N_CLASSES, 1, activation = 'softmax')(conv_dec_4)




model = tf.keras.Model(inputs = inputs, outputs = output)
model.compile(optimizer = Adam(learning_rate = 0.0001), loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])


model_history = model.fit(train, epochs = 3, validation_data = val, use_multiprocessing = True, verbose = 1)
model.save('trained_uNet')
#eval_set = test.batch(BATCH_SIZE)

model.evaluate(test)

other_outputs = model.predict(test)


#outputs = []
#test = test.enumerate()
#iterator = iter(test)
#for x in range(len(test)):
#    buffer = iterator.get_next()
#    
#    out = model.predict_step(buffer)
#    outputs.append(np.array(out))
#outputs = model.predict_step(test)
outputs = np.array(outputs)
#np.save(r'model_predict_step_outputs', outputs)
#np.save(r'model_predict_outputs', other_outputs)

