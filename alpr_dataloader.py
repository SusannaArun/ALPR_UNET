# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 12:07:13 2022

@author: bentw
"""

import os
import numpy as np
import PIL
import tensorflow as tf


data_path = r'C:\data subset'


class DataLoader:
    global_set = []
    
    TRAINING = 0
    TESTING = 1
    VAL = 2
    data = []
    data_x = []
    data_y = []
    data_plate_y = []
    data_car_y = []
    data_plate_text_y = []
    NUM_PIXELS = 512#264
    resizing_factor_x = float(NUM_PIXELS/1080)
    resizing_factor_y = float(NUM_PIXELS/1920)
    im_size = (int(1080*resizing_factor_x), int(1920*resizing_factor_y))

    def __init__(self, root = r'C:\data subset'):
        self.global_set = sorted(os.listdir(data_path))
        self.data_x, self.data_y = self.get_filepaths(self.global_set, data_path, self.data_x, self.data_y)
        self.data_plate_y, self.data_car_y, self.data_plate_text_y = self.parse_y(self.data_y, self.data_plate_y, self.data_car_y, self.data_plate_text_y)
        self.data_plate_y = self.resize_labels(self.data_plate_y)
        self.data_car_y = self.resize_labels(self.data_car_y)
        self.data_x = self.open_images(self.data_x)
#Y is sorted. Do not shuffle until tf.dataset has been created
#this also rectifies the labels by the resizing factor of the shrunken image
    def parse_y(self, data_y, data_plate_y, data_car_y, data_plate_text_y):
        car_pos = 1
        plate_text = 6
        plate_pos = 7
        for x in range(0, len(data_y)):
            f = open(data_y[x])
            lines = f.readlines()
            
            data_car_y.append(lines[car_pos][18:])
            data_car_y[x] = [int(i) for i in data_car_y[x].split() if i.isdigit()] #parsing happens here
            data_plate_y.append(lines[plate_pos][16:])
            data_plate_y[x] = [int(i) for i in data_plate_y[x].split() if i.isdigit()] #parsing happens here
            data_plate_text_y.append(lines[plate_text][7:])
            f.close()
            
        return np.array(data_plate_y), np.array(data_car_y), np.array(data_plate_text_y)

    def resize_labels(self, y):
        for x in range(0, len(y)):
            y[x,0] = y[x,0]*self.resizing_factor_y 
            
            y[x,1] = y[x,1]*self.resizing_factor_x
            
            y[x,2] = y[x,2]*self.resizing_factor_y
            
            y[x,3] = y[x,3]*self.resizing_factor_x
    
        return y
    #this method requires a lot of memory. just don't freak out if you run out of mem. also it takes forever. sorry its just a lot of data
    def open_images(self, data_x):
        for x in range(0, len(data_x)):
            buffer = PIL.Image.open(data_x[x])
            buffer = buffer.resize(self.im_size)
            buffer = np.array(buffer, dtype = np.float32)
            data_x[x] = buffer/255
            print(x)
        
        return np.array(data_x)
        
    def get_filepaths(self, global_set, data_path, data_x, data_y):        
        #this is a really convoluted way to get the file paths of each image and label
        for x in range(0, len(global_set)):
            child_path = os.path.join(data_path, global_set[x])
            buffer = os.listdir(child_path)
            for y in range(0, len(buffer)):
                child_path_y = os.path.join(child_path, buffer[y])
                buff = os.listdir(child_path_y)
                for z in range(0, len(buff)):
                    buff[z] = os.path.join(child_path_y, buff[z])
                    if '.png' in buff[z]:
                        data_x.append(buff[z])
                    elif'txt' in buff[z]:
                        data_y.append(buff[z])
        return sorted(data_x), sorted(data_y)






test = DataLoader()



