# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 12:07:13 2022

@author: bentw
"""

import os
import numpy as np
import PIL
import tensorflow as tf


data_path = r'C:/UFPR-ALPR dataset'


class DataLoader:
    '''
    indexing for labels is as follows:
        x, y, width, hieght
        
        so data_x[index][y][x][channel] gives you the top left corner
    
    
    '''
    
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
    data_dir = "/data"
    NUM_PIXELS = 512#264
    HEIGHT = 1080
    WIDTH = 1920


    def __init__(self, root = r'C:\data subset'):
        self.global_set = sorted(os.listdir(data_path))
        self.data_x, self.data_y = self.__get_filepaths(self.global_set, data_path, self.data_x, self.data_y)
        self.data_plate_y, self.data_car_y, self.data_plate_text_y = self.__parse_y(self.data_y, self.data_plate_y, self.data_car_y, self.data_plate_text_y)

        if(os.path.isdir(self.data_dir)==False):
            os.mkdir(self.data_dir)
            self.__open_images(self.data_x, self.data_plate_y, self.data_car_y)
        

        
#Y is sorted. Do not shuffle until tf.dataset has been created
#this also rectifies the labels by the resizing factor of the shrunken image
    def __parse_y(self, data_y, data_plate_y, data_car_y, data_plate_text_y):
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

    
    #this method requires a lot of memory. just don't freak out if you run out of mem. also it takes forever. sorry its just a lot of data
    def __open_images(self, data_x, data_plate_y, data_car_y, grayscale = False):
        for x in range(0, len(data_x)):
            buffer = PIL.Image.open(data_x[x])

            buffer = self.__crop_images(buffer, x, data_plate_y, data_car_y)
            if(grayscale==True):
                PIL.ImageOps.grayscale(buffer)
            
            buffer.close()
        
        
    
    #needs testing <- works as far as I can tell
    
    def __check_bounds(self, left, top, right, bottom, plate):
        '''Makes sure that we don't crop out the plate or outside of the image'''    
        if(left>plate[0]):
            left = plate[0]
            right = left+self.NUM_PIXELS
        
        if(right<(plate[0]+plate[2])):
            right = (plate[0]+plate[2])
            left = right-self.NUM_PIXELS
        
        if(top>plate[1]):
            top = plate[1]
            bottom = top+self.NUM_PIXELS
        
        if(bottom<(plate[1]+plate[3])):
            bottom = plate[1]+plate[3]
            top = bottom - self.NUM_PIXELS
            
        if(left<0):
            left = 0
            right = self.NUM_PIXELS
        
        if(right>self.WIDTH):
            right = self.WIDTH
            left = right-self.NUM_PIXELS
        
        if(top<0):
            top = 0
            bottom = self.NUM_PIXELS
        
        if(bottom>self.HEIGHT):
            bottom = self.HEIGHT
            top = bottom-self.NUM_PIXELS
            
        return left, top, right, bottom
    
    def __adjust_label(self, plate, left, top):
        
        buffer = plate
        buffer[0] = buffer[0]-left
        buffer[1] = buffer[1]-top
        return buffer
    
    def __crop_and_save(self, image, index, counter, left, top, right, bottom, plate):
        left, top, right, bottom = self.__check_bounds(left, top, right, bottom, plate)
        crop = image.crop((left, top, right, bottom))
        #crop.show()
        crop.save(r'/data/'+str(index)+'-'+str(counter)+'.png')
        return crop
    
    #REDO THIS. MAKE A FUNCTION THAT CHECKS THE BOUNDS AND CORRECTS THEM IF CROP IS OUT OF BOUNDS
    def __crop_images(self, image, index, data_plate_y, data_car_y):
        counter = 0
        
        car_y_buffer = data_car_y[index]
        left = car_y_buffer[0]
        right = car_y_buffer[0]+self.NUM_PIXELS
        top = car_y_buffer[1]
        bottom = car_y_buffer[1]+self.NUM_PIXELS
        #label_buffer = []
        local_plate = np.copy(data_plate_y[index])
        #data_plate_y[index] = self.__adjust_label(index, data_plate_y, left, bottom)
        
        crop1 = self.__crop_and_save(image, index, counter, left, top, right, bottom, local_plate)
        adjusted_label = self.__adjust_label(local_plate, left, top)
        crop1 = self.__show_image_plate_granular(adjusted_label, crop1, index, counter)
        #label_buffer.append(adjusted_label)
        counter+=1
        #crop1.show()
        
        right = car_y_buffer[0]+car_y_buffer[2]
        left = right-self.NUM_PIXELS               
        local_plate = np.copy(data_plate_y[index])
        crop2 = self.__crop_and_save(image, index, counter, left, top, right, bottom, local_plate)
        adjusted_label = self.__adjust_label(local_plate, left, top)
        crop2 = self.__show_image_plate_granular(adjusted_label, crop2, index, counter)
        #crop2.show()
        counter+=1
        
        bottom = car_y_buffer[1]+car_y_buffer[3]
        top = bottom-self.NUM_PIXELS
        local_plate = np.copy(data_plate_y[index])      
        crop3 = self.__crop_and_save(image, index, counter, left, top, right, bottom, local_plate)
        adjusted_label = self.__adjust_label(local_plate, left, top)
        crop3 = self.__show_image_plate_granular(adjusted_label, crop3, index, counter)
        #crop3.show()
        counter+=1
        
        
        left = car_y_buffer[0]
        right = car_y_buffer[0]+self.NUM_PIXELS
        local_plate = np.copy(data_plate_y[index])       
        crop4 = self.__crop_and_save(image, index, counter, left, top, right, bottom, data_plate_y[index])
        adjusted_label = self.__adjust_label(local_plate, left, top)
        crop4 = self.__show_image_plate_granular(adjusted_label, crop4, index, counter)
        #crop4.show()
        
        return crop1#, crop2, crop3, crop4
        
    def __show_image_plate_granular(self, label, image, index, counter):
        image = np.array(image)
        image[label[1]:(label[1]+label[3]), label[0]:(label[0]+label[2]), 1] = 0
        image = PIL.Image.fromarray(image.astype(np.uint8))
        image.save(r'/data/'+str(index)+'-'+str(counter)+'labeled'+'.png')
        return image
    
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
        
        
    
    def show_image_car_box(self, image_index, data_x, data_plate_y):
        pass
    
    def crop_to_y_bbox(self, data_x, data_plate_y, data_car_y):
        pass
        
    def __get_filepaths(self, global_set, data_path, data_x, data_y):        
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






test = DataLoader(root = data_path)
#test.show_image_plate_box(0, test.data_x, test.data_plate_y)
print('stop')



