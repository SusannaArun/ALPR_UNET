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
    data_plate_chars_y = []
    data_dir = "/data"
    INPUT_SIZE = -1#264
    OUTPUT_SIZE = -1
    RATIO = -1 #cast to int when using ratio
    HEIGHT = 1080
    WIDTH = 1920
    dataset = None


    def __init__(self, root = r'C:\data subset', input_size = 512, output_size = 388, val_split = .2):
        self.INPUT_SIZE = input_size
        self.OUTPUT_SIZE = output_size
        self.RATIO = self.OUTPUT_SIZE/self.INPUT_SIZE
        self.global_set = sorted(os.listdir(data_path))
        self.data_x, self.data_y = self.__get_filepaths(self.global_set, data_path, self.data_x, self.data_y)
        self.data_plate_y, self.data_car_y, self.data_plate_text_y, self.data_plate_chars_y = self.__parse_y(self.data_y, 
                                                                                                             self.data_plate_y, 
                                                                                                             self.data_car_y, 
                                                                                                             self.data_plate_text_y, 
                                                                                                             self.data_plate_chars_y)

        if(os.path.isdir(self.data_dir)==False):
            os.mkdir(self.data_dir)
            self.data_plate_chars_y = self.__crop_images(self.data_x, self.data_plate_y, self.data_car_y, self.data_plate_chars_y)
        else:
            self.data_plate_chars_y = self.__masks_only(self.data_plate_y, self.data_car_y, self.data_plate_chars_y)
        self.dataset = tf.keras.utils.image_dataset_from_directory(directory = self.data_dir, 
                                                                   labels = None,
                                                                   label_mode = None,
                                                                   validation_split = .2,
                                                                   seed = 42,
                                                                   subset = "training",
                                                                   batch_size = 32,
                                                                   image_size = (self.INPUT_SIZE, self.INPUT_SIZE))

        
#Y is sorted. Do not shuffle until tf.dataset has been created
#this also rectifies the labels by the resizing factor of the shrunken image
    def __parse_y(self, data_y, data_plate_y, data_car_y, data_plate_text_y, data_plate_chars_y):
        car_pos = 1
        plate_text = 6
        plate_pos = 7
        char_pos = [8, 9, 10, 11, 12, 13, 14]
        for x in range(0, len(data_y)):
            f = open(data_y[x])
            lines = f.readlines()
            data_car_y.append(lines[car_pos][18:])
            data_car_y[x] = [int(i) for i in data_car_y[x].split() if i.isdigit()] #parsing happens here
            data_plate_y.append(lines[plate_pos][16:])
            data_plate_y[x] = [int(i) for i in data_plate_y[x].split() if i.isdigit()] #parsing happens here
            data_plate_text_y.append(lines[plate_text][7:])
            char_buffer = []
            for y in range(0, len(char_pos)):
                coord_string = lines[char_pos[y]][9:]
                coord_string = [int(i) for i in coord_string.split() if i.isdigit()]
                char_buffer.append(coord_string)
                
            data_plate_chars_y.append(char_buffer)
            f.close()
            
        return np.array(data_plate_y), np.array(data_car_y), np.array(data_plate_text_y), np.array(data_plate_chars_y)

    
    #this method requires a lot of memory. just don't freak out if you run out of mem. also it takes forever. sorry its just a lot of data
    
        
        
    
    #needs testing <- works as far as I can tell
    
    def __check_bounds(self, left, top, right, bottom, plate):
        '''Makes sure that we don't crop out the plate or outside of the image'''    
        if(left>plate[0]):
            left = plate[0]
            right = left+self.INPUT_SIZE
        
        if(right<(plate[0]+plate[2])):
            right = (plate[0]+plate[2])
            left = right-self.INPUT_SIZE
        
        if(top>plate[1]):
            top = plate[1]
            bottom = top+self.INPUT_SIZE
        
        if(bottom<(plate[1]+plate[3])):
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
        buffer[0] = buffer[0]-left
        buffer[1] = buffer[1]-top
        return buffer
    
    def __crop_and_save(self, image, index, counter, left, top, right, bottom, plate):
        #left, top, right, bottom = self.__check_bounds(left, top, right, bottom, plate)
        crop = image.crop((left, top, right, bottom))
        #crop.show()
        crop.save(r'/data/'+str(index)+'-'+str(counter)+'.png')
        return crop
    
    def __adjust_input_to_output(self, bbox, left, top):
        for x in range(len(bbox)):
            self.__adjust_label(bbox[x], left, top)
            for y in range(len(bbox[x])):
                bbox[x][y] = int(bbox[x][y]*self.RATIO)
        return bbox
    
    
    #TEST THE MASK CREATION AND MAKE THE LABELS IN THE CROP IMAGES FUNCTION
    def __make_seg_mask(self, index, data_plate_chars_y, left, top):
        buffer = np.copy(data_plate_chars_y[index])
        mask = np.zeros((self.OUTPUT_SIZE, self.OUTPUT_SIZE), dtype = np.float32)
        buffer = self.__adjust_input_to_output(buffer, left, top)
        for x in range(0, len(buffer)):
            #buffer[x] = self.__adjust_label(buffer[x], left, top)
            #buffer[x] = self.__adjust_input_to_output(buffer[x])
            if(x<(len(buffer)-1)):
                stop_x = buffer[x+1][0]
            else:
                stop_x = buffer[x][0]+buffer[x][2]
            mask[buffer[x][1]:(buffer[x][1]+buffer[x][3]), buffer[x][0]:stop_x] = 1
        
        return mask
    

    
    def __masks_only(self, data_plate_y, data_car_y, data_plate_chars_y):
        output_labels = []
        for x in range(len(data_plate_y)):
            counter = 0
            
            car_y_buffer = data_car_y[x]
            left = car_y_buffer[0]
            right = car_y_buffer[0]+self.INPUT_SIZE
            top = car_y_buffer[1]
            bottom = car_y_buffer[1]+self.INPUT_SIZE
            label_buffer = []
            
            #anchor around top left corner
            local_plate = np.copy(data_plate_y[x])
            left, top, right, bottom = self.__check_bounds(left, top, right, bottom, local_plate)
            adjusted_label = self.__make_seg_mask(x, data_plate_chars_y, left, top)
            #self.__save_segment_label(adjusted_label, x, counter)
            output_labels.append(adjusted_label)
            counter+=1
                      
            
            #anchor around top right corner
            right = car_y_buffer[0]+car_y_buffer[2]
            left = right-self.INPUT_SIZE               
            local_plate = np.copy(data_plate_y[x])
            left, top, right, bottom = self.__check_bounds(left, top, right, bottom, local_plate)
            adjusted_label = self.__make_seg_mask(x, data_plate_chars_y, left, top)
            #self.__save_segment_label(adjusted_label, x, counter)
            output_labels.append(adjusted_label)
            counter+=1
            
            #anchor around bottom left corner
            bottom = car_y_buffer[1]+car_y_buffer[3]
            top = bottom-self.INPUT_SIZE
            local_plate = np.copy(data_plate_y[x])      
            left, top, right, bottom = self.__check_bounds(left, top, right, bottom, local_plate)
            adjusted_label = self.__make_seg_mask(x, data_plate_chars_y, left, top)
            #self.__save_segment_label(adjusted_label, x, counter)
            output_labels.append(adjusted_label)
            counter+=1
            
            #anchor around bottom right corner
            left = car_y_buffer[0]
            right = car_y_buffer[0]+self.INPUT_SIZE
            local_plate = np.copy(data_plate_y[x])       
            left, top, right, bottom = self.__check_bounds(left, top, right, bottom, local_plate)
            adjusted_label = self.__make_seg_mask(x, data_plate_chars_y, left, top)
            #self.__save_segment_label(adjusted_label, x, counter)
            output_labels.append(adjusted_label)
        return np.array(output_labels, dtype = np.uint8)
        
    def __crop_images(self, data_x, data_plate_y, data_car_y, data_plate_chars_y):
    
        '''
        If the images aren't already preprocessed, that happens here. Cropped images and their corresponding segmentation
        mask are saved here in a file in the root directory called 'data. it makes and saves 18000 images, so this
        takes a long time to run the first time. 
        '''
        output_labels = []
        for index in range(0, len(data_x)):
            
            
            image = PIL.Image.open(data_x[index])
            
            counter = 0
        
            car_y_buffer = data_car_y[index]
            left = car_y_buffer[0]
            right = car_y_buffer[0]+self.INPUT_SIZE
            top = car_y_buffer[1]
            bottom = car_y_buffer[1]+self.INPUT_SIZE
            label_buffer = []
        
            #anchor around top left corner
            local_plate = np.copy(data_plate_y[index])
            left, top, right, bottom = self.__check_bounds(left, top, right, bottom, local_plate)
            crop1 = self.__crop_and_save(image, index, counter, left, top, right, bottom, local_plate)
            #adjusted_label = self.__adjust_label(local_plate, left, top)
            adjusted_label = self.__make_seg_mask(index, data_plate_chars_y, left, top)
            #self.__save_segment_label(adjusted_label, index, counter)
            #crop1 = self.__show_image_plate_granular(adjusted_label, crop1, index, counter)
            output_labels.append(adjusted_label)
            counter+=1
            #crop1.show()
        
        
            #anchor around top right corner
            right = car_y_buffer[0]+car_y_buffer[2]
            left = right-self.INPUT_SIZE               
            local_plate = np.copy(data_plate_y[index])
            left, top, right, bottom = self.__check_bounds(left, top, right, bottom, local_plate)
            crop2 = self.__crop_and_save(image, index, counter, left, top, right, bottom, local_plate)
            #adjusted_label = self.__adjust_label(local_plate, left, top)
            adjusted_label = self.__make_seg_mask(index, data_plate_chars_y, left, top)
            #self.__save_segment_label(adjusted_label, index, counter)
            #crop2 = self.__show_image_plate_granular(adjusted_label, crop2, index, counter)
            output_labels.append(adjusted_label)
            #crop2.show()
            counter+=1
        
            #anchor around bottom left corner
            bottom = car_y_buffer[1]+car_y_buffer[3]
            top = bottom-self.INPUT_SIZE
            local_plate = np.copy(data_plate_y[index])      
            left, top, right, bottom = self.__check_bounds(left, top, right, bottom, local_plate)
            crop3 = self.__crop_and_save(image, index, counter, left, top, right, bottom, local_plate)
            #adjusted_label = self.__adjust_label(local_plate, left, top)
            adjusted_label = self.__make_seg_mask(index, data_plate_chars_y, left, top)
            #self.__save_segment_label(adjusted_label, index, counter)
            #crop3 = self.__show_image_plate_granular(adjusted_label, crop3, index, counter)
            output_labels.append(adjusted_label)
            #crop3.show()
            counter+=1
        
            #anchor around bottom right corner
            left = car_y_buffer[0]
            right = car_y_buffer[0]+self.INPUT_SIZE
            local_plate = np.copy(data_plate_y[index])       
            left, top, right, bottom = self.__check_bounds(left, top, right, bottom, local_plate)
            crop4 = self.__crop_and_save(image, index, counter, left, top, right, bottom, data_plate_y[index])
            #adjusted_label = self.__adjust_label(local_plate, left, top)
            adjusted_label = self.__make_seg_mask(index, data_plate_chars_y, left, top)
            #self.__save_segment_label(adjusted_label, index, counter)
            #crop4 = self.__show_image_plate_granular(adjusted_label, crop4, index, counter)
            output_labels.append(adjusted_label)
            #crop4.show()
            
            
            
            
            image.close()
        return np.array(output_labels, dtype = np.uint8)
        
        
    def __show_image_plate_granular(self, label, image, index, counter):
        image = np.array(image)
        image[label[1]:(label[1]+label[3]), label[0]:(label[0]+label[2]), 1] = 0
        image = PIL.Image.fromarray(image.astype(np.uint8))
        image.save(r'/data/'+str(index)+'-'+str(counter)+'labeled'+'.png')
        return image
    
    def __save_segment_label(self, image, index, counter):
        image = np.array(image)
        image = PIL.Image.fromarray(image.astype(np.uint8))
        image.save(r'/data/'+str(index)+'-'+str(counter)+'labeled'+'.png')
    
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



