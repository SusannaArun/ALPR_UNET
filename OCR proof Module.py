# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:20:02 2022

@author: bentw
"""

import numpy as np

from PIL import Image, ImageOps
import tensorflow as tf



UPPER_CUTOFF = .55
LOWER_CUTOFF = .45
IMG_SIZE = 512
BUFFER_SIZE = 4


def apply_mask(image, mask):
    for x in range(0, 512):
        for y in range(0, 512):
            if mask[x, y]==0:
                image[x, y, : ] = 0
    return image

test_dir = r'E:\data_test'
label_dir = r'E:\label_test'

im = Image.open(test_dir+'/8.png')
im.show()
im = np.array(im)


label = Image.open(label_dir+'/8.png')
label.show()

#label = ImageOps.grayscale(label)
#label = np.array(im)

#masks_predict_step = np.load(r'E:\csc7760\term project\model_predict_step_outputs.npy')
masks_predict = np.load(r'E:\csc7760\term project\model_predict_outputs-0.npy')
#test = masks[0]
print('debug stop point')

high_0 = masks_predict[:,:,:,0]
low_0 = masks_predict[:,:,:,1]


high_0 = high_0<UPPER_CUTOFF
high_0 = high_0.astype(np.int8())
low_0 = low_0>LOWER_CUTOFF
low_0 = low_0.astype(np.int8())



#high_1 = masks_predict_step[:,:,:,0]
#low_1 = masks_predict_step[:,:,:,1]


#high_1 = high_1<UPPER_CUTOFF
#high_1 = high_1.astype(np.int8())
#low_1= low_1>LOWER_CUTOFF
#low_1 = low_1.astype(np.int8())


del masks_predict





test = high_0[0]
#test = np.flip(test, axis = 1)
copy = im
im = apply_mask(im, test)
im = Image.fromarray(im)
im.show()

im.save('/applied_mask.png')


print("debug stop point")
