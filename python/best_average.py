#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:54:19 2019

@author: jahirmedinacs
"""

import os, numpy, PIL
from PIL import Image

# Access all PNG files in directory
save_frame_path = "/face_frames/"
PATH = os.getcwd() + save_frame_path

allfiles=os.listdir(PATH)
imlist=[filename for filename in allfiles if  filename[-4:] in [".jpg",".JPG"]]

# Assuming all images are the same size, get dimensions of first image
w,h=Image.open(PATH + imlist[0]).size
N=len(imlist)

# Create a numpy array of floats to store the average (assume RGB images)
arr=numpy.zeros((h,w,3),numpy.float)

# Build up average pixel intensities, casting each image as an array of floats
for im in imlist:
    imarr=numpy.array(Image.open(PATH + im),dtype=numpy.float)
    arr=arr+imarr/N

# Round values in array and cast as 8-bit integer
arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)

# Generate, save and preview final image
out=Image.fromarray(arr,mode="RGB")
out.save("Average.jpg")