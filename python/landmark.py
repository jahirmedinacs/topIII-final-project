#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:54:19 2019

@author: jahirmedinacs
"""

import cv2
import numpy as np
import dlib
import random
import copy

DONE = False

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

x_ratio = 0.3
y_ratio = 0.5

frame_id = 0
saved_frames_counter = 0
save_frame_path = "./face_frames/" # overwrite previus images
max_saved_frames = 50
frame_container = [None] * max_saved_frames
frame_container_coords = [None] * max_saved_frames 
while True:
    _, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    
    center = np.zeros(2,dtype=int)
    x_Crop = 0
    y_Crop = 0
    save = False
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),3)

        y_Crop = int((x2 - x1) * (1.0 + x_ratio)) // 2
        x_Crop = int((y2 - y1) * (1.0 + y_ratio)) // 2

        landmarks = predictor(gray, face)
        
        center = np.zeros(2,dtype=int)
        for n in range(0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # cv2.circle(frame,(x,y),4,(255,0,0),-1)
            center += np.array([x,y])
        
        center = center//68

        save = random.random() > 0.5


    # print(center)
    # cv2.circle(frame, (center[0],center[1]), 4, (0,0,255), -1)
    
    frame_coords = [center[1] - x_Crop, center[1] + x_Crop, center[0] - y_Crop , center[0] + y_Crop]
    cv2.imshow("LandMark Detection", frame)
    if save and saved_frames_counter < max_saved_frames:
        frame_container[saved_frames_counter] = copy.copy(frame)
        frame_container_coords[saved_frames_counter] = copy.copy(frame_coords)
        saved_frames_counter += 1

    if saved_frames_counter == max_saved_frames and not DONE:
        print("The Coords")
        shape = np.matrix(frame_container_coords).sum(axis=0) // max_saved_frames
        shape = shape.tolist()[0] 
        print(shape)
        for ii in range(max_saved_frames):
            alt_frame = frame_container[ii][shape[0]:shape[1], shape[2]:shape[3], :]
            cv2.imwrite(save_frame_path + "frame{:d}.jpg".format(ii), alt_frame)
        DONE = True

    
    key =   cv2.waitKey(1)
    if key == 27:
        break
    
    frame_id += 1

cv2.destroyAllWindows()
cap.release()