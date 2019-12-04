#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:54:19 2019

@author: jahirmedinacs
"""

import copy
import os
import sys
import copy
import numpy as np

from OpenGL import GL
from OpenGL import GLU
from OpenGL import GLUT

import cv2
import numpy as np
import dlib
import random
import copy

import multiprocessing as MP

import os, PIL
from PIL import Image

def main():
    GLUT.glutInit(sys.argv)
    GLUT.glutInitDisplayMode(GLUT.GLUT_SINGLE | GLUT.GLUT_RGBA | GLUT.GLUT_DEPTH)
    GLUT.glutInitWindowSize(400, 400)
    GLUT.glutInitWindowPosition(200, 200)

    GLUT.glutCreateWindow("Dummy")

    """
    GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
    GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
    GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, [1.0, 1.0, 1.0, 0.0])

    GL.glEnable(GL.GL_NORMALIZE)
    GL.glEnable(GL.GL_LIGHTING)
    GL.glEnable(GL.GL_LIGHT0)
    """

    GL.glDepthFunc(GL.GL_LEQUAL)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glClearDepth(1.0)
    GL.glClearColor(0.650, 0.780, 0.8, 0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glLoadIdentity()
    #GL.glOrtho(-20, 20, -20, 20, -20, 20)
    GL.glMatrixMode(GL.GL_MODELVIEW)

    GLU.gluPerspective(100, 1.0, 1.0, 100.0)
    GL.glTranslatef(0.0, 0.0, 0.0)
    GLU.gluLookAt(0, 10, 10, 0, 0, 0, 0, 1, 0)


    #### OPENCV

    cap = cv2.VideoCapture(2)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    ########


    while True:
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        ########################################
        ### Getting frame and detecting faces
        _, frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = detector(gray)
        
        center = np.zeros(2,dtype=int)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            # cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),3)
            landmarks = predictor(gray, face)
            
            center = np.zeros(2,dtype=int)
            for n in range(0,68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                # cv2.circle(frame,(x,y),4,(255,0,0),-1)
                center += np.array([x,y])
            
            center = center//68

        print(center)
        cv2.circle(frame, (center[0],center[1]), 4, (0,0,255), -1)
        
        # convert image to OpenGL texture format
        tx_image = cv2.flip(frame, 0)
        tx_image = Image.fromarray(tx_image)     
        ix = tx_image.size[0]
        iy = tx_image.size[1]
        tx_image = tx_image.tobytes('raw', 'BGRX', 0, -1)
             
        # create texture
        texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, ix, iy, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, tx_image)

        # adding obj with video texture

        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glBegin(GL.GL_POLYGON)
        GL.glTexCoord2fv([0.0, 0.0])
        GL.glVertex3fv([5.0, 5.0, -5.0])
        GL.glTexCoord2fv([1.0, 0.0])
        GL.glVertex3fv([00.0, 5.0, -5.0])
        GL.glTexCoord2fv([1.0, 1.0])
        GL.glVertex3fv([00.0, 5.0, 0.0])
        GL.glTexCoord2fv([0.0, 1.0])
        GL.glVertex3fv([5.0, 5.0, 0.0])
        GL.glEnd()

        ########################################

        GL.glFlush()
        GLUT.glutPostRedisplay()

    GLUT.glutMainLoop()

main()