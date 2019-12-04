#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:54:19 2019

@author: jahirmedinacs
"""
from removebg import RemoveBg
import base64

rmbg = RemoveBg("YOUR-API-KEY", "error.log")
with open("Average.jpg", "rb") as image_file:
	encoded_string = base64.b64encode(image_file.read())
    rmbg.remove_background_from_base64_img(encoded_string)