# -*- coding: utf-8 -*-
"""
Created on Wed May  1 09:01:03 2019

@author: MaryamHashemi
"""

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math
import glob
import cv2
 



#images_path1="E:/data/dataset_B_Eye_Images/dataset_B_Eye_Images/closed/"
images_path2="E:/data/dataset_B_Eye_Images/dataset_B_Eye_Images/open/"
#images1=glob.glob(images_path1+"*.jpg")
images2=glob.glob(images_path2+"*.jpg")
images=[images2]

labelname=[]
labelclass=[]
for i in images:
    countimg=0
    for j in i:
        countimg+=1
        img=cv2.imread(j)
        h=img.shape[0]
        w=img.shape[1]
        center = (w / 2, h / 2)
 
        angle90 = 90
        angle180 = 180
        angle270 = 270
        angle120=120
        angle60=60
        angle240=240
        angle330=330
        scale = 1.0
        
        # Perform the counter clockwise rotation holding at the center
#         90 degrees
#        M = cv2.getRotationMatrix2D(center, angle90, scale)
#        rotated90 = cv2.warpAffine(img, M, (h, w))
         
        # 180 degrees
#        M = cv2.getRotationMatrix2D(center, angle180, scale)
#        rotated180 = cv2.warpAffine(img, M, (w, h))

#         
##        # 270 degrees
#        M = cv2.getRotationMatrix2D(center, angle270, scale)
#        rotated270 = cv2.warpAffine(img, M, (h, w))
        
         # 120 degrees
#        M = cv2.getRotationMatrix2D(center, angle120, scale)
#        rotated120 = cv2.warpAffine(img, M, (h, w))
         
         #         60 degrees
        M = cv2.getRotationMatrix2D(center, angle60, scale)
#        rotated60 = cv2.warpAffine(img, M, (h, w))
        
        
        #         240 degrees
#        M = cv2.getRotationMatrix2D(center, angle240, scale)
#        rotated240 = cv2.warpAffine(img, M, (h, w))
        
        
        #         330 degrees
        M = cv2.getRotationMatrix2D(center, angle330, scale)
        rotated330 = cv2.warpAffine(img, M, (h, w))

 
        name ="%d_%d.jpg"%(7 ,countimg)
        filename="E:/data/dataset_B_Eye_Images/dataset_B_Eye_Images/dataaugmentation/"+name
#        cv2.imwrite(filename, rotated90)
#        cv2.imwrite(filename, rotated180)
#        cv2.imwrite(filename, rotated270)
#        cv2.imwrite(filename, rotated120)
#        cv2.imwrite(filename, rotated60)
#        cv2.imwrite(filename, rotated240)
        cv2.imwrite(filename, rotated330)
           
           

