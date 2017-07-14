import cv2
import os
import numpy as np
import random

DESTINATION_DIR = "/home/ninja/spark_projects/dataset/"
classes = ['A', 'L', 'R', 'T', 'W']


img = cv2.imread("f0002_05_gray.jpg",0)
rows,cols = img.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),0,0.8)      #parameters: ((center of rotation, angle, scale)
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imwrite("img_rotated.jpg",dst)
