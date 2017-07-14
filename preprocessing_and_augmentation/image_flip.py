import cv2
import os

DESTINATION_DIR = "/home/ninja/spark_projects/dataset/"
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ]


img = cv2.imread("f0002_05_gray.jpg",0)
img_flipped = cv2.flip(img,1)   #flip image in x-axis
cv2.imwrite("img_flipped.jpg",img_flipped)
