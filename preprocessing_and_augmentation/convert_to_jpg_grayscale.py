import cv2
import numpy as np
import sys
import os


DESTINATION_DIR = "/home/ninja/spark_projects/dataset/"
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ]

def convert_format():
    '''
    read images from all the directories and convert to grayscale
    '''
    for folder in classes:
        img_path = os.path.join(DESTINATION_DIR, folder)

        for img_name in os.listdir(img_path):
            old_path = os.path.join(img_path, img_name)
            img = cv2.imread(str(old_path),0)
            name = img_name.split(".")
            new_name = name[0]+"_gray.jpg"
            new_path = os.path.join(img_path, new_name)
            cv2.imwrite(new_path,img)
            print new_name+ " saved"



def remove_png_files():
    '''
    removes all . files from DESTINATION_DIR
    '''
    file_format = "*.png"
    os.system("find "+DESTINATION_DIR+" -name "+file_format+" -type f -delete")
    print "all .png files deleted"


if __name__== "__main__":
    # convert_format()
    # remove_png_files()
