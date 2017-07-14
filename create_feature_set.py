import cv2
import pickle
import os
from random import shuffle
from tqdm import tqdm
import numpy as np
from config import *



def label_img(label):
    # this is required for one-hot encoding
    # the required labels should be like in the format given below:

    #                         ONE_HOT_ENCODED LABELS
    # for sunya i.e. 0  : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # for ek i.e. 1     : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # for dui i.e. 2    : [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # for tin i.e. 3    : [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # for char i.e. 4   : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    # for paanch i.e. 5 : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    # for cha i.e. 6    : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    # for saat i.e. 7   : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    # for aath i.e. 8   : [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    # for nau i.e. 9    : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


    encoded_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    encoded_label[int(label)] = 1
    return encoded_label




def create_training_data():
    training_data = []
    labels = []
    for file_name in os.listdir(TRAIN_DIR):
        labels.append(file_name)

    for label in labels:
        encoded_label = label_img(label)
        path = os.path.join(TRAIN_DIR,label)
        for file in os.listdir(path):
            path_of_image = os.path.join(path, file)
            img = cv2.imread(path_of_image, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img), np.array(encoded_label)])
        print "Feeding training data for label ", label, " completed."

    shuffle(training_data)

    # np.save('train_data.npy', training_data)


    # training data
    train = training_data[:int(0.85 * len(training_data))]
    # testing data
    test = training_data[int(0.85 * len(training_data)):]

    # training data set
    train_x = np.array([x[0] for x in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    train_y = np.array([x[1] for x in train])

    # testing data set
    test_x = np.array([x[0] for x in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_y = np.array([x[1] for x in test])

    return train_x, train_y, test_x, test_y



# to create feature_sets uncomment the next 4 lines.
if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_training_data()
    print "Dumping data into pickle"
    with open('feature_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
