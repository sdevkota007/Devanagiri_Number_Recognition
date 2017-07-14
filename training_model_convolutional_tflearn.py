import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import os
import pickle
from config import IMG_SIZE, LR


train_x, train_y, test_x, test_y = pickle.load(open("feature_set.pickle","rb"))
X=train_x
Y=train_y

print X[1].shape
print Y[1]



MODEL_NAME = 'fingerprint_classification-{}-{}.model'.format(LR, '2conv-basic') # just so to remember which saved model is which, sizes must match



convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)


convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 512, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format('MODEL_NAME')):
    model.load(MODEL_NAME)
    print('model loaded')


#--------------------------------------------------to train the model-----------------------------------------------


model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)


# ---------------------------------------------------to test the model----------------------------------------------------

# import matplotlib.pyplot as plt
# import numpy as np
#
#
# test_data = np.load('test_data.npy')
# IMG_SIZE = 256
# fig = plt.figure()
#
# for num, data in enumerate(test_data[:9]):
#     # cat: [1,0]
#     # dog: [0,1]
#
#     img_num = data[1]
#     img_data = data[0]
#
#     y = fig.add_subplot(3, 3, num + 1)
#     orig = img_data
#     data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
#     # model_out = model.predict([data])[0]
#     model_out = model.predict([data])[0]
#
#     print model_out
#
#     if np.argmax(model_out) == 0:
#         str_label = 'Arch'
#     elif np.argmax(model_out) == 1:
#         str_label = 'Left loop'
#     elif np.argmax(model_out) == 2:
#         str_label = 'Right loop'
#     elif np.argmax(model_out) == 3:
#         str_label = 'Tented Arch'
#     elif np.argmax(model_out) == 4:
#         str_label = 'Whorl'
#
#     y.imshow(orig, cmap='gray')
#     plt.title(str_label)
#     y.axes.get_xaxis().set_visible(False)
#     y.axes.get_yaxis().set_visible(False)
# plt.show()