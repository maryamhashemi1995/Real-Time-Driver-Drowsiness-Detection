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
#import tensorflow as tf
#from tensorflow.python.client import device_lib 
#from keras.preprocessing import image
from keras.utils import np_utils
from skimage.transform import resize
import glob
from keras.models import Sequential
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, InputLayer, Dropout, Activation
from keras.utils import plot_model
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D

#oc curve and auc
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.models import model_from_json
from sklearn.metrics import average_precision_score

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad



images_path1="E:/data/dataset_B_Eye_Images/dataset_B_Eye_Images/closed/"
images_path2="E:/data/dataset_B_Eye_Images/dataset_B_Eye_Images/open/"
images1=glob.glob(images_path1+"*.jpg")
images2=glob.glob(images_path2+"*.jpg")
images=[images1,images2]


labelname=[]
labelclass=[]
countlabel=-1
for i in images:
    countimg=0
    countlabel+=1
    for j in i:
        countimg+=1
        labelclass.append(countlabel)
        img=cv2.imread(j)
        labelname.append(j)








data=[labelname,labelclass]
X_train = [ ]     # creating an empty array
X_valid=[]
y_train=[]
y_valid=[]
traincount=-1
for img_name in labelname:
    traincount+=1
    img = cv2.imread( img_name)
    if traincount%4==0:
        X_valid.append(img)  # storing each image in array X
        y_valid.append(labelclass[traincount])
    else:
        X_train.append(img)
        y_train.append(labelclass[traincount])
    
    
X_valid = np.array(X_valid)    # converting list to array
X_train=np.array(X_train)

dummy_y_train = np_utils.to_categorical(y_train)    # one hot encoding Classes
dummy_y_valid_images = np_utils.to_categorical(y_valid)    # one hot encoding Classes


del images, images1 ,images2
del images_path1, images_path2
del img
del labelclass,labelname



images_train = []
for i in range(0,X_train.shape[0]):
    a = resize(X_train[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    images_train.append(a)
X_train = np.array(images_train)

images_valid = []
for i in range(0,X_valid.shape[0]):
    a = resize(X_valid[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    images_valid.append(a)
X_valid = np.array(images_valid)

del a

from keras.applications.vgg19 import preprocess_input
X_train = preprocess_input(X_train, mode='tf')      # preprocessing the input data
X_valid = preprocess_input(X_valid, mode='tf')      # preprocessing the input data







base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    # include_top=False to remove the top layer

#plot_model(base_model, to_file='Model picture.pdf',show_shapes=True)



X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
print(X_train.shape, X_valid.shape)



X_train = X_train.reshape(len(X_train), 7*7*512)      # converting to 1-D
X_valid = X_valid.reshape(len(X_valid), 7*7*512)



train = X_train/X_train.max()      # centering the data
X_valid = X_valid/X_train.max()



model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='relu')) # hidden layer
model.add(Dense(2, activation='softmax'))    # output layer
#model.add(Dense(2))
#model.add(Activation('sigmoid'))


model.summary()


sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])




import datetime
start=datetime.datetime.now()
history=model.fit(train, dummy_y_train, batch_size=32,epochs=100, validation_data=(X_valid, dummy_y_valid_images))



score = model.evaluate(X_valid, dummy_y_valid_images, batch_size=32)
x_valid_output_images=model.predict(X_valid)



# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




end=datetime.datetime.now()
elapsed=end-start
plot_model(model, to_file='modelvgg19networkZJU.png')
print('training time',str(elapsed))
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Save the weights
model.save_weights('model_weights_vgg19_zju.h5')

# Save the model architecture
with open('model_architecture_vgg19_zju.json', 'w') as f:
    f.write(model.to_json())





probs = model.predict(X_valid)
# keep probabilities for the positive outcome only
#probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(dummy_y_valid_images, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
#fpr, tpr, thresholds = roc_curve(dummy_y_valid_images, probs)
## plot no skill
#plt.plot([0, 1], [0, 1], linestyle='--')
## plot the roc curve for the model
#plt.plot(fpr, tpr, marker='.')
## show the plot
#plt.show()    
