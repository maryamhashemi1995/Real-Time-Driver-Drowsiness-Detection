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
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
from keras.utils import plot_model
from sklearn.metrics import classification_report
from keras.optimizers import SGD




images_path1="E:/../close-straight/1-1/"
images_path2="E:/.../Open-straight/1-2/"
images_path3="E:/.../close-straight/2-1/"
images_path4="E:/.../Open-straight/2-2/"
images_path5="../close-straight/3-1/"
images_path6="E:/.../Open-straight/3-2/"
images_path7="E:/.../close-straight/4-1/"
images_path8="E:/.../Open-straight/4-2/"
images1=glob.glob(images_path1+"*.jpg")
images2=glob.glob(images_path2+"*.jpg")
images3=glob.glob(images_path3+"*.jpg")
images4=glob.glob(images_path4+"*.jpg")
images5=glob.glob(images_path5+"*.jpg")
images6=glob.glob(images_path6+"*.jpg")
images7=glob.glob(images_path7+"*.jpg")
images8=glob.glob(images_path8+"*.jpg")
images=[images1,images2,images3,images4,images5,images6,images7,images8]


labelname=[]
labelclass=[]
index=0
for i in images:
    index+=1
    countimg=0
    if index%2==0:
        countlabel=1
    else:
        countlabel=0
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


del  images1 ,images2
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

from keras.applications.vgg16 import preprocess_input
X_train = preprocess_input(X_train, mode='tf')      # preprocessing the input data
X_valid = preprocess_input(X_valid, mode='tf')      # preprocessing the input data







base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    # include_top=False to remove the top layer

#plot_model(base_model, to_file='Model picture.pdf',show_shapes=True)



X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
print(X_train.shape, X_valid.shape)



X_train = X_train.reshape(len(X_train), 7*7*512)      # converting to 1-D
X_valid = X_valid.reshape(len(X_valid), 7*7*512)



train = X_train/X_train.max()      # centering the data
X_valid = X_valid/X_train.max()


#adding extra layers to our transfer learning model
model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='relu')) # hidden layer
model.add(Dense(2, activation='softmax'))    # output layer


model.summary()


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])




import datetime
start=datetime.datetime.now()
history=model.fit(train, dummy_y_train,batch_size=10, epochs=25, validation_data=(X_valid, dummy_y_valid_images))



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
print('training time',str(elapsed))
base_model.save('model.h5')
base_model.save_weights('model_weights.h5')
print('Test score:', score[0])
print('Test accuracy:', score[1])

