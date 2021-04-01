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
from keras.layers import Dense, InputLayer, Dropout
from keras.utils import plot_model
from sklearn.metrics import classification_report
from keras.optimizers import SGD

#oc curve and auc
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.models import model_from_json




images_path1="E:/Thesiscodes/drowsinessdetectiondeep/4-1/"
images_path2="E:/Thesiscodes/drowsinessdetectiondeep/4-2/"
images_path3="E:/Thesiscodes/drowsinessdetectiondeep/1-1/"
images_path4="E:/Thesiscodes/drowsinessdetectiondeep/1-2/"
images_path5="E:/Thesiscodes/drowsinessdetectiondeep/2-1/"
images_path6="E:/Thesiscodes/drowsinessdetectiondeep/2-2/"
images_path7="E:/Thesiscodes/drowsinessdetectiondeep/3-1/"
images_path8="E:/Thesiscodes/drowsinessdetectiondeep/3-2/"
images_path9="E:/data/dataset_B_Eye_Images/dataset_B_Eye_Images/closed/"
images_path10="E:/data/dataset_B_Eye_Images/dataset_B_Eye_Images/open/"

images1=glob.glob(images_path1+"*.jpg")
images2=glob.glob(images_path2+"*.jpg")
images3=glob.glob(images_path3+"*.jpg")
images4=glob.glob(images_path4+"*.jpg")
images5=glob.glob(images_path5+"*.jpg")
images6=glob.glob(images_path6+"*.jpg")
images7=glob.glob(images_path7+"*.jpg")
images8=glob.glob(images_path8+"*.jpg")
images9=glob.glob(images_path9+"*.jpg")
images10=glob.glob(images_path10+"*.jpg")
images=[images1,images2,images3,images4,images5,images6,images7,images8,images9,images10]


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
X_test=[]
Y_test=[]
traincount=-1
for img_name in labelname:
    traincount+=1
    img = cv2.imread( img_name)    
#    if traincount>4112 and traincount%10==0:
#        X_test.append(img)
#        Y_test.append(labelclass[traincount])
        
    if traincount%10==0:
        X_test.append(img)
        Y_test.append(labelclass[traincount])
        
    elif traincount%10==1:
        X_valid.append(img)  # storing each image in array X
        y_valid.append(labelclass[traincount])
        
    else:
        X_train.append(img)
        y_train.append(labelclass[traincount])
    
    
X_valid = np.array(X_valid)    # converting list to array
X_train=np.array(X_train)
X_test=np.array(X_test)

dummy_y_train = np_utils.to_categorical(y_train)    # one hot encoding Classes
dummy_y_valid_images = np_utils.to_categorical(y_valid)    # one hot encoding Classes
dummy_y_test = np_utils.to_categorical(Y_test)

del  images1 ,images2,images3,images4,images5,images6,images7,images8,images9,images10
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


images_test = []
for i in range(0,X_test.shape[0]):
    a = resize(X_test[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    images_test.append(a)
X_test = np.array(images_test)


del a

from keras.applications.vgg19 import preprocess_input
X_train = preprocess_input(X_train, mode='tf')      # preprocessing the input data
X_valid = preprocess_input(X_valid, mode='tf')      # preprocessing the input data
X_test = preprocess_input(X_test, mode='tf')      # preprocessing the input data







base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    # include_top=False to remove the top layer

#plot_model(base_model, to_file='Model picture.pdf',show_shapes=True)



X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
X_test = base_model.predict(X_test)
print(X_train.shape, X_valid.shape, X_test.shape)



X_train = X_train.reshape(len(X_train), 7*7*512)      # converting to 1-D
X_valid = X_valid.reshape(len(X_valid), 7*7*512)
X_test = X_test.reshape(len(X_test), 7*7*512)



train = X_train/X_train.max()      # centering the data
X_valid = X_valid/X_train.max()
X_test=X_test/X_test.max()


model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='relu')) # hidden layer
model.add(Dense(2, activation='softmax'))    # output layer


model.summary()


sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])




import datetime
start=datetime.datetime.now()
history=model.fit(train, dummy_y_train,batch_size=32, epochs=70, validation_data=(X_valid, dummy_y_valid_images))



score1 = model.evaluate(X_valid, dummy_y_valid_images, batch_size=32)
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

print('Test score1:', score1[0])
print('Test accuracy:', score1[1])

probs = model.predict(X_valid)
# keep probabilities for the positive outcome only
#probs = probs[:, 1]
# calculate AUC
auc1 = roc_auc_score(dummy_y_valid_images, probs)
print('AUC1: %.3f' % auc1)
# calculate roc curve  

score2 = model.evaluate(X_test, dummy_y_test, batch_size=32)
print(score2)

probs = model.predict(X_test)
# keep probabilities for the positive outcome only
#probs = probs[:, 1]
# calculate AUC
auc2 = roc_auc_score(dummy_y_test, probs)
print('AUC2: %.3f' % auc2)