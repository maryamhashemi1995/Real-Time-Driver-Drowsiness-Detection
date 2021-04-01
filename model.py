# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:47:00 2019

@author: MaryamHashemi
"""


import numpy
import os
from keras.models import load_model


loaded_model = load_model('model.h5')
# load weights into new model
#loaded_model.load_weights("model_wights.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_valid, dummy_y_valid_images, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))