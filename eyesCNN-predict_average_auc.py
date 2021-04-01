
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
from keras.utils import plot_model
import cv2

np.random.seed(1337)  # for reproducibility



from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad
import glob
from six.moves import cPickle as pickle
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from skimage.transform import resize
import datetime
#oc curve and auc
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.models import model_from_json




#file axi ke mikhay bekhuni
img=cv2.imread('E:/Thesiscodes/drowsinessdetectiondeep/eyecrop/1.jpg',0)

##############################################################
#inja preprocess hayee hast ke axe avali ra amade mikonim ke varede shabake beshe. dar har shabake ba tavajoh be preprocess ha in code ha tagheer mikone
#    img=resize(img,preserve_range=True, output_shape=(45,65))
img=resize(img,preserve_range=True, output_shape=(24,24))
img=img/255
img= img[ np.newaxis,...]
img= img[ np.newaxis,...]
###########################################################

#     Model reconstruction from JSON file
with open('model_architecture_test.json', 'r') as f:
#    model = model_from_json(f.read())

#     Load weights into the new model
model.load_weights('model_weights_test.h5')

####################################################
start=datetime.datetime.now()
#predict=model.predict(img, batch_size=1, verbose=1, steps=None)
predict=model.predict(img,batch_size=32,verbose=1,steps=None)
end=datetime.datetime.now()
elapsed=end-start



print(totalelapsed/100)
