
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
from keras.utils import plot_model
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import losses
np.random.seed(1337)  # for reproducibility



from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad

from six.moves import cPickle as pickle

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

pickle_files = ['E:/Thesiscodes/drowsinessdetectiondeep/DrowsyDriverDetection-master/open_eyes_straight.pickle', 'E:/Thesiscodes/drowsinessdetectiondeep/DrowsyDriverDetection-master/close_eyes_straight.pickle']
i = 0
for pickle_file in pickle_files:
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        if i == 0:
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
        else:
            print("here")
            train_dataset = np.concatenate((train_dataset, save['train_dataset']))
            train_labels = np.concatenate((train_labels, save['train_labels']))
            test_dataset = np.concatenate((test_dataset, save['test_dataset']))
            test_labels = np.concatenate((test_labels, save['test_labels']))
        del save  # hint to help gc free up memory
    i += 1

print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 32
nb_classes = 1
epochs =30


X_train = train_dataset
X_train = X_train.reshape((X_train.shape[0], X_train.shape[3]) + X_train.shape[1:3])
Y_train = train_labels

X_test = test_dataset
X_test = X_test.reshape((X_test.shape[0], X_test.shape[3]) + X_test.shape[1:3])
Y_test = test_labels

# print shape of data while model is building
print("{1} train samples, {4} channel{0}, {2}x{3}".format("" if X_train.shape[1] == 1 else "s", *X_train.shape))
print("{1}  test samples, {4} channel{0}, {2}x{3}".format("" if X_test.shape[1] == 1 else "s", *X_test.shape))

# input image dimensions
_, img_channels, img_rows, img_cols = X_train.shape

# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, (3, 3), padding='same',
                        input_shape=(img_channels, img_rows, img_cols),data_format='channels_first'))
model.add(Activation('relu'))
model.add(Convolution2D(24, (3, 3), data_format='channels_first'),)


model.add(Flatten())
model.add(Dense(512))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])


import datetime
start=datetime.datetime.now()

history=model.fit(X_train, Y_train,batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test,  verbose=1)





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
print('Test score:', score[0])
print('Test accuracy:', score[1])


