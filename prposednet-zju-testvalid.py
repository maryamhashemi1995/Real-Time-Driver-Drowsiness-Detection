
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
from keras.utils import plot_model
import cv2
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1337)  # for reproducibility

#oc curve and auc
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.models import model_from_json

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import LocallyConnected1D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad

from six.moves import cPickle as pickle

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

pickle_files = ['open_eyes_testvalid.pickle', 'closed_eyes_testvalid.pickle']
i = 0
for pickle_file in pickle_files:
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        if i == 0:
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
            valid_dataset = save['valid_dataset']
            valid_labels = save['valid_labels']
        else:
            print("here")
            train_dataset = np.concatenate((train_dataset, save['train_dataset']))
            train_labels = np.concatenate((train_labels, save['train_labels']))
            test_dataset = np.concatenate((test_dataset, save['test_dataset']))
            test_labels = np.concatenate((test_labels, save['test_labels']))
            valid_dataset = np.concatenate((valid_dataset, save['valid_dataset']))
            valid_labels = np.concatenate((valid_labels, save['valid_labels']))
        del save  # hint to help gc free up memory
    i += 1

print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
print('valid set', valid_dataset.shape, valid_labels.shape)

batch_size = 32
nb_classes = 1
epochs =50


X_train = train_dataset
X_train = X_train.reshape((X_train.shape[0], X_train.shape[3]) + X_train.shape[1:3])
Y_train = train_labels

X_test = test_dataset
X_test = X_test.reshape((X_test.shape[0], X_test.shape[3]) + X_test.shape[1:3])
Y_test = test_labels


X_valid = valid_dataset
X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[3]) + X_valid.shape[1:3])
Y_valid = valid_labels
# print shape of data while model is building
print("{1} train samples, {4} channel{0}, {2}x{3}".format("" if X_train.shape[1] == 1 else "s", *X_train.shape))
print("{1}  test samples, {4} channel{0}, {2}x{3}".format("" if X_test.shape[1] == 1 else "s", *X_test.shape))
print("{1} valid samples, {4} channel{0}, {2}x{3}".format("" if X_valid.shape[1] == 1 else "s", *X_valid.shape))

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
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


#model.add(Convolution2D(64, (3, 3), padding='same', data_format='channels_first'))
#model.add(Activation('relu'))
#model.add(Convolution2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Activation('relu'))
#model.add(Convolution2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))



model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd , metrics=['accuracy'])


import datetime
start=datetime.datetime.now()

history=model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test,  verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])


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
plot_model(model, to_file='modelproposednetworkZJU.png')
print('training time',str(elapsed))

# Save the weights
model.save_weights('model_weights_test.h5')

# Save the model architecture
with open('model_architecture_test.json', 'w') as f:
    f.write(model.to_json())
    
    

probs1 = model.predict(X_test)
# keep probabilities for the positive outcome only
#probs = probs[:, 1]
# calculate AUC
auc1 = roc_auc_score(Y_test, probs1)
print('AUC1: %.3f' % auc1)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(Y_test, probs1)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()  


probs2 = model.predict(X_valid)
# keep probabilities for the positive outcome only
#probs = probs[:, 1]
# calculate AUC
auc2 = roc_auc_score(Y_valid, probs2)
print('AUC2: %.3f' % auc2)
# calculate roc curve
score2 = model.evaluate(X_valid,Y_valid, batch_size=32)
print(score2)


