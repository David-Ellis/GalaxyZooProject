# -*- coding: utf-8 -*-
"""
Disk - Neural Network

 - Starting by rewritting mnist example
 
"""

from __future__ import print_function
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")

seed=0
trainingPercentage = 0.8

np.random.seed(seed) # fix random seed
tf.random.set_seed(seed)

# %% Create train and test data sets

def plotExample(X, Y, index):
  num = Y[index]
  plt.matshow(X[index,:].reshape(45,45),cmap='binary')
  plt.annotate("Y = {}".format(num), (2,40), size = 12)
  plt.show()
  
def normaliseY(Y):
  '''Normalise classes such that sum of each group equals one'''
  for i, group in enumerate(Y):
    Y[i] = group/np.sum(group)
  return Y

def classDisribution(Y):
  '''Determine how many objects belong to which class'''
  count = np.zeros(len(Y.T))
  
  for i, nums in enumerate(Y):
    count[nums == max(nums)]+=1
  
  return count
    
def plotClassDist(Y):
  '''plot bar graph of class distribution'''
  counts = classDisribution(Y)
  plt.figure()
  plt.bar(range(len(Y.T)), counts)
  plt.xticks(range(len(Y.T)))
  plt.ylabel("# in class")
  plt.xlabel("class")
  plt.show()
  
# input image dimensions
num_classes = 2 # 10 digits

img_rows, img_cols = 45, 45 # number of pixels 

X = np.load("disk_images/images0.npy")
Y = np.load("disk_images/classes0.npy")

plotClassDist(Y)

Y = normaliseY(Y)

totalImages = len(Y)

# Get array of random non-repeating indicies of length 
# trainingPercentage*totalImages

rng = np.random.default_rng()
trainingMask = rng.choice(totalImages, size=int(trainingPercentage*totalImages), 
                          replace=False)

X_train = X[trainingMask,1:]
X_test = np.delete(X,trainingMask, axis = 0)[:,1:]

Y_train = Y[trainingMask]
Y_test = np.delete(Y,trainingMask, axis = 0)

# cast floats to single precesion
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# look at an example of data point
plotExample(X_train, Y_train, 150)
plt.show()

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print()
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# %% Define Neural net and its Architecture

from keras.models import Sequential
from keras.layers import Dense, Dropout#, Flatten
#from keras.layers import Conv2D, MaxPooling2D


def create_DNN():
    # instantiate model
    model = Sequential()
    # add a dense all-to-all relu layer
    model.add(Dense(400,input_shape=(img_rows*img_cols,), activation='relu'))
    # add a dense all-to-all relu layer
    model.add(Dense(100, activation='relu'))
    # apply dropout with rate 0.5
    model.add(Dropout(0.5))
    # soft-max layer
    model.add(Dense(num_classes, activation='softmax'))
    return model

print('Model architecture created successfully!')

# %% Choose the Optimizer and the Cost Function

def compile_model(optimizer=keras.optimizers.Adadelta()):
    # create the mode
    model=create_DNN()
    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

print('Model compiled successfully and ready to be trained.')

# %% Train the model

# training parameters
batch_size = 64
epochs = 15

# create the deep neural net
model_DNN=compile_model(optimizer = keras.optimizers.Nadam())

# train DNN and store training info in history
history=model_DNN.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test))

# %% Plot accuracy over epochs

# evaluate model
score = model_DNN.evaluate(X_test, Y_test, verbose=1)

# print performance
print()
print('Test loss: {:.3}'.format(score[0]))
print('Test accuracy: {:.3}'.format(score[1]))

# look into training history

# summarize history for accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('model accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

plt.figure()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('model loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

# %% Modify hyperparameters

epochs = 5

optimizers = [keras.optimizers.SGD(),
              keras.optimizers.RMSprop(), keras.optimizers.Adagrad(), 
              keras.optimizers.Adadelta(), keras.optimizers.Adamax(),
              keras.optimizers.Adam(), keras.optimizers.Nadam()]

test_accuracy = np.zeros(len(optimizers))
for i, optimizer in enumerate(optimizers):

  model_DNN=compile_model(optimizer = optimizer)
  
  # train DNN and store training info in history
  history=model_DNN.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, Y_test))
  
  score = model_DNN.evaluate(X_test, Y_test, verbose=1)
  test_accuracy[i] = history.history['val_accuracy'][-1]
  
# %%
plt.figure()
plt.plot(range(len(optimizers)),test_accuracy, "o")
print("Optimiser {} performs the best".format(np.arange(len(optimizers))[test_accuracy == max(test_accuracy)]))