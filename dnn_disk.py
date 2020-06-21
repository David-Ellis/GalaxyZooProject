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

import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rc('font', size=18)

plt.close()

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
  
def splitData(X, Y, trainingPercentage):
  ''' Randomly splits data into training set and test set
  '''
  Y = normaliseY(Y)
  totalImages = len(Y)
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
  
  return X_train, Y_train, X_test, Y_test
  
# input image dimensions
num_classes = 2 # 10 digits

img_rows, img_cols = 45, 45 # number of pixels 

X = np.load("disk_images/images0.npy")
Y = np.load("disk_images/classes0.npy")

plotClassDist(Y)

X_train, Y_train, X_test, Y_test = splitData(X, Y, trainingPercentage)

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


def create_DNN(neurons1, neurons2):
    # instantiate model
    model = Sequential()
    # add a dense all-to-all relu layer
    if neurons1 > 0:
      model.add(Dense(neurons1,input_shape=(img_rows*img_cols,), activation='relu'))
    # add a dense all-to-all relu layer
    if neurons2 > 0:
      model.add(Dense(neurons2, activation='relu'))
    # apply dropout with rate 0.5
    model.add(Dropout(0.5))
    # soft-max layer
    model.add(Dense(num_classes, activation='softmax'))
    return model

print('Model architecture created successfully!')

# %% Choose the Optimizer and the Cost Function

def compile_model(optimizer=keras.optimizers.Adadelta(), neurons1 = 600,
                  neurons2 = 100):
    # create the mode
    model=create_DNN(neurons1, neurons2)
    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

print('Model compiled successfully and ready to be trained.')

# %% Train the model

# training parameters
batch_size = 64
epochs = 10

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

# %% Find best optimiser

# Seems to be different every time

tf.random.set_seed(seed)
epochs = 6

optimizers = [keras.optimizers.SGD(),
              keras.optimizers.RMSprop(), keras.optimizers.Adagrad(), 
              keras.optimizers.Adadelta(), keras.optimizers.Adamax(),
              keras.optimizers.Adam(), keras.optimizers.Nadam()]

test_accuracy = np.zeros(len(optimizers))
for i, optimizer in enumerate(optimizers):

  model_DNN=compile_model(optimizer = optimizer, neurons1 = 600)
  
  # train DNN and store training info in history
  history=model_DNN.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, Y_test))
  
  score = model_DNN.evaluate(X_test, Y_test, verbose=1)
  test_accuracy[i] = score[1]
  
plt.figure()  
plt.bar(range(len(optimizers)),test_accuracy)
labels = ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adamax", "Adam", "Nadam"]
plt.xticks(range(len(optimizers)), labels)
plt.ylabel("Accuracy on Test Data")
plt.ylim(0.9, 1)
plt.xlabel("Optimiser")

print("Optimiser {} performs the best".format(np.arange(len(optimizers))[test_accuracy == max(test_accuracy)]))

# %% Find best number of neurons on first dense layer
import time

# training parameters
batch_size = 64
epochs = 5
retest = 10

neurons_list1 = np.linspace(0, 1000, 10)
neurons_list2 = np.array([0, 200, 400])

for j, neurons2 in enumerate(neurons_list2):
  test_accuracy_mean = np.zeros((len(neurons_list1)))
  test_accuracy_upper = np.zeros((len(neurons_list1)))
  test_accuracy_lower = np.zeros((len(neurons_list1)))
  
  for i, neurons1 in enumerate(neurons_list1):
    tic = time.perf_counter()
    # create the deep neural net
    model_DNN=compile_model(optimizer = keras.optimizers.Nadam(), 
                            neurons1 = int(neurons1), neurons2 = neurons2)
    
    # train DNN and store training info in history
    accuracies = np.zeros(retest)
    
    for j in range(retest):
      # Get new data split
      X_train, Y_train, X_test, Y_test = splitData(X, Y, trainingPercentage)
      
      history=model_DNN.fit(X_train, Y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=0,
                validation_data=(X_test, Y_test))
      
      score = model_DNN.evaluate(X_test, Y_test, verbose=0)
      accuracies[j] = score[1]
      
    test_accuracy_mean[i] = np.median(accuracies)
    test_accuracy_upper[i] = np.percentile(accuracies, 84.1)
    test_accuracy_lower[i] = np.percentile(accuracies, 25.9)
    toc = time.perf_counter()
    print("finished {} in {:.3} seconds".format(i, toc-tic))
  np.save("data/num_neurons_test_{}n2".format(neurons2), 
          [neurons_list1, test_accuracy_mean, test_accuracy_upper, 
           test_accuracy_lower])
    
  
#%% Plot

line_colors = ["b", "#ffcc00", "g"]
error_colors = ["#99bbff", "#ffe066", "#80ffaa"]

plt.figure()
for j, neurons2 in enumerate(neurons_list2):
  neurons_list1, test_accuracy_mean, test_accuracy_upper, test_accuracy_lower = np.load("data/num_neurons_test_{}n2.npy".format(neurons2))
  plt.fill_between(neurons_list1, test_accuracy_lower, test_accuracy_upper, 
                 color = error_colors[j], alpha = 0.5)
  plt.plot(neurons_list1, test_accuracy_mean, "-o", lw = 2, 
           color = line_colors[j], label = "$n_2 = {}$".format(neurons2))
  plt.plot(neurons_list1, test_accuracy_lower, ":", lw = 2, 
           color = error_colors[j])
  plt.plot(neurons_list1, test_accuracy_upper, ":", lw = 2, 
           color = error_colors[j])

plt.legend()
plt.ylabel("Accuracy on Test Data")
plt.ylim(0.9, 1)
plt.xlabel("# Neurons in first Dense Layer, $n_1$")
plt.tight_layout()
plt.savefig("figures/disk_num_neurons.pdf")


#%% Investigate impact of having more layers

def create_DNN2(layers, neurons):
    # instantiate model
    model = Sequential()
    # add a dense all-to-all relu layer
    if layers > 0:
      model.add(Dense(neurons,input_shape=(img_rows*img_cols,), activation='relu'))
      # layers-1 more dense all-to-all relu layer
      for i in range(layers-1):
        model.add(Dense(neurons, activation='relu'))
      
    # apply dropout with rate 0.5
    model.add(Dropout(0.5))
    # soft-max layer
    model.add(Dense(num_classes, activation='softmax'))
    return model

print('Model architecture created successfully!')

# Choose the Optimizer and the Cost Function
def compile_model2(optimizer=keras.optimizers.Adadelta(), layers = 2,
                  neurons = 200):
    # create the mode
    model=create_DNN2(layers, neurons)
    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
  
#%%
# training parameters
batch_size = 64
epochs = 5
retest = 10

layers_list = np.array([1,2,3,4,5,6,7])

test_accuracy_mean = np.zeros((len(layers_list)))
test_accuracy_upper = np.zeros((len(layers_list)))
test_accuracy_lower = np.zeros((len(layers_list)))

for i, layers in enumerate(layers_list): 
  tic = time.perf_counter()
  # create the deep neural net
  model_DNN=compile_model2(optimizer = keras.optimizers.Nadam(), 
                           layers = layers, neurons = 200)
  
  # train DNN and store training info in history
  accuracies = np.zeros(retest)
  for j in range(retest):
    # Get new data split
    X_train, Y_train, X_test, Y_test = splitData(X, Y, trainingPercentage)
    
    history=model_DNN.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(X_test, Y_test))
    score = model_DNN.evaluate(X_test, Y_test, verbose=0)
    accuracies[j] = score[1]
  
  #print(i, accuracies, np.median(accuracies))   
  test_accuracy_mean[i] = np.median(accuracies)
  test_accuracy_upper[i] = np.percentile(accuracies, 84.1)
  test_accuracy_lower[i] = np.percentile(accuracies, 25.9)
  toc = time.perf_counter()
  print("finished {} in {:.3} seconds".format(i, toc-tic))
  
np.save("data/disk_num_layers_test", 
          [layers_list, test_accuracy_mean, test_accuracy_upper, 
           test_accuracy_lower])

#%% Plot

line_colors = ["b", "#ffcc00", "g"]
error_colors = ["#99bbff", "#ffe066", "#80ffaa"]

plt.figure()

layers_list, test_accuracy_mean, test_accuracy_upper, test_accuracy_lower \
  = np.load("data/num_layers_test.npy")
plt.fill_between(layers_list, test_accuracy_lower, test_accuracy_upper, 
                 color = error_colors[0], alpha = 0.5)
plt.plot(layers_list, test_accuracy_mean, "-o", lw = 2, 
           color = line_colors[0])
plt.plot(layers_list, test_accuracy_lower, ":", lw = 2, 
           color = error_colors[0])
plt.plot(layers_list, test_accuracy_upper, ":", lw = 2, 
           color = error_colors[0])

plt.ylabel("Accuracy on Test Data")
plt.ylim(0.95, 1)
plt.xlabel("# Layers")
plt.tight_layout()
plt.savefig("figures/disk_num_layers.pdf")