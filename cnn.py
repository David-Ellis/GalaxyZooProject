# -*- coding: utf-8 -*-
"""
Disk - Neural Network
 
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
  
  length = int(np.sqrt(len(X[index,:])))
  plt.matshow(X[index,:].reshape(length,length),cmap='binary')
  
  
  print(num)
  label = ""
  for n in num:
    label += "{:.2},".format(n)
  label = label[:-1]
  print(label)
  plt.annotate("Y = [" + label + "]", (2,40), size = 12)
  plt.show()
  
def normaliseY(Y):
  '''Normalise classes such that sum of each group equals one'''
  for i, group in enumerate(Y):
    Y[i] = group/np.sum(group)
  return Y

def normaliseX(X):
  '''Normalise images such that mean of each image is at zero'''
  for i, image in enumerate(X):
    img_mean = np.mean(image)
    X[i] = image-img_mean/img_mean
  return X

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
  
  plt.ylim(0, 1.2*max(counts))
  # Add percentages
  for i, count in enumerate(counts):
    plt.annotate("{:.3}%".format(count/sum(counts)*100),
                 (i, count+0.05*max(counts)),ha='center')
  
  plt.tight_layout()
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
  
def reshape_data(X_train, X_test, img_rows, img_cols, verbose=0):
  # reshape data, depending on Keras backend
  if keras.backend.image_data_format() == 'channels_first':
    if verbose==1:
      print("backend = channels_first")
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
  else:
    if verbose==1:
      print("backend != channels_first")
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
  return X_train, X_test, input_shape
# input image dimensions

mode = "disk" # "bulge" #  "bulge_full" #   # 


if mode == "bulge_full":
  X = np.load("bulge_images/full_images0.npy")
  Y = np.load("bulge_images/full_classes0.npy")
else:
  X = np.load(mode + "_images/images0.npy")
  Y = np.load(mode + "_images/classes0.npy")

num_classes = len(Y.T) # 10 digits

plotClassDist(Y)

X_train, Y_train, X_test, Y_test = splitData(X, Y, trainingPercentage)

# look at an example of data point
plotExample(X_train, Y_train, 150)
plt.show()


length = int(np.sqrt(len(X[0,:])))
img_rows, img_cols = length, length # number of pixels 

X_train, X_test, input_shape = reshape_data(X_train, X_test, img_rows, img_cols)
    
print("Mode: ", mode)
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print()
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


#%% create CNN

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def create_CNN(optimizer = keras.optimizers.Adamax(), filters = 10,
               kernal_size = 5, pool1 = 2, pool2 = 2):
    # instantiate model
    model = Sequential()
    # add first convolutional layer with 10 filters (dimensionality of output space)
    model.add(Conv2D(10, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    # add 2D pooling layer
    if pool1>1:
      model.add(MaxPooling2D(pool_size=(pool1, pool1)))
    # add second convolutional layer with 20 filters
    model.add(Conv2D(20, (5, 5), activation='relu'))
    # apply dropout with rate 0.5
    model.add(Dropout(0.5))
    # add 2D pooling layer
    if pool2>1:
      model.add(MaxPooling2D(pool_size=(pool2, pool2)))
    # flatten data
    model.add(Flatten())
    # add a dense all-to-all relu layer
    model.add(Dense(20*4*4, activation='relu'))
    # apply dropout with rate 0.5
    model.add(Dropout(0.5))
    # soft-max layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model
  
##############################################################################
# %% Find best number of epochs

# training parameters
batch_size = 64
retest = 10
num_epochs = 50

test_accuracies = np.zeros((retest, num_epochs))
training_accuracies = np.zeros((retest, num_epochs))

for i in range(retest):

  model_CNN = create_CNN()
  
  X_train, Y_train, X_test, Y_test = splitData(X, Y, trainingPercentage)
  X_train, X_test, input_shape = reshape_data(X_train, X_test, img_rows, img_cols)
  
  # train DNN and store training info in history
  history=model_CNN.fit(X_train, Y_train,
            batch_size=64,
            epochs=num_epochs,
            verbose=0,
            validation_data=(X_test, Y_test))
  test_accuracies[i,:] = history.history['val_accuracy']
  training_accuracies[i,:] = history.history['accuracy']
  print(i+1, "of", retest, "complete.")
    
test_accuracy_median = np.median(test_accuracies, axis = 0)
test_accuracy_upper = np.percentile(test_accuracies, 84.1, axis = 0)
test_accuracy_lower = np.percentile(test_accuracies, 25.9, axis = 0)

training_accuracy_median = np.median(training_accuracies, axis = 0)
training_accuracy_upper = np.percentile(training_accuracies, 84.1, axis = 0)
training_accuracy_lower = np.percentile(training_accuracies, 25.9, axis = 0)

np.save("data/cnn_" + mode + "_num_epochs_test", 
          [num_epochs, test_accuracy_median, test_accuracy_upper, 
           test_accuracy_lower, training_accuracy_median, 
           training_accuracy_upper, training_accuracy_lower])

# %% Plot

plt.close("all")

num_epochs, test_accuracy_median, test_accuracy_upper, test_accuracy_lower, \
  training_accuracy_median, training_accuracy_upper, training_accuracy_lower\
    = np.load("data/cnn_" + mode + "_num_epochs_test.npy", allow_pickle=True)
    
line_colors = ["b", "#ffcc00", "g"]
error_colors = ["#99bbff", "#ffe066", "#80ffaa"]

plt.figure()

plt.fill_between(range(num_epochs), test_accuracy_lower, test_accuracy_upper, 
                 color = error_colors[0], alpha = 0.5)
plt.plot(range(num_epochs), test_accuracy_median, "-o", lw = 2, 
           color = line_colors[0], label = "test")
plt.plot(range(num_epochs), test_accuracy_lower, ":", lw = 2, 
           color = error_colors[0])
plt.plot(range(num_epochs), test_accuracy_upper, ":", lw = 2, 
           color = error_colors[0])


plt.fill_between(range(num_epochs), training_accuracy_lower, training_accuracy_upper, 
                 color = error_colors[1], alpha = 0.5)
plt.plot(range(num_epochs), training_accuracy_median, "-o", lw = 2, 
           color = line_colors[1], label = "training")
plt.plot(range(num_epochs), training_accuracy_lower, ":", lw = 2, 
           color = error_colors[1])
plt.plot(range(num_epochs), training_accuracy_upper, ":", lw = 2, 
           color = error_colors[1])

num_epochs, test_accuracy_median, test_accuracy_upper, test_accuracy_lower, \
  training_accuracy_median, training_accuracy_upper, training_accuracy_lower\
    = np.load("data/{}_num_epochs_test.npy".format(mode), allow_pickle=True)
    

plt.plot(range(num_epochs), test_accuracy_median, "-o", lw = 2, 
           color = "k", label = "dnn")


plt.xlim(-1, 50)
plt.legend(loc=2, prop = {"size": 16})
plt.ylabel('model accuracy')
plt.xlabel('epoch')
plt.tight_layout()
#plt.savefig("figures/"+ mode +"_num_epochs.pdf")

#%% Find best optimizers

batch_size = 64
retest = 5
epochs = 8

labels = ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adamax", "Adam", "Nadam"]
optimizers = [keras.optimizers.SGD(),
              keras.optimizers.RMSprop(), keras.optimizers.Adagrad(), 
              keras.optimizers.Adadelta(), keras.optimizers.Adamax(),
              keras.optimizers.Adam(), keras.optimizers.Nadam()]

test_accuracy = np.zeros(len(optimizers))

test_accuracy_median = np.zeros((len(optimizers)))
test_accuracy_upper = np.zeros((len(optimizers)))
test_accuracy_lower = np.zeros((len(optimizers)))

for i, optimizer in enumerate(optimizers):
  print("Retest: ", end = "")
  accuracies = np.zeros(retest)
  for j in range(retest):
    model_DNN= create_CNN(optimizer = optimizer)
    X_train, Y_train, X_test, Y_test = splitData(X, Y, trainingPercentage)
    X_train, X_test, input_shape = reshape_data(X_train, X_test, img_rows, img_cols)
    # train DNN and store training info in history
    model_DNN.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(X_test, Y_test))
    
    score = model_DNN.evaluate(X_test, Y_test, verbose=0)
    accuracies[j] = score[1]
    print(j+1, end = " ")
    
  print(": ", i+1, "of", len(optimizers), "complete")
  test_accuracy_median[i] = np.median(accuracies)
  test_accuracy_upper[i] = np.percentile(accuracies, 84.1)
  test_accuracy_lower[i] = np.percentile(accuracies, 25.9)
  
np.save("data/cnn_" + mode + "_optimizer_test", 
          [labels, test_accuracy_median, test_accuracy_upper, test_accuracy_lower])

#%% Plot

y_err1 = (test_accuracy_upper-test_accuracy_median)/test_accuracy_median
y_err2 = (test_accuracy_median - test_accuracy_lower)/test_accuracy_median
    
plt.figure(figsize = (7, 5))  
plt.bar(range(len(optimizers)),test_accuracy_median, 
        yerr = [y_err1, y_err2], capsize = 7)
labels = ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adamax", "Adam", "Nadam"]
plt.xticks(range(len(optimizers)), labels,rotation=45, size  = 12)
plt.ylabel("Accuracy on Test Data")
plt.ylim(0.7, 0.9)
plt.xlabel("Optimiser")
plt.tight_layout()

##############################################################################
#%% Filter number and kernal size

print("----- Number of filters and kernal size -----")
optimizer = keras.optimizers.RMSprop()

retest = 3

num_filters = [1, 5, 10, 15]
kernals = [2, 5, 10]

test_score = np.zeros((len(num_filters), len(kernals)))

for i, num_filters_i in enumerate(num_filters):
  for j, kernal_j in enumerate(kernals):
    # make array to store retest scores
    retest_scores = np.zeros(len(retest))
    print("Retest: ", end = "")
    for k in range(retest):
      # compile model
      model_DNN= create_CNN(optimizer = optimizer, filters = num_filters_i,
                            kernal_size = kernal_j)
      # choose new training + test data
      X_train, Y_train, X_test, Y_test = splitData(X, Y, trainingPercentage)
      X_train, X_test, input_shape = reshape_data(X_train, X_test, img_rows, img_cols)
      
      # train CNN
      model_DNN.fit(X_train, Y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=0,
                validation_data=(X_test, Y_test))    
      
      # Save final score
      retest_scores[k] = model_DNN.evaluate(X_test, Y_test, verbose=0)
      print(k + 1, end = " ")
    num_complete = i*len(num_filters)+j*len(kernals)+1
    num_total = len(num_filters)*len(kernals)
    print(": {} complete of {}.".format(num_complete, num_total))
    test_score[i, j] = np.mean(retest_scores)

print("\nFinal result:")
print(test_score, "\n")

print("Saving data...")
np.save("data/cnn_" + mode + "_filters_and_kernals.npy", 
          [num_filters, kernals, test_score])
print("Done.")

#%% Plotting

num_filters, kernals, test_score = np.save("data/cnn_" + mode + "_filters_and_kernals.npy")

markers = ["^", "x", "o"]
for i, kernal_i in enumerate(kernals):
  plt.plot(num_filters, test_score[:,i], marker = markers[i], 
           label = "$k = {}$".format(kernal_i))
  
plt.ylabel("Accuracy on Test Data")
plt.xlabel("Number of filters, $f$")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("figures/cnn"+ mode + "_filters.pdf")