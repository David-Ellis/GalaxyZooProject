# -*- coding: utf-8 -*-
"""
Neural Network for classifying galaxy zoo images
 
"""

from __future__ import print_function
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout#, Flatten

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
  
# input image dimensions

mode = "disk" # "bulge" #  "bulge_full" #  


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


print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print()
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# %% Define Neural net and its Architecture

def create_DNN(neurons1 = 200, neurons2 = 200, dropout = 0.5, 
               activation = "softmax"):
    # instantiate model
    model = Sequential()
    # add a dense all-to-all relu layer
    if neurons1 > 0:
      model.add(Dense(neurons1,input_shape=(img_rows*img_cols,), activation='relu'))
    # add a dense all-to-all relu layer
    if neurons2 > 0:
      model.add(Dense(neurons2, activation='relu'))
    # apply dropout with rate 0.5
    if dropout > 0:
      model.add(Dropout(dropout))
    # soft-max layer
    if activation != "None":
      model.add(Dense(num_classes, activation=activation))
    return model

print('Model architecture created successfully!')


def compile_model(optimizer=keras.optimizers.Adamax(), neurons1 = 200,
                  neurons2 = 200, dropout = 0.5, activation = "softmax"):
    # create the mode
    model=create_DNN(neurons1, neurons2, dropout, activation = activation)
    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

print('Model compiled successfully and ready to be trained.')

# %% Train the model

# training parameters
batch_size = 64
epochs = 50

# create the deep neural net
model_DNN=compile_model(optimizer = keras.optimizers.Adamax())

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
plt.tight_layout()
plt.show()

# %% Plot confusion matrix

def ConfusionMatrix(model, X, Y):
  num_classes = len(Y.T)
  conMat = np.zeros((num_classes, num_classes))
  Ypred = model.predict(X)
  
  for Yr, Yp in zip(Y, Ypred):
    realIndex = np.where(Yr == max(Yr))[0][0]
    predIndex = np.where(Yp == max(Yp))[0][0]
    conMat[realIndex, predIndex] += 1
    
  return conMat
    
def plotConfusionMatrix(confusion_matrix, class_names = None, cmap = "Blues"):
  fig, ax = plt.subplots()
  ax.matshow(confusion_matrix, cmap = cmap)
  ax.set_xlabel("Predicted")
  ax.set_ylabel("Actual")
  ax.xaxis.set_label_coords(0.5, 1.2)
  
  # Add numbers
  for i in range(len(confusion_matrix)):
    for j in range(len(confusion_matrix)):
      ax.annotate(int(confusion_matrix[i, j]), (j, i), ha='center', 
                  va='center')
      #print()
  ax.set_xticks(range(num_classes))
  ax.set_yticks(range(num_classes))
  
  if class_names != None:
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names, rotation = 90, va = "center")
  
  plt.tight_layout()

print("Confusion matrix:")
class_names = ["rounded", "boxy", "no bulge"]
#class_names = ["disk", "no disk"]

conMat = ConfusionMatrix(model_DNN, X_test, Y_test)
print(conMat)
plotConfusionMatrix(conMat, class_names = class_names, cmap = "GnBu")

##############################################################################
# %% Find best number of epochs

# Seems to be different every time

retest = 50
num_epochs = 50

test_accuracies = np.zeros((retest, num_epochs))
training_accuracies = np.zeros((retest, num_epochs))

for i in range(retest):

  model_DNN=compile_model()
  
  X_train, Y_train, X_test, Y_test = splitData(X, Y, trainingPercentage)
  # train DNN and store training info in history
  history=model_DNN.fit(X_train, Y_train,
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

np.save("data/" + mode + "_num_epochs_test", 
          [num_epochs, test_accuracy_median, test_accuracy_upper, 
           test_accuracy_lower, training_accuracy_median, 
           training_accuracy_upper, training_accuracy_lower])

#%% Plot


num_epochs, test_accuracy_median, test_accuracy_upper, test_accuracy_lower, \
  training_accuracy_median, training_accuracy_upper, training_accuracy_lower\
    = np.load("data/" + mode + "_num_epochs_test.npy", allow_pickle=True)
    
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

plt.legend(loc=4)
plt.ylabel('model accuracy')
plt.xlabel('epoch')
plt.tight_layout()
plt.savefig("figures/"+ mode +"_num_epochs.pdf")



##############################################################################
# %% Find best optimiser

batch_size = 64
retest = 20
epochs = 20

optimizers = [keras.optimizers.SGD(),
              keras.optimizers.RMSprop(), keras.optimizers.Adagrad(), 
              keras.optimizers.Adadelta(), keras.optimizers.Adamax(),
              keras.optimizers.Adam(), keras.optimizers.Nadam()]

test_accuracy = np.zeros(len(optimizers))

test_accuracy_median = np.zeros((len(optimizers)))
test_accuracy_upper = np.zeros((len(optimizers)))
test_accuracy_lower = np.zeros((len(optimizers)))

for i, optimizer in enumerate(optimizers):

  accuracies = np.zeros(retest)
  for j in range(retest):
    model_DNN=compile_model(optimizer = optimizer)
    X_train, Y_train, X_test, Y_test = splitData(X, Y, trainingPercentage)
    # train DNN and store training info in history
    history=model_DNN.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(X_test, Y_test))
    
    score = model_DNN.evaluate(X_test, Y_test, verbose=0)
    accuracies[j] = score[1]
    
  print(i+1, "of", len(optimizers), "complete")
  test_accuracy_median[i] = np.median(accuracies)
  test_accuracy_upper[i] = np.percentile(accuracies, 84.1)
  test_accuracy_lower[i] = np.percentile(accuracies, 25.9)
    
# %% Plot
    
y_err1 = (test_accuracy_upper-test_accuracy_median)/test_accuracy_median
y_err2 = (test_accuracy_median - test_accuracy_lower)/test_accuracy_median
    
plt.figure(figsize = (7, 5))  
plt.bar(range(len(optimizers)),test_accuracy_median, 
        yerr = [y_err1, y_err2], capsize = 7)
labels = ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adamax", "Adam", "Nadam"]
plt.xticks(range(len(optimizers)), labels,rotation=45, size  = 12)
plt.ylabel("Accuracy on Test Data")
plt.ylim(0.94, 0.98)
plt.xlabel("Optimiser")
plt.tight_layout()



##############################################################################
# %% Find best number of neurons on first dense layer
import time
print("Finding best number of neurons on first dense layer")

# training parameters
batch_size = 64
epochs = 10
retest = 10

neurons_list1 = np.linspace(0, 400, 5)
neurons_list2 = np.array([0, 200])

for j, neurons2 in enumerate(neurons_list2):
  test_accuracy_mean = np.zeros((len(neurons_list1)))
  test_accuracy_upper = np.zeros((len(neurons_list1)))
  test_accuracy_lower = np.zeros((len(neurons_list1)))
  
  for i, neurons1 in enumerate(neurons_list1):
    tic = time.perf_counter()
    # create the deep neural net
    
    
    # train DNN and store training info in history
    accuracies = np.zeros(retest)
    print("Retest: ", end = " ")
    for j in range(retest):
      model_DNN=compile_model(optimizer = keras.optimizers.Nadam(), 
                            neurons1 = int(neurons1), neurons2 = neurons2)
      # Get new data split
      X_train, Y_train, X_test, Y_test = splitData(X, Y, trainingPercentage)
      
      history=model_DNN.fit(X_train, Y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=0,
                validation_data=(X_test, Y_test))
      
      score = model_DNN.evaluate(X_test, Y_test, verbose=0)
      accuracies[j] = score[1]
      print(j, end = " ")
    print()
    test_accuracy_mean[i] = np.median(accuracies)
    test_accuracy_upper[i] = np.percentile(accuracies, 84.1)
    test_accuracy_lower[i] = np.percentile(accuracies, 25.9)
    toc = time.perf_counter()
    print("finished {} in {:.3} seconds".format(i, toc-tic))
  np.save("data/" + mode + "_num_neurons_test_{}n2".format(neurons2), 
          [neurons_list1, test_accuracy_mean, test_accuracy_upper, 
           test_accuracy_lower])
    
  
#%% Plot

line_colors = ["b", "#ffcc00", "g"]
error_colors = ["#99bbff", "#ffe066", "#80ffaa"]

plt.figure()
for j, neurons2 in enumerate(neurons_list2):
  neurons_list1, test_accuracy_mean, test_accuracy_upper, test_accuracy_lower \
    = np.load("data/" + mode + "_num_neurons_test_{}n2.npy".format(neurons2))
  plt.fill_between(neurons_list1, test_accuracy_lower, test_accuracy_upper, 
                 color = error_colors[j], alpha = 0.5)
  plt.plot(neurons_list1, test_accuracy_mean, "-o", lw = 2, 
           color = line_colors[j], label = "$n_2 = {}$".format(neurons2))
  plt.plot(neurons_list1, test_accuracy_lower, ":", lw = 2, 
           color = error_colors[j])
  plt.plot(neurons_list1, test_accuracy_upper, ":", lw = 2, 
           color = error_colors[j])

plt.legend(loc=4, fontsize = 14)
plt.ylabel("Accuracy on Test Data")
plt.ylim(0.90, 0.98)
plt.xlabel("# Neurons in first Dense Layer, $n_1$")
plt.tight_layout()
plt.savefig("figures/"+ mode +"_num_neurons.pdf")

###########################################################################
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
import time  
  
# training parameters
batch_size = 64
epochs = 10
retest = 10

layers_list = np.array([1,2,3,4,5,6,7])

test_accuracy_mean = np.zeros((len(layers_list)))
test_accuracy_upper = np.zeros((len(layers_list)))
test_accuracy_lower = np.zeros((len(layers_list)))

for i, layers in enumerate(layers_list): 
  tic = time.perf_counter()
  # create the deep neural net

  # train DNN and store training info in history
  accuracies = np.zeros(retest)
  for j in range(retest):
    # Make new model
    model_DNN=compile_model2(optimizer = keras.optimizers.Nadam(), 
                           layers = layers, neurons = 200)
  
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
  
np.save("data/"+ mode +"_num_layers_test", 
          [layers_list, test_accuracy_mean, test_accuracy_upper, 
           test_accuracy_lower])

#%% Plot

line_colors = ["b", "#ffcc00", "g"]
error_colors = ["#99bbff", "#ffe066", "#80ffaa"]

plt.figure()

layers_list, test_accuracy_mean, test_accuracy_upper, test_accuracy_lower \
  = np.load("data/" + mode + "_num_layers_test.npy")
plt.fill_between(layers_list, test_accuracy_lower, test_accuracy_upper, 
                 color = error_colors[0], alpha = 0.5)
plt.plot(layers_list, test_accuracy_mean, "-o", lw = 2, 
           color = line_colors[0])
plt.plot(layers_list, test_accuracy_lower, ":", lw = 2, 
           color = error_colors[0])
plt.plot(layers_list, test_accuracy_upper, ":", lw = 2, 
           color = error_colors[0])

plt.ylabel("Accuracy on Test Data")
plt.ylim(0.92, 1)
plt.xlabel("# Layers")
plt.tight_layout()
plt.savefig("figures/"+ mode + "_num_layers.pdf")

##############################################################################
#%% Check impact of normalising data

# Seems to be different every time

retest = 20
num_epochs = 20

## NOT NORMALISED ## 
test_accuracies = np.zeros(retest)
training_accuracies = np.zeros(retest)

for i in range(retest):

  model_DNN=compile_model()
  
  X_train, Y_train, X_test, Y_test = splitData(X, Y, trainingPercentage)
  # train DNN and store training info in history
  history=model_DNN.fit(X_train, Y_train,
            batch_size=64,
            epochs=num_epochs,
            verbose=0,
            validation_data=(X_test, Y_test))
  test_accuracies[i] = history.history['val_accuracy'][-1]
  training_accuracies[i] = history.history['accuracy'][-1]
  print(i+1, "of", retest, "complete.")
    
  
test_accuracy_median = np.array([np.median(test_accuracies),0.0], dtype = np.float64)
test_accuracy_upper = np.array([np.percentile(test_accuracies, 84.1),0.0], dtype = np.float64)
test_accuracy_lower = np.array([np.percentile(test_accuracies, 25.9),0.0], dtype = np.float64)

training_accuracy_median = np.array([np.median(training_accuracies),0.0], dtype = np.float64)
training_accuracy_upper = np.array([np.percentile(training_accuracies, 84.1),0.0], dtype = np.float64)
training_accuracy_lower = np.array([np.percentile(training_accuracies, 25.9),0.0], dtype = np.float64)

## NORMALISED ## 
test_accuracies = np.zeros(retest)
training_accuracies = np.zeros(retest)

Xnorm = normaliseX(X)

for i in range(retest):

  model_DNN=compile_model()
  
  X_train, Y_train, X_test, Y_test = splitData(Xnorm, Y, trainingPercentage)
  # train DNN and store training info in history
  history=model_DNN.fit(X_train, Y_train,
            batch_size=64,
            epochs=num_epochs,
            verbose=0,
            validation_data=(X_test, Y_test))
  test_accuracies[i] = history.history['val_accuracy'][-1]
  training_accuracies[i] = history.history['accuracy'][-1]
  print(i+1, "of", retest, "complete.")
    
  
test_accuracy_median[1] = np.median(test_accuracies)
test_accuracy_upper[1] = np.percentile(test_accuracies, 84.1)
test_accuracy_lower[1] = np.percentile(test_accuracies, 25.9)

training_accuracy_median[1] = np.median(training_accuracies)
training_accuracy_upper[1] = np.percentile(training_accuracies, 84.1)
training_accuracy_lower[1] = np.percentile(training_accuracies, 25.9)

np.save("data/" + mode + "_normalisation_check", 
          [test_accuracy_median, test_accuracy_upper, 
           test_accuracy_lower, training_accuracy_median, 
           training_accuracy_upper, training_accuracy_lower])

#%% Plot

test_accuracy_median, test_accuracy_upper, test_accuracy_lower, \
  training_accuracy_median, training_accuracy_upper, training_accuracy_lower\
    = np.load("data/" + mode + "_normalisation_check.npy", allow_pickle=True)
    

y_err1 = (test_accuracy_upper-test_accuracy_median)/test_accuracy_median
y_err2 = (test_accuracy_median - test_accuracy_lower)/test_accuracy_median
    
plt.figure(figsize = (7, 5))  
plt.bar(range(2),test_accuracy_median, 
        yerr = [y_err1, y_err2], capsize = 7)
labels = ["not normalised", "normalised"]
plt.xticks(range(2), labels,rotation=0, size  = 18)
plt.ylabel("Accuracy on Test Data")
plt.ylim(0.9, 1)
plt.tight_layout()

##############################################################################
# %% Investigate dropout rate

# Seems to be different every time
batch_size = 64
retest = 40
epochs = 10

dropout_vals = np.linspace(0, 70, 8)
test_accuracy = np.zeros(len(dropout_vals))

test_accuracy_median = np.zeros((len(dropout_vals)))
test_accuracy_upper = np.zeros((len(dropout_vals)))
test_accuracy_lower = np.zeros((len(dropout_vals)))

for i, dropout in enumerate(dropout_vals):

  accuracies = np.zeros(retest)
  print(i, ") Retest ", end = "")
  for j in range(retest):
    model_DNN=compile_model(dropout = dropout)
    X_train, Y_train, X_test, Y_test = splitData(X, Y, trainingPercentage)
    # train DNN and store training info in history
    history=model_DNN.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(X_test, Y_test))
    
    score = model_DNN.evaluate(X_test, Y_test, verbose=0)
    accuracies[j] = score[1]
    print(j+1, end = ", ")
    
  print(": ",i+1, "of", len(dropout_vals), "complete")
  test_accuracy_median[i] = np.median(accuracies)
  test_accuracy_upper[i] = np.percentile(accuracies, 84.1)
  test_accuracy_lower[i] = np.percentile(accuracies, 25.9)


np.save("data/" + mode + "_dropout_test", 
          [dropout_vals, test_accuracy_median, test_accuracy_upper, 
           test_accuracy_lower])

# %% Plot
line_colors = ["b", "#ffcc00", "g"]
error_colors = ["#99bbff", "#ffe066", "#80ffaa"]

dropout_vals, test_accuracy_mean, test_accuracy_upper, test_accuracy_lower \
  = np.load("data/" + mode + "_dropout_test.npy")
plt.fill_between(dropout_vals, test_accuracy_lower, test_accuracy_upper, 
               color = error_colors[0], alpha = 0.5)
plt.plot(dropout_vals, test_accuracy_mean, "-o", lw = 2, 
         color = line_colors[0])
plt.plot(dropout_vals, test_accuracy_lower, ":", lw = 2, 
         color = error_colors[0])
plt.plot(dropout_vals, test_accuracy_upper, ":", lw = 2, 
         color = error_colors[0])

plt.ylabel("Accuracy on Test Data")
plt.xlabel("Dropout %")
#plt.ylim(0.75, 0.85)
plt.tight_layout()
