import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import readchar

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D

from ini import pixel_param, print_RAM

if(print_RAM):
    #to print the current memory usage
    import tracemalloc
    import print_memory
    tracemalloc.start()

def plotExample(X, index):
  plt.matshow(X[index,:].reshape(pixel_param, pixel_param), cmap='binary')
  plt.show()

# Turn down for faster convergence
t0 = time.time()
train_size = 0.8
test_size = 0.2
print("Training fraction: {}\n".format(train_size))

### load galaxy data from spiral_images/
chunk = 0 #which of the created chunks should be used
X_train = np.load('spiral_images/images{}.npy'.format(chunk))
y = np.load('spiral_images/bin_labels.npy') #hard classifier
#y = np.load('spiral_images/soft_labels.npy') #soft classifier
imgsInChunk = X_train.shape[0]
print("Processing chunk {} containing {} pictures...\n".format(chunk, imgsInChunk))

imgStart = chunk * imgsInChunk
imgEnd = imgStart + imgsInChunk
y = y[imgStart : imgEnd]

# for shuffling data in a reproducable manner
random_state = 1

# pick training and test data sets
X_train, X_test, y_train, y_test = train_test_split(X_train,y,train_size=train_size,test_size
                                                    =test_size, random_state = random_state)

# We need to split the uploaded X_in arrays into the galaxy ID vector and the image data
id_train = X_train[:,0]
X_train = X_train[:,1:]
id_test = X_test[:,0]
X_test = X_test[:,1:]
print("The training data has {} samples of {} features each. \n".format(X_train.shape[0], X_train.shape[1]))

if(print_RAM):
    #print(current memory usage)
    snapshot = tracemalloc.take_snapshot()
    print_memory.display_memory(snapshot)
    print("Continue?[y/n]")
    c = readchar.readchar()
    if (c != 'y'):
        sys.exit(0)


def create_DNN():
    img_rows = pixel_param
    img_cols = pixel_param
    num_classes = 2 #Spiral or no spiral
    optimizer = keras.optimizers.Adam()

    # instantiate model
    model = Sequential()
    # add a dense all-to-all relu layer
    model.add(Dense(1000,pe=(img_rows*img_cols,), activation='relu'))
    # add a dense all-to-all relu layer
    model.add(Dense(400, activation='relu'))
    # add a dense all-to-all relu layer
    model.add(Dense(400, activation='relu'))
    # add a dense all-to-all relu layer
    model.add(Dense(400, activation='relu'))
    # add a dense all-to-all relu layer
    model.add(Dense(400, activation='relu'))
    # add a dense all-to-all relu layer
    model.add(Dense(400, activation='relu'))
    # apply dropout with rate 0.5
    model.add(Dropout(0.5))
    # soft-max layer
    model.add(Dense(num_classes, activation='softmax'))

    # compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model



def create_CNN():
    img_rows = pixel_param
    img_cols = pixel_param
    num_classes = 2 #Spiral or no spiral
    optimizer = keras.optimizers.Adam()

    # instantiate model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

    # soft-max layer
    #model.add(Dense(num_classes))
    #model.add(Activation('softmax'))

    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


#############
#preprocessing data
#############
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        data_format='channels_last')

# reshape data, depending on Keras backend
img_rows = pixel_param
img_cols = pixel_param
if keras.backend.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    print("channels_first")
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    print("channels_last")

plotExample(X_train, 100)

"""
for x_batch, y_batch in datagen.flow(aux_X, aux_y, batch_size=2): #picks randomly $batch_size pictures from aux_X and augments them randomly
    print("neues batch")
    for i in range(len(x_batch)):
        print("Bild ",i)
        print(x_batch.shape) #has a shape of (samples, channels, hight, width)
        plotExample(x_batch, i)
"""

"""
##########
#DNN
##########

# training parameters
#print(len(X_train))
batch_size = 64
epochs = 5

# create the deep neural net
model_DNN = create_DNN()

#saves an image of the NN model
if not os.path.isdir("NN_models"):
    os.makedirs("NN_models")
keras.utils.plot_model(
    model_DNN,
    to_file="./NN_models/DNN_model.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
)

# train DNN and store training info in history
history = model_DNN.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))

# evaluate model
score = model_DNN.evaluate(X_test, y_test, verbose=1)

# print performance
print()
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# look into training history
print(history.history)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('model accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('model loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
"""

#########
#CNN
#########

# training parameters
batch_size = 32
epochs = 10

# create the deep neural net
model_CNN = create_CNN()

#saves an image of the NN model
if not os.path.isdir("NN_models"):
    os.makedirs("NN_models")
keras.utils.plot_model(
    model_CNN,
    to_file="./NN_models/CNN_model.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
)

# train DNN and store training info in history
history = model_CNN.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / batch_size,
                        epochs=epochs,
                        #verbose=1,
                        validation_data=(X_test, y_test))

# evaluate model
score = model_CNN.evaluate(X_test, y_test, verbose=1)

# print performance
print()
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# look into training history
print(history.history)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('model accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('model loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
