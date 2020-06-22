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

from ini import pixel_param, print_RAM, channels, num_chunks

if(print_RAM):
    #to print the current memory usage
    import tracemalloc
    import print_memory
    tracemalloc.start()

def plotExample(X, index):
    #print(X[index].shape)
    plt.imshow(X[index].reshape(pixel_param, pixel_param, channels), cmap = "binary_r")
    plt.show()


from keras import applications
from keras.preprocessing.image import ImageDataGenerator

batch_size = 32

def save_bottlebeck_features():
    #datagen = ImageDataGenerator(rescale=1. / 255)

    datagen = ImageDataGenerator()
            #rotation_range=90,
            #width_shift_range=0.2,
            #height_shift_range=0.2,
            #shear_range=0.2,
            #zoom_range=0.2,
            #horizontal_flip=True,
            #fill_mode='nearest',
            #data_format='channels_last')

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet', input_shape = input_shape)

    """generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)"""

    generator = datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, len(X_train) / batch_size) #picks in each step (number of steps: len(X_train) / batch_size) $batch_size pictures from generator and predicts them

    datagen = ImageDataGenerator()

    generator = datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, len(X_test) / batch_size)

    return bottleneck_features_train, bottleneck_features_validation

# Turn down for faster convergence
t0 = time.time()
train_size = 0.8
test_size = 0.2
print("Training fraction: {}\n".format(train_size))

train_features = []
validation_features = []
train_labels = []
validation_labels = []
for chunk in range(num_chunks):
    ### load galaxy data from spiral_images/
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

    # reshape data, depending on Keras backend
    img_rows = pixel_param
    img_cols = pixel_param
    if keras.backend.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], channels, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
        print("channels_first")
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)
        print("channels_last")

    #plotExample(X_train, 10)

    train, validation = save_bottlebeck_features()
    train_features      += [t for t in train]
    validation_features += [v for v in validation]
    train_labels        += [t for t in y_train]
    validation_labels   += [v for v in y_test]

train_features      = np.array(train_features)
validation_features = np.array(validation_features)
train_labels        = np.array(train_labels)
validation_labels   = np.array(validation_labels)

"""
l_train = 0
l_validation = 0
for i in range(len(train_features)):
    l_train += len(train_features[i])
    l_validation += len(validation_features[i])

all_train_features = np.zeros(l_train*5*5*512).reshape(l_train,5,5,512)
all_validation_features = np.zeros(l_validation*5*5*512).reshape(l_validation,5,5,512)

l_aux_train = 0
l_aux_validation = 0
for i in range(len(train_features)):
    curr_l_train = l_aux_train + len(train_features[i])
    all_train_features[l_aux_train:curr_l_train] = train_features[i]
    l_aux_train = curr_l_train

    curr_l_validation = l_aux_validation + len(validation_features[i])
    all_validation_features[l_aux_validation:curr_l_validation] = validation_features[i]
    l_aux_validation = curr_l_validation
"""

np.save("./spiral_images/bottleneck_features_train_bin_labels", train_features)
np.save("./spiral_images/bottleneck_features_validation_bin_labels", validation_features)
np.save("./spiral_images/bottleneck_labels_train_bin_labels", train_labels)
np.save("./spiral_images/bottleneck_labels_validation_bin_labels", validation_labels)
