import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D

from ini import pixel_param, print_RAM, channels, num_chunks

def plot_history(history):
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

def train_top_model():
    num_classes = 2

    train_data = np.load("./spiral_images/bottleneck_features_train_bin_labels.npy")
    train_labels = np.load("./spiral_images/bottleneck_labels_train_bin_labels.npy")

    validation_data = np.load("./spiral_images/bottleneck_features_validation_bin_labels.npy")
    validation_labels = np.load("./spiral_images/bottleneck_labels_validation_bin_labels.npy")

    print(train_data.shape)
    #print(train_labels)
    print(train_data.shape[1:])
    inputs = keras.Input(shape=(5,5,512), name='input')
    x = Flatten()(inputs)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='sigmoid', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="top_model")

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save(model_path)

    score = model.evaluate(validation_data, validation_labels, verbose=1)

    # print performance
    print()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return history

model_path = "./spiral_images/top_model"
epochs = 10
batch_size = 16

history = train_top_model()
plot_history(history)
