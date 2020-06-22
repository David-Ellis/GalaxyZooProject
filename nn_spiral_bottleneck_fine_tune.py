import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D

from keras import applications
from keras.preprocessing.image import ImageDataGenerator

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

# path to the model weights files.w
top_model_weights_path = "./spiral_images/top_model_weights"
# dimensions of our images.
img_width, img_height = pixel_param, pixel_param

epochs = 10
batch_size = 16
num_classes = 2

# build the VGG16 network
vgg16 = applications.VGG16(weights='imagenet', include_top=False,  input_shape = (180,180,3)) #noch nicht sehr schön
print('Model loaded.')
print("VGG16 Layers: ")
vgg16.summary() #lists the layers and their names

for layer in vgg16.layers:
    if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool']:
        layer.trainable = True
    else:
        layer.trainable = False

top_model = keras.models.load_model("./spiral_images/top_model")

print("Top Model Layers:")
top_model.summary()

model = keras.Model(input=vgg16.input, output=top_model(vgg16.output))
model.summary()

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning


# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)


# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.SGD(lr=1e-5),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        data_format='channels_last')

# Turn down for faster convergence
#t0 = time.time()
train_size = 0.8
test_size = 0.2
print("Training fraction: {}\n".format(train_size))

history = []
for i,chunk in enumerate(range(num_chunks)):
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


    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

    history.append(model.fit(train_generator,
                            steps_per_epoch=len(X_train) / batch_size,
                            epochs=epochs,
                            #verbose=1,
                            validation_data=(X_test, y_test)))

np.save("./spiral_images/history_bottleneck_fine_tuned", history)
