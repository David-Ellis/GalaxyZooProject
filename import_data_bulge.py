from imageio import imread
import csv
import numpy as np

import matplotlib.pyplot as plt
import os
from skimage.transform import downscale_local_mean

# %% Functions

def rgb2gray(filepath):
    '''
    Function
    - takes filepath of image,
    - converts image to gray scale,
    - flattens image to vector of length 424*424
    - and scales this vector to values between 0 and 1

    returns vector
    '''
    img = imread(filepath)
    gray_img = np.dot(img[...,:3], [0.299, 0.587, 0.144])
    if (scaling == True):
        start = 424//2 - scaling_param // 2
        stop = 424//2 + scaling_param // 2
        gray_img = gray_img[start:stop, start:stop]
    if (downsampling == True):
        gray_img = downscale_local_mean(gray_img, (ds_param, ds_param))
    gray_img = gray_img.flatten()
    gray_img_scaled = gray_img/max(gray_img)
    return gray_img_scaled

def plot_gray_img(img,scale):
    img = img.reshape(scale,scale)
    plt.imshow(img, cmap = "binary_r")
    
def make_filename(id):
    return "{}.jpg".format(int(id)) #returns string with galaxy id
    
# %%

img_labels = open("galaxy_zoo_labels.csv")
all_lines=csv.reader(img_labels)

l= 61578 #number of pictures and rows in the csv table
galaxy_id=np.zeros(l)
class9=np.zeros(l*3).reshape(l,3)
next(all_lines) # Skips first line in file containing titles

for i,row in enumerate(all_lines):
    galaxy_id[i]=row[0]
    class9[i,0],class9[i,1],class9[i,2]=row[26:29]

img_labels.close()

# %% Restrict images to "good" samples

# in class9 search for samples with 9.1+9.2+9.3 > 0.8 
# to select good training candidates
good_samp = np.zeros(l) 
for i in range(l):
    good_samp[i] = abs(class9[i,0] + class9[i,1] + class9[i,2])

good_samp = (good_samp >= 0.8)


# class_disk contains labels to question: 
# "Could this be a disk viewed edge-on?" - Yes / No
bulge_class = class9[good_samp]
bulge_id = galaxy_id[good_samp]         #contains the corresponding galaxy IDs
bulge_ind = [i for i in range(l) if (good_samp[i] == True)] #contains the index number in the galaxy_id
bulge_num = len(bulge_ind)
print("Out of all {} samples we choose {} samples for teaching our algorithms.".format(l, bulge_num))

# we create a new array of labels with binary values 0 and 1 for hard classification
bulge_bin = np.zeros(bulge_num)
for i in range(bulge_num):
    bulge_bin[i] = np.argmax(bulge_class[i,:])


# %% Define desired scaling

#global variables for reading and ploting images
scaling = True
downsampling = False

if( scaling == True ):
    scaling_param = 180 # only change to even numbers which can be divided by ds_param
else:
    scaling_param = 424 #necessary for the plot function
    
pixel_param = scaling_param
if(downsampling == True): # Reduces image resolution
    ds_param = 4
    pixel_param = scaling_param//ds_param
else:
    ds_param = 1


# The images are split up into num_chunks batches and treated separetaley
# to prevent memory overflow

path = "./galaxy_zoo_images/images_training_rev1"
num_chunks = 1 #Increase this if you get a memory error
imageNames = bulge_id #over which images do you want to loop

# %% Example image
fig1=plt.figure()
fileName = path + "/" + make_filename(imageNames[0])
test_img1 = rgb2gray(fileName)
fig1 = plot_gray_img(test_img1, pixel_param)
plt.title("resized")
if( scaling and downsampling):          #not very nicely implemented but it fulfills its purpose
    fig2 = plt.figure()
    scaling = False
    downsampling = False
    fileName = path + "/" + make_filename(imageNames[0])
    test_img2 = rgb2gray(fileName)
    plot_gray_img(test_img2, 424)
    plt.title("original")
    scaling = True
    downsampling = True
    plt.show()
    plt.close(fig2)
else:
    plt.show()
plt.close(fig1)


# %% Loop over all "good" images, process them and save in chunks

# if images file doesn't already exist then make one
if not os.path.isdir("bulge_images"):
    os.makedirs("bulge_images")

imgsInChunk = len(imageNames)//num_chunks
print("There are {} images in each of the {} chunks".format(imgsInChunk, num_chunks))

# Loop over number of "chunks" of images
for i in range(num_chunks):
    print("Chunk {}/{}:".format(i+1, num_chunks), end = " ")

    # Temporary image storage
    tmpImgStore = np.zeros((imgsInChunk, pixel_param * pixel_param + 1))
    # Temporary class storage
    tmpClassStore = np.zeros((imgsInChunk, 3))
    for j in range(imgsInChunk):
        # determine image file name
        gal_index = i*imgsInChunk + j
        fileName = path + "/" + make_filename(imageNames[gal_index])
        # process image
        conv_Img = rgb2gray(fileName)
        # store image
        tmpImgStore[j,0] = imageNames[gal_index]   #first entry is the galaxy id
        tmpImgStore[j,1:] = conv_Img               #the remaining entries are the pixels
        # store class data
        tmpClassStore[j,:] = [bulge_class[gal_index,0], 
                              bulge_class[gal_index,1],
                              bulge_class[gal_index,2]]

    np.save("bulge_images/full_images{}.npy".format(i), tmpImgStore)
    np.save("bulge_images/full_classes{}.npy".format(i), tmpClassStore)
    print(" Complete.")


