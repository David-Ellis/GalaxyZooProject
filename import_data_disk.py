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

class2=np.zeros(l*2).reshape(l,2)

next(all_lines) # Skips first line in file containing titles

for i,row in enumerate(all_lines):
    galaxy_id[i]=row[0]
    class2[i,0],class2[i,1]=row[4:6]

img_labels.close()

# %% Restrict images to "good" samples

# in class2 search for samples with abs(yes - no) > 0.6 
# to select good training candidates
good_samp = np.zeros(l) 
for i in range(l):
    good_samp[i] = abs(class2[i,0] - class2[i,1])

good_samp = (good_samp >= 0.6)

# class_disk contains labels to question: 
# "Could this be a disk viewed edge-on?" - Yes / No
disk_class = class2[good_samp]
disk_id = galaxy_id[good_samp]         #contains the corresponding galaxy IDs
disk_ind = [i for i in range(l) if (good_samp[i] == True)] #contains the index number in the galaxy_id
disk_num = len(disk_ind)
print("Out of all {} samples we choose {} samples for learning our algorithms.".format(l, disk_num))

# we create a new array of labels with binary values 0 and 1 for hard classification
disk_bin = np.zeros(disk_num)
for i in range(disk_num):
    disk_bin[i] = np.argmax(disk_class[i,:])


# %% Define desired scaling

#global variables for reading and ploting images
scaling = True
downsampling = True

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
num_chunks = 10 #Increase this if you get a memory error
imageNames = disk_id #over which images do you want to loop

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
if not os.path.isdir("disk_images"):
    os.makedirs("disk_images")

imgsInChunk = len(imageNames)//num_chunks
print("There are {} images in each of the {} chunks".format(imgsInChunk, num_chunks))

# Loop over number of "chunks" of images
for i in range(num_chunks):
    print("Chunk {}/{}:".format(i+1, num_chunks), end = " ")

    tmpStore = np.zeros((imgsInChunk, pixel_param * pixel_param + 1))
    for j in range(imgsInChunk):
        gal_index = i*imgsInChunk + j
        fileName = path + "/" + make_filename(imageNames[gal_index])
        conv_Img = rgb2gray(fileName)
        tmpStore[j,0] = imageNames[gal_index]   #first entry is the galaxy id
        tmpStore[j,1:] = conv_Img               #the remaining entries are the pixels

    np.save("disk_images/images{}.npy".format(i), tmpStore)
    print(" Complete")

