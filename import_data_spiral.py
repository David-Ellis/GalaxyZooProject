from imageio import imread
import csv
import numpy as np
#from PIL import Image
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from skimage.transform import downscale_local_mean
from skimage.restoration import denoise_tv_chambolle

img_labels = open("galaxy_zoo_labels.csv")
all_lines=csv.reader(img_labels)

l= 61578 #number of pictures and rows in the csv table
galaxy_id=np.zeros(l)
class1=np.zeros(l*3).reshape(l,3)
class2=np.zeros(l*2).reshape(l,2)
class3=np.zeros(l*2).reshape(l,2)
class4=np.zeros(l*2).reshape(l,2)
class5=np.zeros(l*4).reshape(l,4)
class6=np.zeros(l*2).reshape(l,2)
class7=np.zeros(l*3).reshape(l,3)
class8=np.zeros(l*7).reshape(l,7)
class9=np.zeros(l*3).reshape(l,3)
class10=np.zeros(l*3).reshape(l,3)
class11=np.zeros(l*6).reshape(l,6)
next(all_lines) # Skips first line in file containing titles

for i,row in enumerate(all_lines):
    galaxy_id[i]=row[0]
    class1[i,0],class1[i,1],class1[i,2]=row[1:4]
    class2[i,0],class2[i,1]=row[4:6]
    class3[i,0],class3[i,1]=row[6:8]
    class4[i,0],class4[i,1]=row[8:10]
    class5[i,0],class5[i,1],class5[i,2],class5[i,3]=row[10:14]
    class6[i,0],class6[i,1]=row[14:16]
    class7[i,0],class7[i,1],class7[i,2]=row[16:19]
    class8[i,0],class8[i,1],class8[i,2],class8[i,3],class8[i,4],class8[i,5],class8[i,6]=row[19:26]
    class9[i,0],class9[i,1],class9[i,2]=row[26:29]
    class10[i,0],class10[i,1],class10[i,2]=row[29:32]
    class11[i,0],class11[i,1],class11[i,2],class11[i,3],class11[i,4],class11[i,5]=row[32:38]

img_labels.close()

########################################################################################################

# in class4 search for samples with abs(yes - no) > 0.6 
# to select good training candidates
good_samp = np.zeros(l) 
for i in range(l):
    good_samp[i] = abs(class4[i,0] - class4[i,1])

good_samp = (good_samp >= 0.6)

# class_spiral contains labels to question: 
# "Is there a spiral pattern?" - Yes / No
spiral_class = class4[good_samp]
spiral_id = galaxy_id[good_samp]         #contains the corresponding galaxy IDs
spiral_ind = [i for i in range(l) if (good_samp[i] == True)] #contains the index number in the galaxy_id
spiral_num = len(spiral_ind)
print("Out of all {} samples we choose {} samples for learning our algorithms.".format(l, spiral_num))

# we create a new array of labels with binary values 0 (yes) and 1 (no) for hard classification
spiral_bin = np.zeros(spiral_num)
for i in range(spiral_num):
    spiral_bin[i] = np.argmax(spiral_class[i,:])


####################################################################################################

#global variables for reading and ploting images
scaling = True
downsampling = True
denoising = True
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
    if (denoising == True):
        gray_img = denoise_tv_chambolle(gray_img, weight = 5)
    gray_img = gray_img.flatten()
    return gray_img

def plot_gray_img(img,scale):
    img = img.reshape(scale,scale)
    plt.imshow(img, cmap = "binary_r")


# The images are split up into num_chunks batches and treated separetaley
# to prevent memory overflow
def make_filename(id):
    return "{}.jpg".format(int(id)) #returns string with galaxy id

path = "./images_training_rev1"
num_chunks = 5 #Increase this if you get a memory error
imageNames = spiral_id #over which images do you want to loop


# if images file doesn't already exist then make one
if not os.path.isdir("spiral_images"):
    os.makedirs("spiral_images")


#shows an example image before looping over all images
fig1=plt.figure()
fileName = path + "/" + make_filename(imageNames[1])
test_img1 = rgb2gray(fileName)
fig1 = plot_gray_img(test_img1, pixel_param)
plt.title("resized")
if( scaling and downsampling):          #not very nicely implemented but it fulfills its purpose
    fig2 = plt.figure()
    scaling = False
    downsampling = False
    denoising = False
    fileName = path + "/" + make_filename(imageNames[1])
    test_img2 = rgb2gray(fileName)
    plot_gray_img(test_img2, 424)
    plt.title("original")
    scaling = True
    downsampling = True
    denoising = True
    plt.show()
    plt.close(fig2)
else:
    plt.show()
plt.close(fig1)

imgsInChunk = len(imageNames)//num_chunks
print("There are {} images in each of the {} chunks".format(imgsInChunk, num_chunks))

# Loop over number of "chunks" of images
for i in range(num_chunks):
    print("Chunk {}/{}:".format(i+1, num_chunks))

    tmpStore = np.zeros((imgsInChunk, pixel_param * pixel_param + 1))
    for j in range(imgsInChunk):
        gal_index = i*imgsInChunk + j
        fileName = path + "/" + make_filename(imageNames[gal_index])
        conv_Img = rgb2gray(fileName)
        tmpStore[j,0] = imageNames[gal_index]   #first entry is the galaxy id
        tmpStore[j,1:] = conv_Img               #the remaining entries are the pixels
        #print("{} of {} complete".format(j+1, imgsInChunk), end = '\r')

    np.save("spiral_images/images{}.npy".format(i), tmpStore)
    print(" complete \n")

    
np.save("spiral_images/bin_labels.npy", spiral_bin)
np.save("spiral_images/indices.npy", spiral_ind)

    #print(tmpStore[1])
