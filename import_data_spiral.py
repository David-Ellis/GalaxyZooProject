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
from skimage import filters
from skimage.util import invert
from scipy.ndimage import rotate

from ini import *

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

# in class4 / class7 search for samples with abs(yes - no) > 0.6
# to select good training candidates
good_samp = np.zeros(l)
if question == 'spiral':
    for i in range(l):
        good_samp[i] = abs(class4[i,0] - class4[i,1])
if question == 'round':
    for i in range(l):
        good_samp[i] = abs(class7[i,0] + class7[i,1] + class7[i,2])
good_samp = (good_samp >= 0.6)

#additional randomly chosen smooth Galaxies. Without these images, there are 6103 Spirals and 1214 other
smooth_samp = (class1[:,0]>0.85) #5187 galaxies
class4[smooth_samp,1]= 1 #only ok for the hard classifier. circumvents the issue where both values are zero
if question == 'spiral':
    samp = smooth_samp + good_samp
if question == 'round':
    samp = good_samp
    
# class_spiral contains labels to question:
# "Is there a spiral pattern?" - Yes / No or "How round is it?"
if question == 'spiral':
    spiral_class = class4[samp]
if question == 'round':
    spiral_class = class7[samp]
spiral_id = galaxy_id[samp]         #contains the corresponding galaxy IDs
spiral_ind = [i for i in range(l) if (samp[i] == True)] #contains the index number in the galaxy_id
spiral_num = len(spiral_ind)
print("Out of all {} samples we choose {} samples for learning our algorithms.".format(l, spiral_num))

# we create a new array of labels with binary values 0 (yes) and 1 (no) for hard classification
if question == 'spiral':
    spiral_bin = np.zeros(spiral_num*2).reshape(spiral_num,2)
    num_spiral = 0 #Number of spiral galaxies
    spiral_soft = np.zeros(spiral_num*2).reshape(spiral_num,2) #soft classifier (no normalisation because the value already corresponds to the probability that it is or is not a spiral)
    for i in range(spiral_num):
        index = np.argmax(spiral_class[i,:]) #is either zero (it is a spiral) or one (no spiral)
        spiral_bin[i, index] = 1 #creates a list with [spiral, no spiral]
        spiral_soft[i,0] = spiral_class[i,0]        # [1,      0]
        spiral_soft[i,1] = spiral_class[i,1]        # [0,      1]...
        if(index == 0):
            num_spiral += 1

if question == 'round':
    cig = np.zeros(spiral_num, dtype = bool)
    ind_cig = np.array([])
    spiral_bin = np.zeros(spiral_num*3).reshape(spiral_num,3)
    num_round = 0 #Round
    num_cig = 0
    spiral_soft = np.zeros(spiral_num*3).reshape(spiral_num,3) #soft classifier (no normalisation because the value already corresponds to the probability that it is or is not a spiral)
    for i in range(spiral_num):
        index = np.argmax(spiral_class[i,:])        #is either zero (round), one (in between) or 2 (cigar shaped)
        spiral_bin[i, index] = 1                    #spiral_bin creates a list with [round, in between, cigar]
        spiral_soft[i,0] = spiral_class[i,0]        # [1, 0, 0]
        spiral_soft[i,1] = spiral_class[i,1]        # [0, 1, 0]
        spiral_soft[i,2] = spiral_class[i,2]        # [0, 0, 1]
        if(index == 0):
            num_round += 1
        if (index == 2):
            num_cig += 1
            cig[i] = True
            ind_cig = np.append(ind_cig, i)
    cig_class = spiral_class[cig]

if question == 'spiral':
    print("{} are spiral galaxies and {} not.\n".format(num_spiral,spiral_num - num_spiral))
if question == 'round':
    print("{} are round galaxies, {} are in between and {} cigar shaped.".format(num_round, spiral_num - (num_round + num_cig), num_cig))
"""
#Which Galaxies could be used to increase the no spiral labels?
print("Galaxies that are simply smooth:", galaxy_id[class1[:,0]>0.9])
no_spiral=(class4[:,0]<0.1) * (good_samp)
print("Galaxie that are a disk viewd ontop but are no spirals:", galaxy_id[no_spiral]) #look very similar too smooth and round but they have sometimes special features
#---> take additional pictures of smooth and raound galaxies (the spiral label is undefined for galaxies viewd edge on)
print(len(galaxy_id[class1[:,0]>0.9]))
"""

####################################################################################################


def rgb2gray(filepath):
    '''
    Function
    - takes filepath of image,
    - converts image to gray scale,
    - resizes, downsamples and denoises image (optional)
    - flattens image to vector of length 424*424

    returns vector
    '''
    img = imread(filepath)
    if (gray):
        gray_img = np.dot(img[...,:3], [0.299, 0.587, 0.144])
    else:
        gray_img = np.asarray(img)
        gray_img = gray_img.astype('float32')

    if (scaling == True):
        start = 424//2 - scaling_param // 2
        stop = 424//2 + scaling_param // 2
        gray_img = gray_img[start:stop, start:stop]
    if (downsampling == True):
        gray_img = downscale_local_mean(gray_img, (ds_param, ds_param))
    if (normalising == True):
        if (gray):
            gray_img = (gray_img - np.min(gray_img)) / (np.max(gray_img) - np.min(gray_img))
        else:
            gray_img[:,:,0] = (gray_img[:,:,0] - np.mean(gray_img[:,:,0])) / (np.max(gray_img[:,:,0]) - np.mean(gray_img[:,:,0]))
            gray_img[:,:,1] = (gray_img[:,:,1] - np.mean(gray_img[:,:,1])) / (np.max(gray_img[:,:,1]) - np.mean(gray_img[:,:,1]))
            gray_img[:,:,2] = (gray_img[:,:,2] - np.mean(gray_img[:,:,2])) / (np.max(gray_img[:,:,2]) - np.mean(gray_img[:,:,2]))
    if (denoising == True):
        gray_img = denoise_tv_chambolle(gray_img, weight = 0.02)
    if (contouring == True):
        try:
            thresh = filters.threshold_minimum(gray_img)
        except RuntimeError:
            thresh = filters.threshold_otsu(gray_img)
        #print(np.mean(gray_img))
        gray_img = gray_img > thresh
    gray_img = gray_img.flatten()
    return gray_img

def rotate_and_mirror(img, scale):
    if(gray):
        new_imgs = np.zeros((7, len(img)))
        img = img.reshape(scale,scale)
    else:
        img = img.reshape(scale,scale,channels)
    for k in np.arange(3):
        img = rotate(img, 90.0)
        new_imgs[k,:] = img.flatten()
    img = np.flip(img, 0)
    for k in np.arange(4):
        img = rotate(img, 90.0)
        new_imgs[k+3,:] = img.flatten()
    return new_imgs

def plot_original(filepath):
    img = imread(filepath)
    #img = np.dot(img[...,:3], [0.299, 0.587, 0.144])
    plt.imshow(img, cmap = 'binary_r')

def plot_gray_img(img,scale):
    if(gray):
        img = img.reshape(scale,scale)
        plt.imshow(img, cmap = "binary_r")
    else:
        img = img.reshape(scale,scale,channels)
        plt.imshow(img, cmap = "binary_r")


# The images are split up into num_chunks batches and treated separetaley
# to prevent memory overflow
def make_filename(id):
    return "{}.jpg".format(int(id)) #returns string with galaxy id

path = "./images_training_rev1"
#Increase num_chunks in ini.py if you get a memory error
imageNames = spiral_id #over which images do you want to loop


# if images file doesn't already exist then make one
if question == 'spiral':
    img_folder = "spiral_images"
if question == 'round':
    img_folder = "round_images"
    
if not os.path.isdir(img_folder):
    os.makedirs(img_folder)

for k in range(1,1001,500):
    #shows an example image before looping over all images
    fig1=plt.figure()
    fileName = path + "/" + make_filename(imageNames[k])
    test_img1 = rgb2gray(fileName)
    #print(test_img1.shape)
    fig1 = plot_gray_img(test_img1, pixel_param)
    plt.title("resized")
    plt.show()
    plt.close(fig1)
    if(scaling):          #not very nicely implemented but it fulfills its purpose
        fig2 = plt.figure()
        fileName = path + "/" + make_filename(imageNames[k])
        test_img2 = plot_original(fileName)

        plt.title("original")
        plt.show()
        plt.close(fig2)

# To alter the unbalanced number of round vs cigar galaxies, the cigar galaxies
# are mirrored and rotated to create 7 new images per galaxy
if question == 'round':
    cig_Store = np.zeros((7*num_cig, (pixel_param * pixel_param * channels) + 1))
    for j in range(num_cig):
        gal_index = int(ind_cig[j])
        fileName = path + "/" + make_filename(imageNames[gal_index])
        conv_Img = rgb2gray(fileName)
        rot_Img = rotate_and_mirror(conv_Img, pixel_param)
        cig_Store[7*j:(j+1)*7,0] = imageNames[gal_index]   #first entry is the galaxy id
        cig_Store[7*j:(j+1)*7,1:] = rot_Img               #the remaining entries are the pixels
    
    cig_num = len(cig_Store)
    cig_bin = np.zeros(3 * cig_num).reshape(cig_num,3)
    cig_soft = np.zeros(cig_num*3).reshape(cig_num,3)
    cig_ind = np.zeros(cig_num)
    for i in range(len(cig_class)):
        cig_bin[7*i:7*(i+1), 2] = 1                    #spiral_bin creates a list with [round, in between, cigar]
        cig_soft[7*i:7*(i+1),0] = cig_class[i,0]        # [1, 0, 0]
        cig_soft[7*i:7*(i+1),1] = cig_class[i,1]        # [0, 1, 0]
        cig_soft[7*i:7*(i+1),2] = cig_class[i,2]        # [0, 0, 1]
        cig_ind[7*i:7*(i+1)]    = ind_cig[i]

    print("The number of cigar shaped galaxies was increased from {} to {}.\n".format(num_cig, len(cig_Store)))    


imgsInChunk = len(imageNames) // num_chunks

if question == 'spiral':
    # Loop over number of "chunks" of images
    for i in range(num_chunks):
        print("Chunk {}/{}:".format(i+1, num_chunks))
        
        tmpStore = np.zeros(((imgsInChunk), (pixel_param * pixel_param * channels) + 1))
        for j in range(imgsInChunk):
            gal_index = i*imgsInChunk + j
            fileName = path + "/" + make_filename(imageNames[gal_index])
            conv_Img = rgb2gray(fileName)
            tmpStore[j,0] = imageNames[gal_index]   #first entry is the galaxy id
            tmpStore[j,1:] = conv_Img               #the remaining entries are the pixels
            #print("{} of {} complete".format(j+1, imgsInChunk), end = '\r')
            
        np.save("{}/images{}.npy".format(img_folder, i), tmpStore)
        print(" saving images complete \n")
        
        imgStart = i * imgsInChunk
        imgEnd = imgStart + imgsInChunk
        np.save("{}/bin_labels{}.npy".format(img_folder, i), spiral_bin[imgStart : imgEnd, :])
        np.save("{}/soft_labels{}.npy".format(img_folder, i), spiral_soft[imgStart : imgEnd, :])
        np.save("{}/indices{}.npy".format(img_folder, i), spiral_ind[imgStart : imgEnd, :])
                  
    print("There are {} images in each of the {} chunks".format(imgsInChunk, num_chunks))
    
    

if question == 'round':
    cigsInChunk = len(cig_Store)//num_chunks
    # Loop over number of "chunks" of images
    for i in range(num_chunks):
        print("Chunk {}/{}:".format(i+1, num_chunks))
        '''
        tmpStore = np.zeros(((imgsInChunk + cigsInChunk), (pixel_param * pixel_param * channels) + 1))
        for j in range(imgsInChunk):
            gal_index = i*imgsInChunk + j
            fileName = path + "/" + make_filename(imageNames[gal_index])
            conv_Img = rgb2gray(fileName)
            tmpStore[j,0] = imageNames[gal_index]   #first entry is the galaxy id
            tmpStore[j,1:] = conv_Img               #the remaining entries are the pixels
            #print("{} of {} complete".format(j+1, imgsInChunk), end = '\r')
        tmpStore[imgsInChunk:,:] = cig_Store[i*cigsInChunk:(i+1)*cigsInChunk,:]

        np.save("{}/images{}.npy".format(img_folder, i), tmpStore)
        print(" saving images complete \n")
        '''
        imgStart = i * imgsInChunk
        imgEnd = imgStart + imgsInChunk
        cigStart = i * cigsInChunk
        cigEnd = cigStart + cigsInChunk
        
        bin_labels = np.vstack((spiral_bin[imgStart : imgEnd, :], cig_bin[cigStart : cigEnd, :]))
        soft_labels = np.vstack((spiral_soft[imgStart : imgEnd, :], cig_soft[cigStart : cigEnd, :]))
        ind_labels = np.append(spiral_ind[imgStart : imgEnd], cig_ind[cigStart : cigEnd])
        np.save("{}/bin_labels{}.npy".format(img_folder, i), bin_labels)
        np.save("{}/soft_labels{}.npy".format(img_folder, i), soft_labels)
        np.save("{}/indices{}.npy".format(img_folder, i), ind_labels)
           
                  
    print("There are {} images in each of the {} chunks".format(imgsInChunk + cigsInChunk, num_chunks))


    #print(tmpStore[1])
