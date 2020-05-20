##
from imageio import imread
import csv
import numpy as np
#from PIL import Image
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from skimage.transform import downscale_local_mean

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
    if i == 20:          #break is inserted here to only test program with 20 examples
        break
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

#global variables for reading and ploting images
scaling = True
downsampling = True
if( scaling == True ):
    scaling_param = 210 # only change to even numbers which can be divided by three
    pixel_param = scaling_param
else:
    scaling_param = 424 #necessary for the plot function
if(downsampling == True): # Reduces image resolution
    ds_param = 3
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
    gray_img = gray_img.flatten()
    gray_img_scaled = gray_img/max(gray_img)
    return gray_img_scaled

def plot_gray_img(img,scale):
    img = img.reshape(scale,scale)
    plt.imshow(img, cmap = "binary_r")

# The images are split up into num_chunks batches and treated separetaley
# to prevent memory overflow

path = "./example_images" 

imageNames2 = [f for f in listdir(path) if isfile(join(path, f))]

num_chunks = 1 #Increase this if you get a memory error

imageNames = imageNames2 #[:1271] # Problem hier: Sortierung passt nicht zu galaxy_id
# if images file doesn't already exist then make one
if not os.path.isdir("images"):
    os.makedirs("images")



#shows an example image before looping over all images
scaling = True
fig1=plt.figure()
fileName = path + "/" + imageNames[0]
test_img1 = rgb2gray(fileName)
fig1 = plot_gray_img(test_img1, pixel_param)
plt.title("resized")
#plt.close(fig1)
if( scaling ):
    fig2 = plt.figure()
    scaling = False
    downsampling = False
    fileName = path + "/" + imageNames[0]
    test_img2 = rgb2gray(fileName)
    plot_gray_img(test_img2, 424)
    plt.title("original")
    scaling = True
    downsampling = True
    plt.show()
    #plt.close(fig2)





# Loop over number of "chunks" of images
for i in range(num_chunks):
    print("Chunk {}/{}:".format(i+1, num_chunks))

    imgsInChunk = len(imageNames)//num_chunks

    tmpStore = np.zeros((pixel_param * pixel_param + 1, imgsInChunk))
    for j in range(imgsInChunk):
        gal_index = i*imgsInChunk + j
        fileName = path + "/" + imageNames[gal_index]
        conv_Img = rgb2gray(fileName)
        tmpStore[0] = imageNames[gal_index]
        tmpStore[1:,j] = conv_Img 
        for j in range(2):
            print(tmpStore[:,j])
        #print("{} of {} complete".format(j+1, imgsInChunk))#, end = '\r')

    np.save("images/images{}.npy".format(i), tmpStore)
    print("\n")
