##
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

file=open("../galaxy_zoo_labels.csv")
all_lines=csv.reader(file)

l=61578 #number of pictures and rows in the csv table
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
next(all_lines)

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
file.close()

def image2pixelarray(filepath):
    """
    Parameters
    ----------
    filepath : str
        Path to an image file

    Returns
    -------
    list
        A list of lists which make it simple to access the greyscale value by
        im[y][x]
    """
    im = Image.open(filepath).convert('L')
    (width, height) = im.size
    greyscale_map = list(im.getdata())
    greyscale_map = np.array(greyscale_map)/max(greyscale_map) #all pixel values are between 0 and 1
    #greyscale_map = greyscale_map.reshape((height, width))    #in this case the shape is always 424x424
    im.close()
    return greyscale_map


batch_number=100
for batch_id in range(0,batch_number,200):
    start=batch_id*l//batch_number
    if(batch_id < batch_number-1):
        end=(batch_id+1)*l//batch_number
    else:
        end=l

    Galaxy_pics=np.zeros((end-start)*424*424).reshape(end-start,424*424)
    for i,id in enumerate(galaxy_id[start:end]):
        pic=image2pixelarray("../images_training_rev1/{}.jpg".format(int(id)))
        Galaxy_pics[i,:]=pic

        if(pic.shape!=(424*424,)):
            print("Error image shape is different")

    np.save("Galaxy_pics_batch{}".format(batch_id),Galaxy_pics)


example=Galaxy_pics[10].reshape(424,424)
fig=plt.figure(figsize=(6,6))
plt.imshow(example, origin='lower', cmap='viridis')
plt.xticks([])
plt.yticks([])
plt.show()
plt.close(fig)
