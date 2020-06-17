#global variables for reading and ploting images
gray = False
scaling = True
downsampling = False
normalising = True
denoising = False
contouring = False
PCAnalysis = False
components = 500
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

if (gray):
    channels = 1
else:
    channels = 3

num_chunks = 3 #Increase this if you get a memory error
print_RAM = False
