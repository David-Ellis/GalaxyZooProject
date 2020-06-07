#global variables for reading and ploting images
scaling = True
downsampling = False
denoising = False
contouring = False
PCAnalysis = True
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

num_chunks = 5 #Increase this if you get a memory error

