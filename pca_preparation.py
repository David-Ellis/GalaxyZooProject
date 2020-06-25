#######################################################
#                                                     #
#  HARD CLASSIFICATION FOR SPIRAL GALAXIES USING PCA  #
#                                                     #
#######################################################


import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
#from sklearn.utils import check_random_state
import seaborn as sns
from sklearn import metrics
from ini import *

#print(__doc__)

# Turn down for faster convergence
t0 = time.time()
train_size = 0.8
test_size = 0.2
pixel = pixel_param
num_chunks = num_chunks
components = components
if question == 'spiral':
    img_folder = "spiral_images"
if question == 'round':
    img_folder = "round_images"
spiral_ind = np.load("{}/indices0.npy".format(img_folder))
spiral_num = len(spiral_ind) * num_chunks
batch_size = int((spiral_num * train_size / num_chunks)//1)
print("In total there are {} samples.\n".format(spiral_num))
print("Training fraction: {}\n".format(train_size))

# for shuffling data in a reproducable manner
random_state = 2

###############################################################################

### load galaxy data from spiral_images/ 
def get_minibatch(num_batches):
    for num in range(num_batches):
        print("Loading chunk {}...\n".format(num))
        batch_path = '{}/images{}.npy'.format(img_folder, num)
        xx = np.load(batch_path)
        # Bin or soft labels?
        yy = np.load('{}/soft_labels{}.npy'.format(img_folder, num))
        imgsInChunk = xx.shape[0]
        print("Processing chunk {} containing {} pictures...\n".format(num, imgsInChunk))
        yield xx, yy

### transform data with PCA and print reconstructed images
def principal_components(xtrain, xtest, components, chunk):
    # trasnform and reduce components
    xtrain = pca.transform(xtrain)
    xtest = pca.transform(xtest)
    
    # inverse transform for plotting
    proj_test = pca.inverse_transform(xtest[0:4])
    proj_test = scaler.inverse_transform(proj_test)
    if chunk == 0:
        plt.figure('fig1')
        fig, ax = plt.subplots(2, 4, figsize=(5, 4),
                               subplot_kw={'xticks':[], 'yticks':[]},
                               gridspec_kw=dict(hspace=0.1, wspace=0.1))
        for i in range(4):
            ax[0, i].imshow(X_test0[i].reshape(pixel, pixel), cmap='binary_r')
            ax[1, i].imshow(proj_test[i].reshape(pixel, pixel), cmap='binary_r')
            ax[0, 0].set_ylabel('full-dim\ninput')
            ax[1, 0].set_ylabel('{}-dim\nreconstruction'.format(components))
        plt.show()
        plt.close()    
    return xtrain, xtest




###############################################################################

# Incremental PCA fitting
pca = IncrementalPCA(n_components=components, batch_size = batch_size)
scaler = StandardScaler()


minibatch = get_minibatch(num_chunks)

X_train, y_train = next(minibatch)
y_train = [np.argmax(i) for i in y_train]
# pick training and test data sets 
X_train, *_ = train_test_split(X_train,y_train,train_size=train_size,test_size
                                                          =test_size, random_state = random_state)
# We need to split the uploaded X_in arrays into the galaxy ID vector and the image data
X_train = X_train[:,1:]
# Scale data to zero mean and unit variance
X_train = scaler.fit_transform(X_train)
# Partially fit PCA to data
pca.fit(X_train)
print("First batch fitted with PCA.\n")

# Defining final arrays for training and testing data
X_train = np.array([]) 
X_test = np.array([]) 
y_train = np.array([])
y_test = np.array([])
#id_train = np.array([])
#id_test = np.array([])

# Load data into final array after scaling and transforming
minibatch = get_minibatch(num_chunks)
for chunk in range(num_chunks):
    X_train_in, y_train_in = next(minibatch)
    # For hard labels
    #y_train_in = [np.argmax(i) for i in y_train_in]
    # pick training and test data sets 
    X_train_in, X_test_in, y_train_in, y_test_in = train_test_split(X_train_in,y_train_in,train_size=train_size,test_size
                                                    =test_size, random_state = random_state)
    
    # We need to split the uploaded X_in arrays into the galaxy ID vector and the image data
    id_train_in = X_train_in[:,0]
    X_train_in = X_train_in[:,1:]
    id_test_in = X_test_in[:,0]
    X_test_in = X_test_in[:,1:]
    
    # Save original data for final plots
    if chunk == 0:
            X_test0 = np.copy(X_test_in[0:100,:])
    
    # Scale data to zero mean and unit variance
    X_train_in = scaler.fit_transform(X_train_in)
    X_test_in = scaler.transform(X_test_in)
        
    if PCAnalysis == True:
        X_train_in, X_test_in = principal_components(X_train_in, X_test_in, components, chunk)
    
    X_train = np.append(X_train, X_train_in)
    X_test = np.append(X_test, X_test_in) 
    y_train = np.append(y_train, y_train_in)
    y_test = np.append(y_test, y_test_in)
    #id_train = np.append(id_train, id_train_in)
    #id_test = np.append(id_test, id_test_in)

scaler2 = StandardScaler()
X_train = X_train.reshape((-1, components))
print(np.var(X_train), np.mean(X_train))

X_train = scaler2.fit_transform(X_train)
X_test = X_test.reshape((-1, components))
X_test = scaler2.transform(X_test)
print(np.var(X_test), np.mean(X_test))

# only for soft labels 
y_train = y_train.reshape((2, -1))
y_test = y_test.reshape((2, -1))

tsne_analysis = False
if (tsne_analysis == True):
    is_spiral = y_train[0:2000] == 0
    no_spiral = y_train[0:2000] == 1   
    tsne_model = TSNE(n_components = 2, random_state = random_state, n_iter = 1000, perplexity = 100)
    tsne_data = tsne_model.fit_transform(X_train[0:2000,:])
    plt.figure('fig_tsne')
    plt.scatter(tsne_data[is_spiral, 0], tsne_data[is_spiral, 1], c = 'r')
    plt.scatter(tsne_data[no_spiral, 0], tsne_data[no_spiral, 1], c = 'b')
    plt.xlim((-5,5))
    plt.ylim((-5,5))
    plt.axis("off")
    plt.show()
    plt.close('fig_tsne')

print("The training data has {} samples of {} features each. \n".format(X_train.shape[0], X_train.shape[1]))

np.save("{}/xtrain.npy".format(img_folder), X_train)
np.save("{}/xtest.npy".format(img_folder), X_test)
np.save("{}/ytrain.npy".format(img_folder), y_train)
np.save("{}/ytest.npy".format(img_folder), y_test)
np.save("{}/xtest0.npy".format(img_folder), X_test0)


if PCAnalysis == True:
    # plot mean galaxy 
    fig_mean = plt.figure()
    plt.imshow(pca.mean_.reshape((pixel, pixel)))
    plt.axis("off")
    plt.show()
    plt.close(fig_mean)
    # plot eigengalaxies (example)
    fig_eigen = plt.figure()
    plt.imshow(pca.components_[1].reshape((pixel, pixel)))
    plt.axis("off")
    plt.show()
    plt.close(fig_eigen)
    # plot explained variance ratio
    fig_var = plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    #plt.xlabel('number of components')
    #plt.ylabel('cumulative explained variance')
    plt.plot(np.arange(components), np.ones(components) * 0.90)
    plt.plot(np.arange(components), np.ones(components) * 0.95)
    plt.ylim((0.47, 1.0))
    plt.show()
    plt.close(fig_var)

