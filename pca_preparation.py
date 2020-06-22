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
#from sklearn.manifold import TSNE
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
        yy = np.load('{}/bin_labels{}.npy'.format(img_folder, num))
        imgsInChunk = xx.shape[0]
        print("Processing chunk {} containing {} pictures...\n".format(num, imgsInChunk))
        yield xx, yy

### transform data with PCA and print reconstructed images
def principal_components(xtrain, xtest, components, chunk):
    # trasnform and reduce components
    xtrain = ipca.transform(xtrain)
    xtest = ipca.transform(xtest)
    
    # inverse transform for plotting
    proj_test = ipca.inverse_transform(xtest[0:5])
    if chunk == 0:
        plt.figure('fig1')
        fig, ax = plt.subplots(2, 4, figsize=(10, 4),
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
ipca = IncrementalPCA(n_components=components, batch_size = batch_size)
scaler = StandardScaler()


minibatch = get_minibatch(num_chunks)
for chunk in range(num_chunks):
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
    ipca.partial_fit(X_train)
    print("Batch {} fitted with PCA.\n".format(chunk))

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
    y_train_in = [np.argmax(i) for i in y_train_in]
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

tsne_analysis = False
if (tsne_analysis == True):
    is_spiral = y_train[0:2000] == 0
    no_spiral = y_train[0:2000] == 1   
    tsne_model = TSNE(n_components = 2, random_state = random_state, n_iter = 1000, perplexity = 100)
    tsne_data = tsne_model.fit_transform(X_train[0:2000,:])
    plt.figure('fig_tsne')
    plt.scatter(tsne_data[is_spiral, 0], tsne_data[is_spiral, 1], c = 'r')
    plt.scatter(tsne_data[no_spiral, 0], tsne_data[no_spiral, 1], c = 'b')
    plt.show()
    plt.close('fig_tsne')

scaler2 = StandardScaler()
X_train = X_train.reshape((-1, components))
print(np.var(X_train), np.mean(X_train))
X_train = scaler2.fit_transform(X_train)
X_test = X_test.reshape((-1, components))
X_test = scaler2.transform(X_test)
print(np.var(X_test), np.mean(X_test))

print("The training data has {} samples of {} features each. \n".format(X_train.shape[0], X_train.shape[1]))

np.save("{}/xtrain.npy".format(img_folder), X_train)
np.save("{}/xtest.npy".format(img_folder), X_test)
np.save("{}/ytrain.npy".format(img_folder), y_train)
np.save("{}/ytest.npy".format(img_folder), y_test)
np.save("{}/xtest0.npy".format(img_folder), X_test0)


if PCAnalysis == True:
    # plot mean galaxy 
    fig_mean = plt.figure()
    plt.imshow(ipca.mean_.reshape((pixel, pixel)), cmap = plt.cm.bone)
    plt.close(fig_mean)
    # plot eigengalaxies (example)
    fig_eigen = plt.figure()
    plt.imshow(ipca.components_[0].reshape((pixel, pixel)))
    plt.close(fig_eigen)
    # plot explained variance ratio
    fig_var = plt.figure()
    plt.plot(np.cumsum(ipca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.plot(np.arange(components), np.ones(components) * 0.90)
    plt.ylim((0.47, 1.0))
    plt.plot()
    plt.close(fig_var)

'''

###############################################################################
# apply logistic regressor 
# hinge und modified huber erkennen mehr "no spirals"
# l1: mh: 69%, log: 70%, hinge: 60%
# l2: mh: 75%, hinge: 65%, log: 66%

# Stochastic Gradient decent classifier with logistic regression or linear SVM
alpha = 0.001
eta0 = 0.001
sgd = SGDClassifier(loss = 'log', penalty = 'l1', alpha=alpha, 
                        max_iter=100, random_state = random_state,
                        learning_rate = "invscaling", eta0 = eta0)


# Support vector machine classifier with linear, rbf, poly or sigmoid kernel
svc = SVC(max_iter = 1e6)



## Comparison of accuracy for different maximum iteration numbers
#score = np.zeros(6)
#for (i, var) in enumerate([10,1e2,1e3,1e4,1e5,1e6]):
#    sgd = SVC(max_iter = var)
#    sgd.fit(X_train, y_train)
#    score[i] = sgd.score(X_test, y_test)
#
#fig_score = plt.figure()
#plt.plot([10,1e2,1e3,1e4,1e5,1e6], score)
#plt.xscale("log")


# Fit algorithm to training data
alg = svc
alg.fit(X_train, y_train)

if (alg == sgd):
    # percentage of nonzero weights
    sparsity = np.mean(sgd.coef_ == 0) 
    print("Sparsity: {:.2f}".format(sparsity))

# compute accuracy
score = alg.score(X_test, y_test)
print("Score : {:.4f}".format(score))

#display run time
run_time = time.time() - t0
print('Run in {:.3f} s\n'.format(run_time))

#######################################################################################################
### Making predictions on the test data

predictions = alg.predict(X_test)

if (PCAnalysis == False):
    # plot weights vs the pixel position
    coef = alg.coef_.copy()
    plt.figure(figsize=(10, 5))
    scale = np.abs(coef).max()
    for i in range(1):
        l2_plot = plt.subplot(1,1,i+1)
        im = l2_plot.imshow(coef[i].reshape(pixel, pixel), interpolation='nearest',
                            cmap=plt.cm.Greys, vmin=-scale, vmax=scale)
        l2_plot.set_xticks(())
        l2_plot.set_yticks(())
        cbar = plt.colorbar(im)
    plt.subtitle('classification weights for spiral pattern')
    plt.show()


# plot confusion matrix
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

# first alternative
plt.figure(figsize=(9,9))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 20)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["spiral", "no spiral"], size = 12)
plt.yticks(tick_marks, ["spiral", "no spiral"], size = 12)
plt.tight_layout()
plt.ylabel('Actual label', size = 18)
plt.xlabel('Predicted label', size = 18)
width, height = cm.shape
for x in range(width):
    for y in range(height):
        plt.annotate(str(cm[x][y]), xy=(y, x), 
        horizontalalignment='center',
        verticalalignment='center')
        
# second alternative
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 20);
        
# Plotting misclassified pictures
index = 0
misclassifiedIndexes = []
for label, predict in zip(y_test, predictions):
    if label != predict: 
        misclassifiedIndexes.append(index)
    index +=1

plt.figure(figsize=(20,4))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:3]):
    plt.subplot(1, 3, plotIndex + 1)
    plt.imshow(np.reshape(X_test0[badIndex], (pixel, pixel)), cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(predictions[badIndex], y_test[badIndex]), fontsize = 15)       
'''
