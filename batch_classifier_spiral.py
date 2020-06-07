###############################################################
#                                                             #
#  LOGISTIC REGRESSION FOR SPIRAL GALAXIES USING MINIMATCHES  #
#                                                             #
###############################################################


import  time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
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
spiral_ind = np.load("spiral_images/indices.npy")
spiral_num = len(spiral_ind)
print("In total there are {} samples.\n".format(spiral_num))
print("Training fraction: {}\n".format(train_size))

# for shuffling data in a reproducable manner
random_state = 2

###############################################################################

### load galaxy data from spiral_images/ 
def get_minibatch(num_batches):
    for num in range(num_batches):
        batch_path = 'spiral_images/images{}.npy'.format(num)
        xx = np.load(batch_path)
        yy = np.load('spiral_images/bin_labels.npy')
        imgsInChunk = xx.shape[0]
        print("Processing chunk {} containing {} pictures...\n".format(num, imgsInChunk))
        imgStart = num * imgsInChunk
        imgEnd = imgStart + imgsInChunk
        yy = yy[imgStart : imgEnd, :]
        yield xx, yy

def principal_components(xtrain, xtest, components, chunk):
    xtrain = pca.transform(xtrain)
    xtest = pca.transform(xtest)
    proj_train = pca.inverse_transform(xtrain)
    proj_test = pca.inverse_transform(xtest)
    if chunk == 0:
        plt.figure('fig1')
        fig, ax = plt.subplots(2, 4, figsize=(10, 4),
                               subplot_kw={'xticks':[], 'yticks':[]},
                               gridspec_kw=dict(hspace=0.1, wspace=0.1))
        for i in range(4):
            ax[0, i].imshow(X_train_in[i].reshape(pixel, pixel), cmap='binary_r')
            ax[1, i].imshow(proj_train[i].reshape(pixel, pixel), cmap='binary_r')
            ax[0, 0].set_ylabel('full-dim\ninput')
            ax[1, 0].set_ylabel('150-dim\nreconstruction')
            plt.plot()
            plt.figure('fig2')
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance');
            plt.plot()
    return xtrain, xtest, proj_train, proj_test




###############################################################################

# apply logistic regressor 
# hinge und modified huber erkennen mehr "no spirals"
# l1: mh: 69%, log: 70%, hinge: 60%
# l2: mh: 75%, hinge: 65%, log: 66%

num_chunks = num_chunks
max_iter = 10

sgd = SGDClassifier(loss = 'hinge', penalty = 'l1', alpha=0.005, max_iter=max_iter, 
                    random_state = random_state)


X_train = np.array([]) #np.zeros((round(spiral_num * train_size), components))
X_test = np.array([]) #np.zeros((round(spiral_num * test_size), components))
X_test_all = np.array([])
y_test_all = np.array([])
id_test_all = np.array([])

minibatch = get_minibatch(num_chunks)
for chunk in range(num_chunks):
    X_in, y = next(minibatch)
    y = [np.argmax(i) for i in y]
    # pick training and test data sets 
    X_train_in, X_test_in, y_train, y_test= train_test_split(X_in,y,train_size=train_size,test_size
                                                             =test_size, random_state = random_state)
    # We need to split the uploaded X_in arrays into the galaxy ID vector and the image data
    id_train = X_train_in[:,0]
    X_train_in = X_train_in[:,1:]
    id_test = X_test_in[:,0]
    X_test_in = X_test_in[:,1:]
    #print("The training data has {} samples of {} features each. \n".format(X_train_in.shape[0], X_train_in.shape[1]))
    # scale data to have zero mean and unit variance [required by regressor]
    #scaler = StandardScaler()
    #X_train_in = scaler.fit_transform(X_train_in)
    #X_test_in = scaler.transform(X_test_in)
    if PCAnalysis == True:
        if chunk == 0:
            pca = PCA(n_components=components, svd_solver='randomized',
                      random_state = random_state).fit(X_train_in)
        X_train, X_test, X_train_proj, X_test_proj = principal_components(X_train_in, X_test_in, 100, chunk)
    else:
        X_train = X_train_in
        X_test = X_test_in
    #len_train = len(X_train)
    #len_test = len(X_test)
    #print(len_train, len_test)
    state = np.random.get_state()    
    for _ in range(max_iter):
        np.random.set_state(state)
        np.random.shuffle(X_train)
        np.random.set_state(state)
        np.random.shuffle(y_train)
        sgd.partial_fit(X_train, y_train, classes = [0, 1])

    # percentage of nonzero weights
    sparsity = np.mean(sgd.coef_ == 0) * 100
    print("Sparsity after batch {}: {:.2f}".format(chunk, sparsity))
    # compute accuracy
    score = sgd.score(X_test, y_test)
    print("Score after batch {}: {:.4f}".format(chunk, score))

    #display run time
    run_time = time.time() - t0
    print('Run {} in {:.3f} s\n'.format(chunk, run_time))
    
    X_test_all = np.append(X_test_all, X_test)
    y_test_all = np.append(y_test_all, y_test)
    id_test_all = np.append(id_test_all, id_test)

#######################################################################################################
X_test_all = X_test_all.reshape(-1, X_test.shape[1])
predictions = sgd.predict(X_test_all)

if PCAnalysis == False:
    # plot weights vs the pixel position
    coef = sgd.coef_.copy()
    plt.figure(figsize=(10, 5))
    scale = np.abs(coef).max()
    for i in range(1):
        l2_plot = plt.subplot(1,1,i+1)
        im = l2_plot.imshow(coef[i].reshape(pixel, pixel), interpolation='nearest',
                            cmap=plt.cm.Greys, vmin=-scale, vmax=scale)
        l2_plot.set_xticks(())
        l2_plot.set_yticks(())
        cbar = plt.colorbar(im)
        plt.suptitle('classification weights for spiral pattern')

    plt.show()
    
if PCAnalysis == True:
    # plot mean galaxy 
    fig1 = plt.figure()
    plt.imshow(pca.mean_.reshape((pixel, pixel)), cmap = plt.cm.bone)

    # plot eigengalaxies (example)
    fig2 = plt.figure()
    plt.imshow(pca.components_[0].reshape((pixel, pixel)))
    
    # plot explained variance ratio
    fig3 = plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.ylim((0.47, 1.0))
    plt.plot()
# plot confusion matrix
cm = metrics.confusion_matrix(y_test_all, predictions)
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
for label, predict in zip(y_test_all, predictions):
    if label != predict: 
        misclassifiedIndexes.append(index)
    index +=1

plt.figure(figsize=(20,4))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:3]):
    plt.subplot(1, 3, plotIndex + 1)
    plt.imshow(np.reshape(X_test_in[badIndex], (pixel, pixel)), cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(predictions[badIndex], y_test_all[badIndex]), fontsize = 15)       

