###############################################################
#                                                             #
#  LOGISTIC REGRESSION FOR SPIRAL GALAXIES USING MINIBATCHES  #
#                                                             #
###############################################################


import  time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import Isomap #LocallyLinearEmbedding #
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
import seaborn as sns
from sklearn import metrics
from ini import *


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
        #imgStart = num * imgsInChunk
        #imgEnd = imgStart + imgsInChunk
        #yy = yy[imgStart : imgEnd, :]
        yield xx, yy

def principal_components(xtrain, xtest, components, chunk):
    if chunk == 0:
        xtrain0 = np.copy(xtrain[0:4,:])
        xtrain0 = scaler.inverse_transform(xtrain0)
        
    xtrain = pca.transform(xtrain)
    xtest = pca.transform(xtest)
    proj_train = pca.inverse_transform(xtrain)
    proj_train = scaler.inverse_transform(proj_train)
    
    if chunk == 0:
        plt.figure('fig_rec')
        fig, ax = plt.subplots(2, 4, figsize=(10, 4),
                               subplot_kw={'xticks':[], 'yticks':[]},
                               gridspec_kw=dict(hspace=0.1, wspace=0.1))
        for i in range(4):
            ax[0, i].imshow(xtrain0[i].reshape(pixel, pixel), cmap='binary_r')
            ax[1, i].imshow(proj_train[i].reshape(pixel, pixel), cmap='binary_r')
            ax[0, 0].set_ylabel('full-dim\ninput')
            ax[1, 0].set_ylabel('{}-dim\nreconstruction'.format(components))
            plt.plot()
        plt.close('fig_rec')
    return xtrain, xtest #, proj_train, proj_test




###############################################################################

max_iter = 10
alpha = 0.001
eta0 = 0.001
max_iter_range= [max_iter]          #_range = [0.0001, 0.001, 0.005, 0.01, 0.1, 0.5]
score_val = np.zeros(num_chunks)
score_train = np.zeros(num_chunks)
score_test = np.zeros(len(max_iter_range))
for (i, max_iter) in enumerate(max_iter_range):
    #print("Max Iter: {}\n".format(eta))
    warm_start = False
    # Stochastic gradient descent classifier with ca 77% accuracy,
    # underfits the problem, training accuracy ca 80% even for high max iter
    sgd = SGDClassifier(loss = 'log', penalty = 'l1', alpha=alpha, 
                        max_iter=max_iter, random_state = random_state,
                        learning_rate = "invscaling", eta0 = eta0)
    # Using bagging to average over multiple logistic regression classifiers
    # does not converge for max_iter <=10 and takes long otherwise
    # hardly increases accuracy training acc: 81% testing acc: 75%
    bag = BaggingClassifier(n_estimators = 10, base_estimator = sgd, 
                            max_samples = 0.9, warm_start = warm_start, 
                            random_state = random_state)
    # Random Forest using Decision Tree ensemble, for max depth >=10 overfitting
    # nearly 100% training acc but only 84% test acc
    forest = RandomForestClassifier(max_depth = 20, warm_start = warm_start,
                                    random_state = random_state, max_samples = 0.4)
    # Extremly randomized forest
    extra = ExtraTreesClassifier(max_depth = 20, criterion = "entropy", 
                                 max_samples = 0.8, warm_start = warm_start,
                                 random_state = random_state+1, bootstrap = True)
    # choose classifier from above
    clf = extra

    X_test_all = np.array([])
    y_test_all = np.array([])
    id_test_all = np.array([])

    minibatch = get_minibatch(num_chunks)
    for chunk in range(num_chunks):
        X_train, y_train = next(minibatch)
        y_train = [np.argmax(i) for i in y_train] # 0 for spiral, 1 for no spiral
        
        # pick training and test data sets 
        X_train, X_test, y_train, y_test= train_test_split(X_train,y_train,train_size=train_size,test_size
                                                             =test_size, random_state = random_state)
        # We need to split the uploaded X_in arrays into the galaxy ID vector and the image data
        id_train = X_train[:,0]
        X_train = X_train[:,1:]
        id_test = X_test[:,0]
        X_test = X_test[:,1:]
        if chunk == 0:
                X_test0 = np.copy(X_test[0:100,:])

        # scale data to have zero mean and unit variance [required by regressor]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        if PCAnalysis == True:
            if chunk == 0:
                pca = PCA(n_components=components, svd_solver='randomized',
                          random_state = random_state).fit(X_train)
                #pca = Isomap(n_neighbors = 5, n_components = 2,
                #                 ).fit(X_train)
                print("Fitting of PCA to data is completed.\n")
            X_train, X_test = principal_components(X_train, X_test, components, chunk)
            print("Mean and variance of training data after PCA:\n")
            print(np.mean(X_train), np.var(X_train))
            scaler2 = StandardScaler()
            X_train = scaler2.fit_transform(X_train)
            X_test = scaler2.transform(X_test)
            print("Mean and variance of training data after scaling twice:\n")
            print(np.mean(X_train), np.var(X_train))
        
        #Crossvalidation set
        X_train, X_val, y_train, y_val= train_test_split(X_train,y_train,train_size = 0.95,test_size
                                                             = 0.05, random_state = random_state)
        '''state = np.random.get_state()    
        for _ in range(max_iter):
            np.random.set_state(state)
            np.random.shuffle(X_train)
            np.random.set_state(state)
            np.random.shuffle(y_train)
            clf.partial_fit(X_train, y_train, classes = [0, 1, 2])
        '''   
        if chunk == 0:
            warm_start = False
        else: warm_start = True
        clf.fit(X_train, y_train)
        
        # percentage of nonzero weights
        #sparsity = np.mean(clf.coef_ == 0) 
        #print("Sparsity after batch {}: {:.2f}".format(chunk, sparsity))
        
        # compute train accuracy
        score = clf.score(X_train, y_train)
        print("Training score after batch {}: {:.4f}".format(chunk, score))
        score_train[chunk] = score
        
        # compute val accuracy
        score = clf.score(X_val, y_val)
        print("Validation score after batch {}: {:.4f}".format(chunk, score))
        score_val[chunk] = score
        
        #display run time
        run_time = time.time() - t0
        print('Run {} in {:.3f} s\n'.format(chunk, run_time))
    
        X_test_all = np.append(X_test_all, X_test)
        y_test_all = np.append(y_test_all, y_test)
        id_test_all = np.append(id_test_all, id_test)

    #######################################################################################################

    
    X_test_all = X_test_all.reshape(-1, X_test.shape[1])
    predictions = clf.predict(X_test_all)
    score_test[i] = clf.score(X_test_all, y_test_all)
    
    fig_acc = plt.figure()
    plt.plot((np.arange(num_chunks)+1), score_train, label = "Training")
    plt.plot((1 + np.arange(num_chunks)), score_val, label = "Validation")
    plt.plot((1 + np.arange(num_chunks)), (np.ones(num_chunks) * score_test[i]), label = "Test")
    plt.xlabel("Number of trained minibatches")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    plt.close(fig_acc)
    
    #print("Max Iter {} analysis finished.\n".format(eta))


if PCAnalysis == False:
    # plot weights vs the pixel position
    if clf == extra or clf == forest:
        coef = clf.feature_importances_
        plt.figure(figsize=(15, 7))
        scale = np.abs(coef).max()
        for i in range(1):
            l2_plot = plt.subplot(1,1,i+1)
            im = l2_plot.imshow(coef.reshape(pixel, pixel), interpolation='nearest',
                                vmin=-scale, vmax=scale)
            l2_plot.set_xticks(())
            l2_plot.set_yticks(())
            cbar = plt.colorbar(im)
            plt.title('classification weights for spiral pattern', size= 20)

        plt.show()

    
if PCAnalysis == True:
    # plot mean galaxy 
    fig1 = plt.figure()
    mean_img = scaler.inverse_transform(pca.mean_)
    plt.imshow(mean_img.reshape((pixel, pixel)))

    # plot eigengalaxies (example)
    fig2 = plt.figure()
    plt.imshow(pca.components_[0].reshape((pixel, pixel)))
    
    # plot explained variance ratio
    fig3 = plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.ylim((0.47, 1.0))
    plt.plot(np.arange(components), np.ones(components) * 0.95)
    plt.plot()
    
# plot confusion matrix
cm = metrics.confusion_matrix(y_test_all, predictions)
print(cm)
score = clf.score(X_test_all, y_test_all)
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
all_sample_title = 'Accuracy Score: {:.4f}'.format(score)
plt.title(all_sample_title, size = 20);
        
# Plotting misclassified pictures
index = 0
misclassifiedIndexes = []
for label, predict in zip(y_test_all, predictions):
    if label != predict: 
        misclassifiedIndexes.append(index)
    index +=1

plt.figure(figsize=(20,4))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:4]):
    plt.subplot(1, 4, plotIndex + 1)
    plt.imshow(np.reshape(X_test0[badIndex], (pixel, pixel)), cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(predictions[badIndex], y_test_all[badIndex]), fontsize = 15)       

