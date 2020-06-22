import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.tree import DecisionTreeClassifier
#from sklearn.manifold import Isomap #LocallyLinearEmbedding #
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

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
X_train = np.load("{}/xtrain.npy".format(img_folder))
X_test = np.load("{}/xtest.npy".format(img_folder))
y_train = np.load("{}/ytrain.npy".format(img_folder))
y_test = np.load("{}/ytest.npy".format(img_folder))
X_test0 = np.load("{}/xtest0.npy".format(img_folder))


print("The training data has {} samples of {} features each. \n".format(X_train.shape[0], X_train.shape[1]))

###############################################################################
# apply logistic regressor 
# hinge und modified huber erkennen mehr "no spirals"
# l1: mh: 69%, log: 70%, hinge: 60%
# l2: mh: 75%, hinge: 65%, log: 66%

# Stochastic Gradient decent classifier with logistic regression or linear SVM
alpha = 0.001
eta0 = 0.001
sgd = SGDClassifier(loss = 'log', penalty = 'l1', alpha=alpha, 
                        max_iter=1000, random_state = random_state,
                        learning_rate = "invscaling", eta0 = eta0)


# Support vector machine classifier with linear, rbf, poly or sigmoid kernel
svc = SVC(max_iter = 1e4)

# Random Forest using Decision Tree ensemble, for max depth >=10 overfitting
# nearly 100% training acc but only 84% test acc
forest = RandomForestClassifier(max_depth = 10, random_state = random_state, 
                                max_samples = 0.9)
# Extremly randomized forest
extra = ExtraTreesClassifier(max_depth = 15, criterion = "gini", 
                             max_samples = 0.8, 
                             random_state = random_state, bootstrap = True)
    
boost = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 10))

'''
# Comparison of accuracy for different maximum iteration numbers
score = np.zeros(6)
for (i, var) in enumerate([10,1e2,1e3,1e4,1e5,1e6]):
    sgd = SVC(max_iter = var)
    sgd.fit(X_train, y_train)
    score[i] = sgd.score(X_test, y_test)

fig_score = plt.figure()
plt.plot([10,1e2,1e3,1e4,1e5,1e6], score)
plt.xscale("log")
'''

# Fit algorithm to training data
alg = boost

# if several hyperparameters need to be tested loop around this
#scores = cross_val_score(alg, X_train, y_train, cv = 4)
#print("Cross validation scores : {}".format(scores))

alg.fit(X_train, y_train)

if (alg == sgd):
    # percentage of nonzero weights
    sparsity = np.mean(alg.coef_ == 0) 
    print("Sparsity: {:.2f}".format(sparsity))

print("Training score: {:.2f}".format(alg.score(X_train, y_train)))

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
score = alg.score(X_test, y_test)
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


