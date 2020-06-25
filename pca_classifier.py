import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
#from sklearn.decomposition import PCA
#from sklearn.decomposition import IncrementalPCA
from sklearn.tree import DecisionTreeClassifier
#from sklearn.manifold import Isomap #LocallyLinearEmbedding #
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
import seaborn as sns
from sklearn import metrics
from ini import *
from sklearn.model_selection import GridSearchCV

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
y_train = np.load("{}/ytrainhard.npy".format(img_folder))
y_test = np.load("{}/ytesthard.npy".format(img_folder))
X_test0 = np.load("{}/xtest0.npy".format(img_folder))


print("The training data has {} samples of {} features each. \n".format(X_train.shape[0], X_train.shape[1]))

###############################################################################

# Stochastic Gradient decent classifier with logistic regression or linear SVM
sgd = SGDClassifier(loss = "log", random_state = random_state, early_stopping=True,
                    validation_fraction=0.2, max_iter=10000, penalty='l1',
                    n_iter_no_change=20, learning_rate='constant',
                    eta0 = 0.001, alpha = 0.01)#,
                  
# comparing constant learning rates
sgd_params = {'eta0':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.5] }
alg = GridSearchCV(sgd, sgd_params, verbose = 2, cv = 3)
alg.fit(X_train, y_train)
means = alg.cv_results_['mean_test_score']
stds = alg.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, alg.cv_results_['params']):
    print('%.03f (+/-%0.03f) for %r' %(mean, std * 2, params))
print(alg.best_params_)
fig_sgd1 = plt.figure()
plt.errorbar([0.0001, 0.001, 0.01, 0.1, 0.2, 0.5], means, yerr = stds, fmt = 'o', ecolor = 'lightgray')
plt.xscale('log')
plt.show()
plt.close()

np.save("./{}_sgd_eta0_means.npy".format(img_folder), means)
np.save("./{}_sgd_eta0_stds.npy".format(img_folder), stds)
np.save("./{}_sgd_eta0_results.npy".format(img_folder), alg.cv_results_)


# comparing different regularization schemes
sgd_params = {'alpha':[ 0.0001, 0.001, 0.01, 0.1, 0.2]}

alg = GridSearchCV(sgd, sgd_params, verbose = 2, cv = 3)
alg.fit(X_train, y_train)
means = alg.cv_results_['mean_test_score']
stds = alg.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, alg.cv_results_['params']):
    print('%.03f (+/-%0.03f) for %r' %(mean, std * 2, params))
print(alg.best_params_)
fig_sgd2 = plt.figure()
plt.errorbar([0.0001, 0.001, 0.01, 0.1, 0.2], means, yerr = stds, fmt = 'o', ecolor = 'lightgray')
plt.xscale('log')
plt.show()
plt.close()

np.save("./{}_sgd_alpha_means.npy".format(img_folder), means)
np.save("./{}_sgd_alpha_stds.npy".format(img_folder), stds)
np.save("./{}_sgd_alpha_results.npy".format(img_folder), alg.cv_results_)


# Training: 0.78, Test: 0.76 - linear SVM also scores 0.77
###############################################################################
# Support vector machine classifier with linear, rbf, poly or sigmoid kernel
# learning rate best for C = 10.0, gamma = 0.001, TA = 1.00, score = 80%

svc = SVC(max_iter = 1e5, random_state=random_state, verbose = True, 
          kernel="rbf", C = 10.0, gamma = 0.001)

# comparing different regularization schemes
svc_params = {'C':[ 0.1, 1.0, 10., 100.0], 'gamma':[0.0001, 0.001, 0.01, 0.1]}

alg = GridSearchCV(svc, svc_params, cv = 2, verbose = 2)
alg.fit(X_train, y_train)
means = alg.cv_results_['mean_test_score']
stds = alg.cv_results_['std_test_score']
#for mean, std, params in zip(means, stds, alg.cv_results_['params']):
#    print('%.03f (+/-%0.03f) for %r' %(mean, std * 2, params))
print(alg.best_params_)

means = means.reshape((2,2))
fig1 = plt.figure(figsize = (12,8))
fig_svc = plt.imshow(means, interpolation='nearest')
plt.yticks(np.arange(2), [ 0.1, 1.0, 10., 100.0], size = 18)
plt.xticks(np.arange(2), [0.0001, 0.001, 0.01, 0.1], size = 18)
plt.xlabel('gamma', size = 18)
plt.ylabel('C', size = 18)
for x in range(2):
    for y in range(2):
        plt.annotate(str(np.round(means[x][y], decimals = 2)), xy=(y, x), 
        horizontalalignment='center',
        verticalalignment='center', size = 18, color = "w")
plt.show()
plt.close()

means = means.flatten()
np.save("./{}_svm_gs_means.npy".format(img_folder), means)
np.save("./{}_svm_gs_stds.npy".format(img_folder), stds)
np.save("./{}_svm_gs_results.npy".format(img_folder), alg.cv_results_)

#Training: Test:
###############################################################################
# Random Forest using Decision Tree ensemble, for max depth >=10 overfitting
# nearly 100% training acc but only 84% test acc
# best spiral params: dep = 11, est = 300, sam = 0.8, TA = 0.98, score = 79%
forest = RandomForestClassifier(random_state = random_state, max_samples = 0.8, verbose = 1,
                                max_depth = 11, n_estimators = 200)
                                
# Extremly randomized forest
# best spiral params: dep = 8, est = 200, sam = 0.8, TA = 0.98, score = 0.73
extra = ExtraTreesClassifier(criterion = "gini", max_samples = 0.8, n_estimators = 200,
                             random_state = random_state, bootstrap = True, verbose=1, 
                             max_depth = 10)#,
                             #n_estimators = 200, max_depth = 15                          

# comparing different  estimator numbers
forest_params = {'n_estimators':[ 10, 50, 100, 200, 300]}

alg = GridSearchCV(forest, forest_params, cv = 3, verbose = 3)
alg.fit(X_train, y_train)
means = alg.cv_results_['mean_test_score']
stds = alg.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, alg.cv_results_['params']):
    print('%.03f (+/-%0.03f) for %r' %(mean, std * 2, params))
print(alg.best_params_)
fig_forest = plt.figure()
plt.errorbar([ 10, 50, 100, 200, 300], means, yerr = stds, fmt = 'o', ecolor = 'lightgray')
plt.show()
plt.close()

np.save("./{}_forest_gs_means.npy".format(img_folder), means)
np.save("./{}_forest_gs_stds.npy".format(img_folder), stds)
np.save("./{}_forest_gs_results.npy".format(img_folder), alg.cv_results_)

# comparing different max_depths
extra_params = {'max_depth':[ 2,5,10,15,20]}
alg = GridSearchCV(forest, extra_params, cv = 3, verbose=3)
alg.fit(X_train, y_train)
means = alg.cv_results_['mean_test_score']
stds = alg.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, alg.cv_results_['params']):
    print('%.03f (+/-%0.03f) for %r' %(mean, std * 2, params))
print(alg.best_params_)
fig_extra = plt.figure()
plt.errorbar([2,5,10,15,20], means, yerr = stds, fmt = 'o', ecolor = 'lightgray')
plt.show()
plt.close()

np.save("./{}_extra_gs_means.npy".format(img_folder), means)
np.save("./{}_extra_gs_stds.npy".format(img_folder), stds)
np.save("./{}_extra_gs_results.npy".format(img_folder), alg.cv_results_)



###############################################################################
# best spiral params: lr = 0.6, dep = 2, est = 150, TA = 0.92, score = 81%  
boost = AdaBoostClassifier( base_estimator = DecisionTreeClassifier(max_depth = 2, random_state=random_state),
                           random_state = random_state, learning_rate=0.6,
                           n_estimators = 150)

boost_params = {'learning_rate':[0.2, 0.4, 0.6, 0.8]}

alg = GridSearchCV(boost, boost_params, cv = 3, verbose=2)
alg.fit(X_train, y_train)
means = alg.cv_results_['mean_test_score']
stds = alg.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, alg.cv_results_['params']):
    print('%.03f (+/-%0.03f) for %r' %(mean, std * 2, params))
print(alg.best_params_)
fig_boost = plt.figure()
plt.errorbar([0.2, 0.4, 0.6, 0.8], means, yerr = stds, fmt = 'o', ecolor = 'lightgray')
#plt.xscale('log')
plt.show()
plt.close()

np.save("./{}_boost_est_means.npy".format(img_folder), means)
np.save("./{}_boost_est_stds.npy".format(img_folder), stds)
np.save("./{}_boost_est_results.npy".format(img_folder), alg.cv_results_)


'''
alg = boost
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
plt.figure(figsize=(12,12))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
#plt.title('Confusion matrix', size = 2)
cb = plt.colorbar()
cb.ax.tick_params(labelsize = 24)
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["spiral", "no spiral"], size = 24)
plt.yticks(tick_marks, ["spiral", "no spiral"], size = 24)

plt.ylabel('Actual label', size = 26)
plt.xlabel('Predicted label', size = 26)
width, height = cm.shape
for x in range(width):
    for y in range(height):
        plt.annotate(str(cm[x][y]), xy=(y, x), 
        horizontalalignment='center',
        verticalalignment='center', size = 24)
        
# second alternative
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label', size = 18);
plt.xlabel('Predicted label', size = 18);
all_sample_title = 'Accuracy Score: {0:.2f}'.format(score)
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
