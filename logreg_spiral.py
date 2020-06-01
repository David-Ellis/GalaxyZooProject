############################################
#                                          #
#  LOGISTIC REGRESSION FOR SPIRAL GALAXIES #
#                                          #
############################################

import  time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import seaborn as sns
from sklearn import metrics

print(__doc__)

# Turn down for faster convergence
t0 = time.time()
train_size = 0.8
test_size = 0.2
print("Training fraction: {}\n".format(train_size))

### load galaxy data from spiral_images/ 
X_in = np.load('spiral_images/images1.npy')
y = np.load('spiral_images/bin_labels.npy')
chunk = 4
imgsInChunk = X_in.shape[0]
print("Processing chunk {} containing {} pictures...\n".format(chunk, imgsInChunk))

imgStart = chunk * imgsInChunk
imgEnd = imgStart + imgsInChunk
y = y[imgStart : imgEnd]

# for shuffling data in a reproducable manner
random_state = 1

# pick training and test data sets 
X_train_in, X_test_in, y_train, y_test = train_test_split(X_in,y,train_size=train_size,test_size
                                                    =test_size, random_state = random_state)

# We need to split the uploaded X_in arrays into the galaxy ID vector and the image data
id_train = X_train_in[:,0]
X_train = X_train_in[:,1:]
id_test = X_test_in[:,0]
X_test = X_test_in[:,1:]
print("The training data has {} samples of {} features each. \n".format(X_train.shape[0], X_train.shape[1]))

# scale data to have zero mean and unit variance [required by regressor]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# apply logistic regressor with 'sag' solver, C is the inverse regularization strength
clf = LogisticRegression(C=1e5,
                         multi_class='multinomial',
                         penalty='l2', solver='sag', tol=0.1)
# fit data
clf.fit(X_train, y_train)
# percentage of nonzero weights
sparsity = np.mean(clf.coef_ == 0) * 100
# compute accuracy
score = clf.score(X_test, y_test)

#display run time
run_time = time.time() - t0
print('Example run in %.3f s' % run_time)

print("Sparsity with L2 penalty: %.2f%%" % sparsity)
print("Test score with L2 penalty: %.4f" % score)


#######################################################################################################

predictions = clf.predict(X_test)


# plot weights vs the pixel position
coef = clf.coef_.copy()
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(1):
    l2_plot = plt.subplot(1,1,i+1)
    im = l2_plot.imshow(coef[i].reshape(45, 45), interpolation='nearest',
                   cmap=plt.cm.Greys, vmin=-scale, vmax=scale)
    l2_plot.set_xticks(())
    l2_plot.set_yticks(())
    cbar = plt.colorbar(im)
plt.suptitle('classification weights for spiral pattern')

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
    plt.imshow(np.reshape(X_test[badIndex], (45, 45)), cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(predictions[badIndex], y_test[badIndex]), fontsize = 15)       

