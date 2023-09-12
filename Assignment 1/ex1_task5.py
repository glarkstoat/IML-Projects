""" Testing and developing custom kernel for SVM using 
    20 newsgroup data-set """

#%%
import numpy as np
import sklearn 
from sklearn import datasets
import matplotlib.pyplot as plt # %matplotlib qt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups_vectorized

#%%
# Loading data-set and data splitting
newsgroups_train = fetch_20newsgroups_vectorized(subset='train')
newsgroups_test = fetch_20newsgroups_vectorized(subset='test')
xtrain, xtest = newsgroups_train.data, newsgroups_test.data
ytrain, ytest = newsgroups_train.target, newsgroups_test.target

# %%
""" Subtask 1: Evaluating different kernels with default parameters """
'''
linearsvm = SVC(kernel='linear').fit(xtrain, ytrain)
print(accuracy_score(linearsvm.predict(xtrain), ytrain)) # 96.1%
print(accuracy_score(linearsvm.predict(xtest), ytest)) # 75.9%

linearsvm = SVC(kernel='poly').fit(xtrain, ytrain) # degree=3
print(accuracy_score(linearsvm.predict(xtrain), ytrain)) # 99.5%
print(accuracy_score(linearsvm.predict(xtest), ytest)) # 69.9% 

linearsvm = SVC(kernel='rbf', gamma=1).fit(xtrain, ytrain)
print(accuracy_score(linearsvm.predict(xtrain), ytrain)) # 98.6%
print(accuracy_score(linearsvm.predict(xtest), ytest)) # 74.5%

linearsvm = SVC(kernel='sigmoid').fit(xtrain, ytrain)
print(accuracy_score(linearsvm.predict(xtrain), ytrain)) # 89.4%
print(accuracy_score(linearsvm.predict(xtest), ytest)) # 71.3% 
'''
# %%
""" Selecting optimal hyperparameters for the respective kernels via 
    GridSearchCV function. Cycles through the different combinations
    of parameters and calculates the average loss via cross-validation.
    Returns the set of parameters with the best score. 
    Since linear kernel has no relavant hyperparameters it's not part of this 
    evaluation. """

# RBF parameters
param_grid = [
  {'gamma': [1,3,5], 'kernel': ['rbf']}
 ]
'''
# Uses k randomly chosen samples for fitting the model
k = int(ytrain.shape[0] / 5) # fraction of the training-set
ran = np.random.choice(range(0, ytrain.shape[0]), k, replace=False)
svc = SVC() # svm-model
# Evaluates model with different parameter combinations
clf = GridSearchCV(svc, param_grid, n_jobs=-1).fit(xtrain[ran], ytrain[ran])
print(clf.best_estimator_) # best parameters (highest score)
'''
#%%
# Poly parameters
param_grid = [
  {'degree': [2,3,4], 'coef0': [5,7,10], 'kernel': ['poly']}
 ]
'''
# Uses k randomly chosen samples for fitting the model
k = int(ytrain.shape[0] / 5) # fraction of the training-set
ran = np.random.choice(range(0, ytrain.shape[0]), k, replace=False)
svc = SVC() # svm-model
# Evaluates model with different parameter combinations
clf = GridSearchCV(svc, param_grid, n_jobs=-1).fit(xtrain[ran], ytrain[ran])
print(clf.best_estimator_) # best parameters (highest score)
'''
#%%
# Sigmoid parameters
param_grid = [
  {'gamma': [1, 2, 0.01], 'coef0': [0,1,2,3], 'kernel': ['sigmoid']}
 ]
'''
# Uses k randomly chosen samples for fitting the model
k = int(ytrain.shape[0] / 5) # fraction of the training-set
ran = np.random.choice(range(0, ytrain.shape[0]), k, replace=False)
svc = SVC() # svm-model
# Evaluates model with different parameter-combinations
clf = GridSearchCV(svc, param_grid, n_jobs=-1).fit(xtrain[ran], ytrain[ran])
print(clf.best_estimator_) # best parameters (highest score)
'''
#%%
""" Evaluating the kernels with optimal hyperparameters """
'''
linearsvm = SVC(coef0=7, degree=2, kernel='poly').fit(xtrain, ytrain)
print(accuracy_score(linearsvm.predict(xtrain), ytrain)) # 99.9%
print(accuracy_score(linearsvm.predict(xtest), ytest)) # 78.2%

linearsvm = SVC(gamma=1, kernel='rbf').fit(xtrain, ytrain)
print(accuracy_score(linearsvm.predict(xtrain), ytrain)) # 98.7%
print(accuracy_score(linearsvm.predict(xtest), ytest)) # 74.5%

linearsvm = SVC(coef0=0, gamma=1, kernel='sigmoid').fit(xtrain, ytrain)
print(accuracy_score(linearsvm.predict(xtrain), ytrain)) # 89.4%
print(accuracy_score(linearsvm.predict(xtest), ytest)) # 71.3%
'''
#%%
""" Subtask 2: Developing custom kernel """

def my_kernel(X, Y):
    """ Sclaled version of the linear kernel """
    return 8.05*X.dot(Y.T)

def my_kernel_poly(X, Y):
    """ Sclaled version of the linear kernel """
    return (X.dot(Y.T))**2

# Calculating the accuracy score using the custom kernel
linearsvm = SVC(kernel=my_kernel).fit(xtrain, ytrain) # uses custom kernel
print(accuracy_score(linearsvm.predict(xtrain), ytrain)) # 99.9%
print(accuracy_score(linearsvm.predict(xtest), ytest)) # 78.3%

'''

# Checking whether the Gram-matrix is semi. pos. def.
""" computes all permutations of the kernel on the sample vectors. 
    G_ij = k(x_i, x_j) """
G = my_kernel[xtrain, xtrain]
w,v = np.linalg.eig(G) # computes all eigenvectors of G. Takes a long time!
# Checks whether all eigenvectors are non-negative
print('Gram-matrix is semi. pos. def.: ', np.sum(w<0)==0) # True if all ev. are non-negative
'''
# %%
