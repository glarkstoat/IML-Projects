"""
Task: 2: Non-linear Regression and Classification
"""

#%%
import numpy as np
import scipy as sp
import sklearn 
from sklearn import datasets
import h5py
import matplotlib.pyplot as plt # %matplotlib qt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import cm

sns.set_style('darkgrid')
sns.set(color_codes=True) # Nice layout.
sns.set_context('paper')
sns.set_palette("mako")

#%% 
# Loading of the dataset
hf = h5py.File('toy-regression.h5', 'r')
xtrain = np.array(hf.get('x_train'))
ytrain = np.array(hf.get('y_train'))
xtest = np.array(hf.get('x_test'))
ytest = np.array(hf.get('y_test'))
hf.close()

# %%
""" Subtask 1:  Toy Data Regression """

""" For easier computation of the feature transformations the data is standardized.
Thus, centered to the mean and component-wise scaled to unit variance."""

scaler = StandardScaler().fit(xtrain) # scaler is fitted with training data
xtrain_scaled = scaler.transform(xtrain) # standardized training-set
xtest_scaled = scaler.transform(xtest) # standardized test-set

# Setting up the 3-d plot. Shows the effect of the standardization.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xtest[:,0], xtest[:,1],ytest,s=3, label="non-scaled") # 3d plot of the data points
ax.scatter(xtest_scaled[:,0], xtest_scaled[:,1],ytest,s=3,alpha=0.5,c="g",label="scaled")
ax.legend()

#%%
''' Non-linear feature transformation. This aims to recreate the 3-d cone-shape
of the data. '''

ones = np.ones((len(xtrain_scaled),1)) # Ones for the offset terms
# the square root for recreating the 3-d cone-shape of the data
# corresponds to sqrt(x1**2 + x2**2) for every row of xtrain_scaled
a = np.array([np.sqrt(row[0]**2 + row[1]**2) for row in xtrain_scaled]).reshape(len(xtrain_scaled),1)

''' New X. Similar when using polynomial feature transformation. 
    Column_stack extends the columns of X by arbitrary features. '''
xtrain_new = np.column_stack((ones,
                         a,
                         xtrain_scaled[:,0], # feature 1
                         xtrain_scaled[:,1])) # feature 2))

# Same is necessary for using the linear model to predict the outcome of the test set. 
a = np.array([np.sqrt(row[0]**2 + row[1]**2) for row in xtest_scaled]).reshape(len(xtest_scaled),1)

xtest_new = np.column_stack((ones, 
                              a,
                              xtest_scaled[:,0],
                              xtest_scaled[:,1]))

# linear model fitted with new training data
reg = LinearRegression().fit(xtrain_new, ytrain) 

mse_train = mean_squared_error(ytrain, reg.predict(xtrain_new))
mse_test = mean_squared_error(ytest, reg.predict(xtest_new))

print('Using the non-linear feature transformation:\n',
      'MSE of training set: ', np.round(mse_train,8),
    '\nMSE of test set: ', np.round(mse_test,8), '\n')

#%%
# Setting up the meshgrid for plotting the function Z
x1 = np.linspace(-5,5); x2= np.linspace(-5,5)
X1, X2 = np.meshgrid(x1,x2)

def Z(w,x,y): 
    """ 2-d function equivalent to the feature transformation. """
    return  w[0] + w[1] * np.sqrt(x**2 + y**2) + w[2] * x + w[3] * y

w = reg.coef_ # weight vector of the linear model obtained above

# Visually showcases the accuracy of the linear model
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, Z(w,X1, X2), s=1, alpha=0.2)
ax.scatter(xtrain_scaled[:,0], xtrain_scaled[:,1], Z(w, xtrain_scaled[:,0]
                                                     ,xtrain_scaled[:,1]),
                                                     s=0.5,alpha=0.8,c="brown")

#%%
# Using polynomial features to fit the linear model

def PolyFeatures(degree):
      """ Calculates the MSEs for a given polynomial feature-transformation. """
      
      poly = PolynomialFeatures(degree).fit(xtrain_scaled) # Model is fitted on training set
      xtrain_poly = poly.transform(xtrain_scaled) # transform features
      xtest_poly = poly.transform(xtest_scaled) # transform features
      reg = LinearRegression().fit(xtrain_poly, ytrain) # linear model fitted polynomial features

      train_mse = mean_squared_error(ytrain, reg.predict(xtrain_poly))
      test_mse = mean_squared_error(ytest, reg.predict(xtest_poly)) 

      return train_mse, test_mse

mses_train, mses_test = [], []
for i in range(2,12): # computes MSEs for degrees 2-11
      train_mse, test_mse = PolyFeatures(i)
      mses_train.append(train_mse); mses_test.append(test_mse)

best = np.argmin(mses_test) # degree with minimal test error
print('Best performance (PolyFeatures) at degree: ', best+2, 
      '\nMSE training set: ', np.round(mses_train[best],5),
      '\nMSE test set: ', np.round(mses_test[best], 5), "\n")

#%%
""" Subtask 2:  Toy Data Classification """

# Loading the dataset
hf = h5py.File('toy-classification.h5', 'r')
xtrain = np.array(hf.get('x_train'))
ytrain = np.array(hf.get('y_train'))
xtest = np.array(hf.get('x_test'))
ytest = np.array(hf.get('y_test'))
hf.close()

#%% 
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

# Constructing the non-linear feature transformations
""" Using the circle-shaped data, a circle equation e.g. a*x**2 + b*y**2 - c is used. """
a = np.array([(row[0]**2 + row[1]**2) for row in xtrain]).reshape(len(xtrain),1)
# Additional transformation that proved itself useful
b = np.array([np.sqrt(row[0]**2 + row[1]**2) for row in xtrain]).reshape(len(xtrain),1)

xtrain_new = np.column_stack((np.ones((len(xtrain))), # offset features
                         xtrain[:,0]**2, # feature 1
                         xtrain[:,1]**2, # feature 2
                         a,
                         b))

# Initializing and training the SVM
linearsvm = LinearSVC().fit(xtrain_new, ytrain)

# Checking the accuracy of the model for the train-set
y_pred =  linearsvm.predict(xtrain_new)
correct = np.sum(ytrain == y_pred) / ytrain.shape[0]

print(np.round(correct * 100, 1), "% training samples correctly classified")

# Constructing the feature transformations for the test-set
a = np.array([(row[0]**2 + row[1]**2) for row in xtest]).reshape(len(xtrain),1)
b = np.array([np.sqrt(row[0]**2 + row[1]**2) for row in xtest]).reshape(len(xtrain),1)

xtest_new = np.column_stack((np.ones((len(xtrain))),
                         xtest[:,0]**2,
                         xtest[:,1]**2, 
                         a,
                         b))

# Checking accuracy for test-set
y_pred = linearsvm.predict(xtest_new) # predicted labels using the trained model
correct = np.sum(ytest == y_pred) / ytest.shape[0] # fraction of correct predictions
print(np.round(correct * 100, 1), "% test samples correctly classified")

# 3-d plot of the test-set
fig = plt.figure()
plt.scatter(xtest[:,0], xtest[:,1], c = ytest, s = 3, cmap = plt.cm.Spectral)

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, s=3, cmap=plt.cm.Spectral)
    



#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(xtest[:,0], xtest[:,1], ytest, s=1, alpha=1, c="brown")

conf = confusion_matrix(ytest, linearsvm.predict(xtest_new))
print(conf)
#sklearn.metrics.plot_confusion_matrix(linearsvm, xtest_new, ytest)
#%%