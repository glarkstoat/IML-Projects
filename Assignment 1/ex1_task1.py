# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 18:13:40 2020
@author: Christian Wiskott

- California Housing Data Set

Attribute Information: - MedInc: median income in block
                       - HouseAge: median house age in block
                       - AveRooms: average number of rooms
                       - AveBedrms: average number of bedrooms
                       - Population: block population
                       - AveOccup: average house occupancy
                       - Latitude: house block latitude
                       - Longitude: house block longitude
                       
The target variable is the median house value for California districts.
A block group is the smallest geographical unit for which the U.S. Census Bureau 
publishes sample data (a block group typically has a population of 600 to 3,000 people).

"""
# %% 
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt # %matplotlib qt for plots in separate window
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Loading of the dataset
dataframe = datasets.fetch_california_housing()

X = dataframe['data']
Y = dataframe['target']
feature_names = dataframe['feature_names']; dim = len(feature_names)
target_names = dataframe['target_names']

#%% 
''' Subtask 1: Splitting X & Y into test (30%) and training set (70%) '''

# shuffled by default   
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, train_size=0.7) 
#ytrain, ytest = ytrain.reshape(-1,1), ytest.reshape(-1,1) # avoids complications calculations

print('Size of dataset: ', len(X), 
      '\nSize of training set: ', len(xtrain),
      '\nSize of test set: ', len(xtest),
      '\nNumber of observed features: ', dim, "\n")

#%% 
''' Subtask 2: Train a least squares linear regression model and determine MSE '''

reg = LinearRegression(fit_intercept=False).fit(xtrain, ytrain) # without bias term
weights = reg.coef_ # weight vector of linear model

mse_test = mean_squared_error(ytest, reg.predict(xtest)) # mse of model using test set
mse_train = mean_squared_error(ytrain, reg.predict(xtrain)) # mse of model using training set

print('MSE of training set: ', np.round(mse_train,5),
    '\nMSE of test set: ', np.round(mse_test,5), '\n')

#%% 
''' Subtask 3: Manual implementation of linear regression via closed form solution '''

A = xtrain.T @ xtrain # system matrix without bias terms
b = xtrain.T @ ytrain 
w_manual = np.linalg.pinv(A) @ b # closed-form solution for weight vector
print('Computed weight vector: \n', w_manual)

# manual prediction. X @ w equivalent to calculated hypothesis for every row of X
mse_test_manual = mean_squared_error(ytest, xtest @ w_manual)
mse_train_manual = mean_squared_error(ytrain, xtrain @ w_manual)

print('Manual MSE of training set: ', np.round(mse_train_manual, 5),
      '\nManual MSE of test set: ', np.round(mse_test_manual, 5), "\n")

# %%
''' Subtask 4: Gradient Descent for finding optimal weight vector '''

def GD(A, b, xtrain, xtest, ytrain, ytest, k):
      """ Calculates the optimal weight vector via gradient descent. Uses an alternative 
      formulation of the problem, where A and b can be reused for the iterations. Only one 
      vector-matrix multiplication necessary per iteration step. Returns the MSEs for the 
      training- and the test-set. """
      
      w0 = np.zeros((dim,1)) # starts with an all-zero weight vector

      for t in range(1,k+1): # Starts with t=1
        lr = 1e-6 / (1+t) # Learning rate scheme. Gets increasingly smaller
        
        # equivalent to ((y[i] - w0.T @ x[i]) * x[i]) 
        w0 = w0 - (2 / len(ytrain) * lr * (w0.T @ A - b.T).T) 
      
      # Using manual prediction again to calculate MSEs
      mse_test = mean_squared_error(ytest, xtest @ w0) # mse of test-set
      mse_train = mean_squared_error(ytrain, xtrain @ w0) # mse of training-set
        
      return w0, mse_test, mse_train

def MSEs():   
      """ Returns the MSE-plot of the training and the test set using GD for
      a given number of iterations"""
      
      plot_range = [1 + 1000*k for k in range(0,101,4)] # x-axis of MSE plot. Every 4th result
      mses_test, mses_train = [], []
      for i in plot_range:
            w, mse_test, mse_train = GD(A, b, xtrain, xtest,
                                        ytrain, ytest, i) # MSEs for given iterations
            
            # Adds the errors of the last step to the array
            mses_test.append(mse_test); 
            mses_train.append(mse_train) 
      
      print('Training-MSE after ', int(i/1000), 'k iterations: '
            , np.round(mses_train[-1],5),
            '\nTest-MSE after ', int(i/1000), 'k iterations: ', 
            np.round(mses_test[-1],5), '\n')
      
      # Plotting the MSEs for the given set of iterations
      fig = plt.figure()
      plt.title('MSE non-normalized GD', fontweight="bold")
      plt.plot(np.array(plot_range)/1000, mses_train, label="train", c="r")
      plt.plot(np.array(plot_range)/1000, mses_test, label="test", c='g')
      plt.xlabel('Number of iterations in steps of 1000', fontweight="bold")
      plt.ylabel('Mean Squared Error', fontweight="bold")
      plt.legend()
      
      # Subplot that shows the descending error
      ax2 = fig.add_axes([0.41, 0.41, 0.4, 0.4])
      ax2.plot(np.array(plot_range[-20:])/1000, mses_train[-20:], label="train", c='r')
      ax2.plot(np.array(plot_range[-20:])/1000, mses_test[-20:], label="test", c='g')
      plt.show()
      
      return mses_train, mses_test

mses_train, mses_test = MSEs()

#%%
''' Subtask 5: Adaptive Learning Scheme '''

def cost(x,y,w): # Least Squares loss
      sum=0
      for i in range(len(y)):
            sum += (y[i] - w.T @ x[i])**2
      return sum #np.sum(y - x @ w)**2 is much slower (very strange)

def GD_adaptive(A, b, xtrain, xtest, ytrain, ytest, k):      
       """ Calculates the optimal weight vector via gradient descent using an adaptive
       learning rate. Only one vector-matrix multiplication necessary per iteration
       step. Returns the MSEs for the training- and the test-set. """
      
       w = np.zeros((dim,1)) # starts with an all-zero weight vector
       lr = 0.0001 # initial learning rate
       
       for t in range(1,k+1):
         # cost before new w0. Necessary for bold driver evualtion
         cost_before = cost(xtrain, ytrain, w) #np.sum(ytrain - xtrain @ w)**2
         w = w - (2 / len(ytrain) * lr * (w.T @ A - b.T).T) # new weight vector
         
        # Bold driver adaptive learning rate
         cost_new = cost(xtrain, ytrain, w) # cost with new weight vector
         
         # Checks if the cost with new weight vector has improved
         if cost_new < cost_before: 
             lr *= 1.1 # Right direction. --> Increase learning rate
         else: # Cost has increased!
             lr *= 0.5 # Wrong direction --> Decrease learning rate

       mse_test = mean_squared_error(ytest, xtest @ w) # mse of the test-set
       mse_train = mean_squared_error(ytrain, xtrain @ w) # mse of the training-set
      
       return w, mse_test, mse_train

def MSEs():   
      """ Returns the MSEs of the training and the test set 
      using GD for a given number of iterations"""   
      
      plot_range = [1 + 1000*k for k in [0,1,2,5,10,20,30]] # x-axis of the MSE plot
      mses_test, mses_train = [], []
      for i in plot_range:
            w, mse_test, mse_train = GD_adaptive(A, b, xtrain, xtest,
                                                 ytrain, ytest, i) # mses
      
            # Adds the error of the last step to the array
            mses_test.append(mse_test)
            mses_train.append(mse_train) 
            
      print('Training-MSE after ', int(i/1000), 'k iterations: ',
            np.round(mses_train[-1],5),
            '\nTest-MSE after ', int(i/1000), 'k iterations: ', 
            np.round(mses_test[-1],5), '\n')
      
      # Plotting the MSEs for the given set of iterations
      fig = plt.figure()
      plt.title('MSE non-normalized GD adaptive-lr', fontweight="bold")
      plt.plot(np.array(plot_range)/1000, mses_train, label="train", c='r')
      plt.plot(np.array(plot_range)/1000, mses_test, label="test", c='g')
      plt.xlabel('Number of iterations in steps of 1000', fontweight="bold")
      plt.ylabel('Mean Squared Error', fontweight="bold")
      plt.legend()
      
      # Subplot that shows the descending errors
      ax2 = fig.add_axes([0.4, 0.4, 0.4, 0.4])
      ax2.plot(np.array(plot_range[-4:])/1000, mses_train[-4:], label="train", c='r')
      ax2.plot(np.array(plot_range[-4:])/1000, mses_test[-4:], label="test", c='g')

      plt.show()
      
      return mses_train, mses_test

#mses_train, mses_test = MSEs()

#%%
''' Subtask 6: Normalizing data with MinMaxScaler and repeating GD calcs. '''

scaler = MinMaxScaler().fit(xtrain)
xtrain_scaled = scaler.transform(xtrain)
xtest_scaled = scaler.transform(xtest)

A = xtrain_scaled.T @ xtrain_scaled # new system matrix
b = xtrain_scaled.T @ ytrain

def MSE1():
      """ Returns the MSEs of the training and the test set using GD
      for a given number of iterations"""   
      
      plot_range = [1 + 1000*k for k in range(0,101,4)] # x-axis of the MSE plot
      mses_test, mses_train = [], []
      for i in plot_range:
            w, mse_test, mse_train = GD(A, b, xtrain_scaled, xtest_scaled,
                                        ytrain, ytest, i) # Computes the mses
            # Adds the error of the last step to the array
            mses_test.append(mse_test)
            mses_train.append(mse_train)
      
      print('Training-MSE after ', int(i/1000), 'k iterations: ',
            np.round(mses_train[-1],5),
            '\nTest-MSE after ', int(i/1000), 'k iterations: ',
            np.round(mses_test[-1],5), '\n')
       
      # Plotting the MSEs for the given set of iterations
      fig = plt.figure()
      plt.title('MSE normalized GD', fontweight="bold")
      plt.plot(np.array(plot_range)/1000, mses_train, label="train", c='r')
      plt.plot(np.array(plot_range)/1000, mses_test, label="test", c='g')
      plt.xlabel('Number of iterations in steps of 1000', fontweight="bold")
      plt.ylabel('Mean Squared Error', fontweight="bold")
      plt.legend()
      plt.show()

MSE1()

def MSE2():   
      """ Returns the MSEs of the training and the test set using GD with adaptive
      learning rate for a given number of iterations""" 
       
      plot_range = [1 + 1000*k for k in [0,1,2,5,10,20,30]] # x-axis of the MSE plot
      mses_test, mses_train = [], []
      for i in plot_range:
            w, mse_test, mse_train = GD_adaptive(A, b, 
                              xtrain_scaled, xtest_scaled,
                              ytrain, ytest, i) # Computes the mses for the given iteration
            
            # Adds the error of the last step to the array
            mses_test.append(mse_test) 
            mses_train.append(mse_train) 
      print('Training-MSE after ', int(i/1000), 'k iterations: ', 
            np.round(mses_train[-1],5),
            '\nTest-MSE after ', int(i/1000), 'k iterations: ', 
            np.round(mses_test[-1],5), '\n')
      
      # Plotting the MSEs for the given set of iterations
      fig = plt.figure()
      plt.title('MSE normalized GD adaptive-lr', fontweight="bold")
      plt.plot(np.array(plot_range)/1000, mses_train, label="train", c='r')
      plt.plot(np.array(plot_range)/1000, mses_test, label="test", c='g')
      plt.xlabel('Number of iterations in steps of 1000', fontweight="bold")
      plt.ylabel('Mean Squared Error', fontweight="bold")
      plt.legend()
      
      ax2 = fig.add_axes([0.4, 0.4, 0.4, 0.4])
      ax2.plot(np.array(plot_range[-4:])/1000, mses_train[-4:], label="train", c='r')
      ax2.plot(np.array(plot_range[-4:])/1000, mses_test[-4:], label="test", c='g')
      plt.show()      
      
#MSE2()

#%% 
""" Subtask 7: Polynomial features and corresponding MSEs """

def PolyFeatures(degree):
      """ Calculates the MSEs for a given polynomial feature-transformation. """
      
      poly = PolynomialFeatures(degree, include_bias=False).fit(xtrain_scaled) # Model is fitted on training set
      xtrain_poly = poly.transform(xtrain_scaled) # transform features
      xtest_poly = poly.transform(xtest_scaled) # transform features
      reg = LinearRegression(fit_intercept=False).fit(xtrain_poly, ytrain) # linear model fitted polynomial features

      train_mse = mean_squared_error(ytrain, reg.predict(xtrain_poly))
      test_mse = mean_squared_error(ytest, reg.predict(xtest_poly)) 

      return train_mse, test_mse

for i in range(2,5): # computes MSEs for degrees 2,3,4
      train_mse, test_mse = PolyFeatures(i)
      
      print('MSE training set, degree =', i, ': ',  np.round(train_mse,5),
            '\nMSE test set, degree =', i, ': ', np.round(test_mse, 5), "\n")

#%%
# Alternative versions of GD. Not part of the computation

def GD1(x, y, k):
     w0 = np.zeros((dim,1))
     for t in range(k):
       lr = 1e-6 / (1+t)  
       gr = 0
       for i in range(len(y)):
             gr += ((y[i] - w0.T @ x[i]) * x[i]).reshape(dim,1)
       w0 = w0 + lr * (2 / len(y) * gr)
     mse_test = mean_squared_error(ytest, xtest @ w0) # mse of the test-set
     mse_train = mean_squared_error(ytrain, xtrain @ w0) # mse of the training-set
      
     return w0, mse_test, mse_train

def GD2(x, y, k):
     w = np.zeros((dim,1))
     costs = []
     lr = 0.00001
     for t in range(k):
       cost_before = cost(x,y,w)  
       gr = np.zeros((dim,1))
       for i in range(len(y)):
             gr += ((y[i] - w.T @ x[i]) * x[i]).reshape(dim,1)
       w = w + lr * (2 / len(y) * gr)
       # Bold driver adaptive learning rate
       cost_new = cost(x,y,w); costs.append(cost_new)
       if cost_new < cost_before:
             lr *= 1.5
       else:
             lr *= 0.5
     return w, costs