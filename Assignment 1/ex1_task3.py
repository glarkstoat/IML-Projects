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

# Splitting X & Y into test (70%) and training set (30%)
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.7, train_size=0.3) # shuffled by default   

# Scaling the data 
scaler = MinMaxScaler().fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

# %%
""" Subtask 1: Baseline """
print("Subtask 1: Baseline")

# Fitting linear regression model to the training data
reg = LinearRegression().fit(xtrain, ytrain) 

# mse of the model using test set
mse_test = mean_squared_error(ytest, reg.predict(xtest))
# mse of the model using training set
mse_train = mean_squared_error(ytrain, reg.predict(xtrain)) 

print('MSE of training set: ', np.round(mse_train,5),
    '\nMSE of test set: ', np.round(mse_test,5))

# Polynomial features of degree=2
def PolyFeatures(degree, xtrain, xtest, ytrain, ytest):
      """ Calculates the MSEs for a given polynomial feature-transformation. """
      
      # Fit and feature transformation
      poly = PolynomialFeatures(degree).fit(xtrain) 
      xtrain_poly = poly.transform(xtrain)
      xtest_poly = poly.transform(xtest) 
      # linear model fitted polynomial features
      reg = LinearRegression().fit(xtrain_poly, ytrain) 

      train_mse = mean_squared_error(ytrain, reg.predict(xtrain_poly))
      test_mse = mean_squared_error(ytest, reg.predict(xtest_poly)) 

      return train_mse, test_mse

train_mse, test_mse = PolyFeatures(2, xtrain, xtest, ytrain, ytest)
      
print('MSE training set, degree =', 2, ': ',  np.round(train_mse,5),
    '\nMSE test set, degree =', 2, ': ', np.round(test_mse, 5))

# %%
""" Subtask 2: Implementing Cross-validation """
print('Subtask 2: Implementing Cross-validation')

def cross_validation(estimator, xtrain, ytrain, k):
    """ Manual k-fold cross-validation on training set using a given estimator. 
        Uses array slicing to define the training- and the validation sets. 
        Due to implementation it's possible that the remainder of 
        len(xtrain) % k samples will not be used for CV. """
    
    fold_size = int(len(xtrain) / k) # size of one fold
    mses = [] # contains the calculated MSEs

    for i in range(k):
        """ Iterates through all possible configurations for training- and
            validation sets. Estimator is fitted with the corresponding 
            training set for each iteration. xtrain[0] corresponds to the 
            first row of xtrain. """
            
        if i == 0: 
            """ First iteration. Validation set is the first fold, i.e. 
                        the first fold_size rows """
            reg = estimator.fit(xtrain[fold_size:],
                                        ytrain[fold_size:])
        elif i == k-1: 
            """ Last iteration. Validation set is the last fold, i.e. 
                        the last fold_size rows """
            reg = estimator.fit(xtrain[:i*fold_size],
                                        ytrain[:i*fold_size])
        else:
            """ Concatenates the array slices that make up the training set
            i.e. the whole training set except the validation set.
            Necessary if the validation set is in the middle, i.e. neither the 
            first nor the last fold. """
            reg = estimator.fit(np.concatenate((xtrain[: i * fold_size], 
                                        xtrain[(i+1) * fold_size :])),
                                        
                                    np.concatenate((ytrain[: i * fold_size], 
                                        ytrain[(i+1) * fold_size :])))
        # Range of row indices, where validation set is currently located
        test_range = range(i * fold_size, (i+1) * fold_size)
        
        # MSE calculated w.r.t the validation set
        mses.append(mean_squared_error(ytrain[test_range], 
                                    reg.predict(xtrain[test_range])))
    return np.mean(mses) # cross-validation score (average of all MSEs)

print('MSE on the training data using k-fold cross-validation: ',
      cross_validation(LinearRegression(), xtrain, ytrain, 10))

#%%
""" Subtask 3: Model Selection """
print('Subtask 3: Model Selection')
# (1) Select the best degree with respect to the MSE on the training data.
mses1 = []
for i in [1,2,3,4,5]:
    # MSEs for every degree
    train_mse, test_mse = PolyFeatures(i, xtrain, xtest, 
                                                    ytrain, ytest)
    mses1.append(train_mse)
    print('(1) MSE training set, degree =', i, ': ', np.round(train_mse,5))

# (2) Splitting training data into 50/50 split
xtrain_new, xtrain_validation, ytrain_new, ytrain_validation = train_test_split(xtrain,
                                        ytrain, test_size=0.5, train_size=0.5) # shuffled by default   
mses2 = []
for i in [1,2,3,4,5]:
    # MSEs w.r.t. the new validation set for every degree
    train_mse, validation_mse = PolyFeatures(i, xtrain_new, xtrain_validation, 
                                                    ytrain_new, ytrain_validation)
    mses2.append(validation_mse)
    print('(2) MSE on validation set, degree =', i, ': ',
          np.round(validation_mse,5))

# (3) Using cross-validation
mses3 = []
for i in [1,2,3,4,5]:
    # 10-fold cv-score on the training data for every degree 
    cv = cross_validation(LinearRegression(), 
                          PolynomialFeatures(i).fit_transform(xtrain),
                          ytrain,
                          10)
    mses3.append(cv)
    print('(3) MSE training set (CV), degree =', i, ': ',  np.round(cv,5))

""" Evaluating MSEs of different optimal degrees obtained in the previous
    taks"""
    
# (1) Test set MSE for ideal degree
best = np.argmin(mses1); optimal_degree = best + 1 # degree with lowest MSE
train_mse, test_mse = PolyFeatures(optimal_degree, xtrain, xtest, 
                                                    ytrain, ytest)
print('(1) MSE on test set, optimal degree =', optimal_degree, ': ',
          np.round(test_mse,5))

# (2) Test set MSE for ideal degree
best = np.argmin(mses2); optimal_degree = best + 1 # degree with lowest MSE
train_mse, test_mse = PolyFeatures(optimal_degree, xtrain, xtest, 
                                                    ytrain, ytest)
print('(2) MSE on test set, optimal degree =', optimal_degree, ': ',
          np.round(test_mse,5))

# (3) Test set MSE for ideal degree
best = np.argmin(mses3); optimal_degree = best + 1 # degree with lowest MSE
train_mse, test_mse = PolyFeatures(optimal_degree, xtrain, xtest, 
                                                    ytrain, ytest)
print('(3) MSE on test set, optimal degree =', optimal_degree, ': ',
          np.round(test_mse,5))
# %%