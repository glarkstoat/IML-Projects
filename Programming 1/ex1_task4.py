"""
Wine recognition dataset
------------------------

**Data Set Characteristics:**

    :Number of Instances: 178 (50 in each of three classes)
    :Number of Attributes: 13 numeric, predictive attributes and the class
    :Attribute Information:
 		- Alcohol
 		- Malic acid
 		- Ash
		- Alcalinity of ash  
 		- Magnesium
		- Total phenols
 		- Flavanoids
 		- Nonflavanoid phenols
 		- Proanthocyanins
		- Color intensity
 		- Hue
 		- OD280/OD315 of diluted wines
 		- Proline

    - class:
            - class_0
            - class_1
            - class_2
		
    :Summary Statistics:
    
    ============================= ==== ===== ======= =====
                                   Min   Max   Mean     SD
    ============================= ==== ===== ======= =====
    Alcohol:                      11.0  14.8    13.0   0.8
    Malic Acid:                   0.74  5.80    2.34  1.12
    Ash:                          1.36  3.23    2.36  0.27
    Alcalinity of Ash:            10.6  30.0    19.5   3.3
    Magnesium:                    70.0 162.0    99.7  14.3
    Total Phenols:                0.98  3.88    2.29  0.63
    Flavanoids:                   0.34  5.08    2.03  1.00
    Nonflavanoid Phenols:         0.13  0.66    0.36  0.12
    Proanthocyanins:              0.41  3.58    1.59  0.57
    Colour Intensity:              1.3  13.0     5.1   2.3
    Hue:                          0.48  1.71    0.96  0.23
    OD280/OD315 of diluted wines: 1.27  4.00    2.61  0.71
    Proline:                       278  1680     746   315
    ============================= ==== ===== ======= =====
"""
#%%
import numpy as np
import sklearn 
from sklearn import datasets
import h5py
import matplotlib.pyplot as plt # %matplotlib qt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

# %%
# Loading the dataset
data = load_wine()
X = data['data']
Y = data['target']
feature_names = data['feature_names']
target_names = data['target_names']

# Splitting X & Y into test (30%) and training set (70%)
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, 
                                                train_size=0.7) # shuffled=True   
ytrain, ytest = ytrain.reshape(-1,1), ytest.reshape(-1,1) # avoids future complications

print('Size of dataset: ', len(X), 
      '\nSize of training set: ', len(xtrain),
      '\nSize of test set: ', len(xtest),
      '\nNumber of observed features: ', xtrain.shape[1], "\n")

# Sclaing the data via the MinMax-Scaler
scaler = MinMaxScaler().fit(xtrain) # Fitting the scalar with the training data
# Applying the scalar to test- and training data
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest) 

linearsvm = LinearSVC().fit(xtrain, ytrain) # SVM classification model

#%%
""" Subtask 1: Forward Greedy Feature Selection """
print('Subtask 1: Forward Greedy Feature Selection\n')

def fwd_selection(xtrain, ytrain, k):
        """ Returns the optimal set of features that maximize the k-fold 
        cross-validation score as well as the MSEs on training- and test set.
        Includes the error evaluation which helps the algorithm decide when to
        stop adding another feature. """
        
        dimension = xtrain.shape[1] # features in the training set
        features = list(range(dimension)) # indices of all features
        optimal_set = [] # Optional list of features
        error = np.inf # Initial error set to infinity
        
        for i in range(dimension): # Iterates through all features.
                errors = [] # CV-errors of different subsets of features
                for feature in features: # iterates trough all features 
                        subset = list(optimal_set)
                        subset.append(feature)
                        # selects the feature
                        xtrain_subset = xtrain[:,subset] 
                        # CV-score for subset of features
                        mse = np.mean(-cross_val_score(LinearSVC(), # estimator
                                                       xtrain_subset, 
                                                       ytrain,
                                                       cv=k, # folding-strategy
                                                       scoring='neg_mean_squared_error')) 
                        errors.append(mse) # adds error to list
                
                best = np.argmin(errors) # Index of subset with lowest error
                error_new = errors[best] # Error of the best subset

                if  error_new > error:
                        """ Checks whether the error of the new training-subset 
                        has increased. Necessary to decide, whether to continue
                        with adding another features or to stop because the error
                        is getting worse. """
                        
                        break # Error got worse. Too many features added!
                else: 
                        """ error_new < error: continue with adding new features 
                                to the list! """
                        # Adds feature to list of optimal features
                        optimal_set.append(features[best])
                        del features[best] # Ensures no duplicates in optimal set
                        error = error_new # Updates error. For next iteration!
                
        print("Set of features with lowest training error using forward greedy" 
              " selection:\n ", optimal_set, '\n')
        for i in optimal_set: # Prints the indices and the names of the features
                print('- ', i, 
                      " ", feature_names[i])
        
        # selects the optimal subset of features that was computed by algorithm
        xtrain_subset = xtrain[:,optimal_set] 
        # fits the model with the optimal subset of features    
        linearsvm = LinearSVC().fit(xtrain_subset, ytrain) 
        # predicted values using the newly optained model
        train_pred = linearsvm.predict(xtrain_subset)
        mse_train = mean_squared_error(ytrain, train_pred)
        print("MSE on training-set: ", np.round(mse_train,12))
        
        ytest_pred = linearsvm.predict(xtest[:,optimal_set])
        mse_test = mean_squared_error(ytest, ytest_pred)
        print("MSE on test-set: ", np.round(mse_test,5), '\n')

def fwd_selection_naive(length, xtrain, ytrain, k):
        """ Returns the set of features of size length which minimize the cost and its
        MSEs on training- and test set via k-fold cross-validation. 
                Code has been adapted from the lecture. """

        dimension = xtrain.shape[1] # features in the training set
        features = list(range(dimension)) # indices of all features
        optimal_set = [] # Optional list of features

        for i in range(length): # Iterates through all features.
                errors = [] # CV-errors of different subsets of features
                for feature in features: # iterates trough all features 
                        subset = list(optimal_set)
                        subset.append(feature)
                        # selects the feature
                        xtrain_subset = xtrain[:,subset] 
                        # CV-score for subset of features
                        mse = np.mean(-cross_val_score(LinearSVC(), # estimator
                                                       xtrain_subset, 
                                                       ytrain,
                                                       cv=k, # folding-strategy
                                                       scoring='neg_mean_squared_error')) 
                        errors.append(mse) # adds error to list
                        
                best = np.argmin(errors) # feature which minimizes the error
                optimal_set.append(features[best]) # Adds feature optimal features
                del features[best] # Ensures no duplicates in optimal set
      
        # selects the optimal subset of features that was computed by algorithm
        xtrain_subset = xtrain[:,optimal_set] 
        # fits the model with the optimal subset of features    
        linearsvm = LinearSVC().fit(xtrain_subset, ytrain) 
        # predicted training values using the newly optained model
        train_pred = linearsvm.predict(xtrain_subset)
        mse_train = mean_squared_error(ytrain, train_pred) 
        
        # predicted test values using the newly optained model
        ytest_pred = linearsvm.predict(xtest[:,optimal_set])
        mse_test = mean_squared_error(ytest, ytest_pred)

        return mse_train, mse_test, optimal_set

# MSEs for each number of selected features
mses_train, mses_test, optimal_sets = [], [], []

for i in range(1,xtest.shape[1]+1): # Iterates through all features range(1,13)
        mse_train, mse_test, optimal_set = fwd_selection_naive(i, xtrain,
                                                               ytrain, 10)
        print('Selected Features: ', i, # MSEs for every cardinality
              ', MSE(Training) = ', np.round(mse_train,5), 
              ', MSE(Test) = ', np.round(mse_test,5))
        
        mses_train.append(mse_train); mses_test.append(mse_test)
        optimal_sets.append(optimal_set)

# Best perfoming subset of features
best = np.argmin(mses_test)
optimal_set = optimal_sets[best]
print("\nSet of features with lowest test error using forward greedy selection:\n",
      optimal_set, '\n')
for i in optimal_set: # Prints the indices and the names of the features
     print('- ', i, 
          " ", feature_names[i])
print("MSE on training-set: ", np.round(mses_train[best],12))
print("MSE on test-set: ", np.round(mses_test[best],5), '\n')

# %%
""" Subtask 2: Backward Greedy Feature Selection """
print('Subtask 2: Backward Greedy Feature Selection\n')

def bwd_selection(xtrain, ytrain, k):
        """ Returns the set of features of length card which minimize the cost and its
        MSEs on training- and test set via k-fold cross-validation. """
        
        dimension = xtrain.shape[1] # number of features in the training set
        features = list(range(dimension)) # list of all potential features for the optimal_set features
        error = np.inf # Initial error set to infinity
        
        for i in range(dimension): # Every            
                errors = [] # Empty list for the cross-validation errors using the different subsets of features
                for j,c in enumerate(features):
                        subset = list(features)
                        del subset[j]
                        xtrain_subset = xtrain[:,subset]
                        
                        # computes the mean of the cross-validation errors for a specific subset of features S
                        mse = np.mean(-cross_val_score(LinearSVC(), 
                                                       xtrain_subset,
                                                       ytrain, 
                                                       cv=k, 
                                                       scoring='neg_mean_squared_error')) 
                        errors.append(mse)
                
                best_id = np.argmin(errors) # error with best S \ s_i
                error_new = errors[best_id]

                if  error_new > error:
                        break
                else:
                        del features[best_id]
                        error = error_new         
                  
        print("Set of features with lowest training error using backward greedy selection:\n ",
              features, '\n')
        for i in features:
                print('- ', i, 
                      " ", feature_names[i], '\n')

        xtrain_subset = xtrain[:,features]
        linearsvm = LinearSVC().fit(xtrain_subset, ytrain) # fits the model with the optimal subset of features    
        ytrain_pred = linearsvm.predict(xtrain_subset)
        mse_train = mean_squared_error(ytrain, ytrain_pred)
        print("MSE on training-set: ", np.round(mse_train, 3))
        
        ytest_pred = linearsvm.predict(xtest[:,features])
        mse_test = mean_squared_error(ytest, ytest_pred)
        print("MSE on test-set: ", np.round(mse_test,3), '\n')

def bwd_selection_naive(length, xtrain, ytrain, k):
        """ Returns the set of features of length card which minimize the cost and its
        MSEs on training- and test set via k-fold cross-validation. 
                Code has been adapted from the lecture. """
        
        dimension = xtrain.shape[1] # number of features in the training set
        features = list(range(dimension)) # list of all potential features for the optimal_set features
        
        for i in range(dimension-length): # dimension-card features considered
                errors = [] # Empty list for the cross-validation errors
                for j,c in enumerate(features):
                        subset = list(features)
                        del subset[j]
                        # new subset of features
                        xtrain_subset = xtrain[:,subset]
                        # cv-scores
                        mse = np.mean(-cross_val_score(LinearSVC(),
                                                       xtrain_subset,
                                                       ytrain,
                                                       cv=k, 
                                                       scoring='neg_mean_squared_error')) # computes the mean of the cross-validation errors for a specific subset of features S
                        errors.append(mse)
                
                best_id = np.argmin(errors) # error with best S \ s_i
                del features[best_id]
                  
        xtrain_subset = xtrain[:,features]
        linearsvm = LinearSVC().fit(xtrain_subset, ytrain) # fits the model with the optimal subset of features    
        ytrain_pred = linearsvm.predict(xtrain_subset)
        mse_train = mean_squared_error(ytrain, ytrain_pred)
        
        ytest_pred = linearsvm.predict(xtest[:,features])
        mse_test = mean_squared_error(ytest, ytest_pred)
        
        return mse_train, mse_test, features

# MSEs for each number of selected features
mses_train, mses_test, optimal_sets = [], [], []

for i in range(xtest.shape[1], 0, -1): # Iterates through all features range(13,1)
        mse_train, mse_test, optimal_set = bwd_selection_naive(i, xtrain,
                                                               ytrain, 10)
        print('Selected Features: ', i, # MSEs for every cardinality
              ', MSE(Training) = ', np.round(mse_train,5), 
              ', MSE(Test) = ', np.round(mse_test,5))
        
        mses_train.append(mse_train); mses_test.append(mse_test)
        optimal_sets.append(optimal_set)

# Best perfoming subset of features
best = np.argmin(mses_test)
optimal_set = optimal_sets[best]
print("\nSet of features with lowest test error using backward greedy selection:\n",
      optimal_set, '\n')
for i in optimal_set: # Prints the indices and the names of the features
     print('- ', i, 
          " ", feature_names[i], '\n')
print("MSE on training-set: ", np.round(mses_train[best],12))
print("MSE on test-set: ", np.round(mses_test[best],5), '\n')

#%%
""" Subtask 3: Feature Importance. Only 4/13 features allowed """

# Forward selection

mse_train, mse_test, optimal_set = fwd_selection_naive(4, xtrain,
                                                               ytrain, 10)

# Best perfoming subset of features
print("\nSet of 4 features with lowest test error using forward "
      "greedy selection:\n",
      optimal_set, '\n')
for i in optimal_set: # Prints the indices and the names of the features
     print('- ', i, 
          " ", feature_names[i], '\n')
print("MSE on training-set: ", np.round(mse_train,12))
print("MSE on test-set: ", np.round(mse_test,5), '\n')

## Backward selection
mse_train, mse_test, optimal_set = bwd_selection_naive(4, xtrain,
                                                ytrain, 10)

# Best perfoming subset of features
print("\nSet of 4 features with lowest test error using backward "
      "greedy selection:\n",
      optimal_set, '\n')
for i in optimal_set: # Prints the indices and the names of the features
     print('- ', i, 
          " ", feature_names[i], '\n')
print("MSE on training-set: ", np.round(mse_train,12))
print("MSE on test-set: ", np.round(mse_test,5), '\n')
# %%