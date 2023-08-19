#%%
# Import Packages

import numpy as np
from time import time
from random import sample
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

showPlot = False

'''
- Task 1 -
Subtask 1 - Data Loading and Data Preparation
'''
print('Subtask 1 - Data Loading and Data Preparation\n')
t0 = time()

# Load in Data
LFW = fetch_lfw_people(min_faces_per_person=30, resize=0.5)
samples, height, width = LFW.images.shape

# How many different people are in the data?
print(f'There are {len(LFW.target_names)} different people in the dataset.')

# How many images are in the data?
print(f'There are {samples} images in the dataset.')

# What are the sizes of the images?
print(f'There size of the images is h = {height}, w = {width}.')

# Plot images of ten different people in the data set
nRows, nCols = 2, 5
plt.figure(figsize=(8 * nRows, 2 * nCols))
plt.subplots_adjust(bottom=0.1, left=.05, right=.95, top=0.9)
pictures = sample(range(0, 34), 10)

for i, j in enumerate(pictures):
    for k in range(len(LFW.target)):
        if LFW.target[k] == j:
            plt.subplot(nRows, nCols, i + 1)
            plt.title(LFW.target_names[LFW.target[k]], size=16)
            plt.imshow(LFW.images[k], cmap=plt.cm.gray)
            plt.xticks(())
            plt.yticks(())
            break

#plt.savefig('Plots/10_diff_faces.png', dpi=150)
if showPlot == True:
    plt.show()

# Split the Data into Training and Validation Set with Class Labels for stratify
xTrain, xTest, yTrain, yTest = train_test_split(LFW.data, LFW.target, 
                            train_size=0.7, random_state=42, stratify=LFW.target)
#%%
print(f'\nRuntime for Task 1.1: {time()-t0:.3f}s')

'''
- Task 1 -
Subtask 2 - Write a class that does PCA and do logistic regression
'''
print('\nSubtask 2 - Write a class that does PCA and do logistic regression\n')
t0 = time()

class PCA:

    '''
    Class for the PCA implementation
    Inspired by the sklearn implementation
    Paramters: numComp, data
    '''

    def __init__(self, numComp, X):
        self.numComp = numComp
        self.X = X
        self.samples_, self.features_ = X.shape

    
    def fit(self):

        # Center the data
        self.mean = np.mean(self.X, axis=0)
        X = self.X - self.mean

        # Get components with SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # Calculate various parameters of interest
        explainedVariance = (S ** 2) / (self.X.shape[0] - 1)

        self.eigvals_ = S[:self.numComp]
        self.components_ = Vt[:self.numComp]
        self.explainedVariance_ = explainedVariance[:self.numComp]
        self.explainedVarianceRatio_ = explainedVariance[:self.numComp] / np.sum(explainedVariance)

        return U, S, Vt


    def fitTransform(self):

        # Call the fit function
        U, S, Vt = self.fit()

        # Transform the data
        U = U[:, :self.numComp]
        U *= S[:self.numComp]

        return U

    
    def transform(self, X):

        try:
            # Transform new data with the fitted PCA Components
            if self.mean is not None:
                X = X - self.mean
                transformedX = np.dot(X, self.components_.T)
        
        except:
            raise AttributeError('PCA needs to be fitted first.')
            
        return transformedX

    
    def reconstruct(self, X):

        # Reconstruct the original data from the principal components
        transformedX = np.dot(X, self.components_) + self.mean
        
        return transformedX


d = [5, 10, 20, 40, 80, 160, 320, 640]
for components in d:

    pca = PCA(components, xTrain)
    pcaTrain = pca.fitTransform()
    pcaTest = pca.transform(xTest)

    reconstructedData = pca.reconstruct(pcaTrain)
    reconstructedFaces = reconstructedData.reshape((len(reconstructedData), height, width))

    if components == d[0]:
        
        plt.figure(figsize=(8 * nRows, 2 * nCols))
        plt.subplots_adjust(bottom=0.1, left=.05, right=.95, top=0.9)
        
        for i in range(d[0]):
            plt.subplot(nRows, nCols, i + 1)
            plt.title(f'Principal Component {i+1}', size=16)
            plt.imshow((pca.components_[i] + pca.mean).reshape((height, width)), cmap=plt.cm.gray)
            plt.xticks(())
            plt.yticks(())
        
        for i in range(d[0]):
            plt.subplot(nRows, nCols, i + 6)
            plt.title(f'Principal Component {i+1}\n No Mean Correction', size=16)
            plt.imshow((pca.components_[i]).reshape((height, width)), cmap=plt.cm.gray)
            plt.xticks(())
            plt.yticks(())

        #plt.savefig('Plots/PCA_Components.png', dpi=150)
        if showPlot == True:
            plt.show()
    
    plt.figure(figsize=(8 * nRows, 2 * nCols))
    plt.subplots_adjust(bottom=0.1, left=.05, right=.95, top=0.9)

    for i, j in enumerate(pictures):
        for k in range(len(yTrain)):
            if yTrain[k] == j:
                plt.subplot(nRows, nCols, i + 1)
                plt.title(LFW.target_names[yTrain[k]], size=16)
                plt.imshow(reconstructedFaces[k], cmap=plt.cm.gray)
                plt.xticks(())
                plt.yticks(())
                break

    #plt.savefig(f'Plots/10_faces_PCA{components}.png', dpi=150)
    if showPlot == True:
        plt.show()
    
    scaler = MinMaxScaler()
    pcaTrain, pcaTest = scaler.fit_transform(pcaTrain), scaler.transform(pcaTest)

    regr = LogisticRegression(max_iter=1000)
    regr.fit(pcaTrain, yTrain)
    predTrain = regr.predict(pcaTrain)
    predTest = regr.predict(pcaTest)

    print(f'Classification Accuracy w/ {components} PCA components for the training set: {np.mean(predTrain == yTrain):.5f}')
    print(f'Classification Accuracy w/ {components} PCA components for the test set: {np.mean(predTest == yTest):.5f}\n')

print(f'Runtime for Task 1.2: {time()-t0:.3f}s')
'''
- Task 1 -
Subtask 3 - Use MLPRegressor as autoencoder
'''

#%%
print('\nSubtask 3 - Use MLPRegressor as autoencoder\n')
t0 = time()

def encode(X, mlp):
    
    z = X
    for i in range(len(mlp.coefs_) // 2):
        z = z @ mlp.coefs_[i] + mlp.intercepts_[i]
        z = np.maximum(z, 0)

    return z


def decode(Z, mlp):

    z = Z
    for i in range(len(mlp.coefs_) // 2, len(mlp.coefs_)):
        z = z @ mlp.coefs_[i] + mlp.intercepts_[i]
        z = np.maximum(z, 0)

    return z


D = [40]
a, b = 500, 100

for d in D:
    
    print(f'Start encoding w/ {d} dimensions')
    mlp = MLPRegressor(hidden_layer_sizes=((a, b, d, b, a)), activation="relu", max_iter=300,
                       random_state=0)
    mlp.fit(xTrain, xTrain)

    encodedTrain, encodedTest = encode(xTrain, mlp), encode(xTest, mlp)
    print('Finished encoding')

    decodedTrain = decode(encodedTrain, mlp)
    decodedTrain = decodedTrain.reshape(len(decodedTrain), height, width)
    
    plt.figure(figsize=(8 * nRows, 2 * nCols))
    plt.subplots_adjust(bottom=0.1, left=.05, right=.95, top=0.9)

    for i, j in enumerate(pictures):
        for k in range(len(yTrain)):
            if yTrain[k] == j:
                plt.subplot(nRows, nCols, i + 1)
                plt.title(LFW.target_names[yTrain[k]], size=16)
                plt.imshow(decodedTrain[k], cmap=plt.cm.gray)
                plt.xticks(())
                plt.yticks(())
                break
    
    #plt.savefig(f'Plots/autoencoder_faces_{d}.png', dpi=150)
    if showPlot == True:
        plt.show()
    '''
    scaler = MinMaxScaler()
    encodedTrain, encodedTest = scaler.fit_transform(encodedTrain), scaler.fit_transform(encodedTest)

    print('Start Regression')
    regr = LogisticRegression(max_iter=1000)
    regr.fit(encodedTrain, yTrain)
    predTrain = regr.predict(encodedTrain)
    predTest = regr.predict(encodedTest)
    print('Finished Regression')

    print(f'Classification Accuracy w/ {d} dimensions for the training set: {np.mean(predTrain == yTrain):.5f}')
    print(f'Classification Accuracy w/ {d} dimensions for the test set: {np.mean(predTest == yTest):.5f}\n')
    '''
#print(f'Runtime for Task 1.3: {time()-t0:.3f}s')

# %%
