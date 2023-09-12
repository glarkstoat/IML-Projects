#%%
import numpy as np
import h5py
import matplotlib.pyplot as plt # %matplotlib qt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from NeuralNetwork import NeuralNetwork

np.random.seed(0)

sns.set_style('darkgrid')
sns.set(color_codes=True) # Nice layout.
sns.set_context('paper')

# Loading of the dataset
hf = h5py.File('regression.h5', 'r')
xtrain = np.array(hf.get('x_train'))
ytrain = np.array(hf.get('y_train'))
xtest = np.array(hf.get('x_test'))
ytest = np.array(hf.get('y_test'))
hf.close()

# Normalization
scaler = MinMaxScaler().fit(xtrain) 
xtrain = scaler.transform(xtrain) 
xtest = scaler.transform(xtest)

print('Size of training set: ', len(xtrain),
      '\nSize of test set: ', len(xtest),
      '\nNumber of observed features: ', xtrain.shape[1], "\n")

# 3-d plot of the test-set
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xtest[:,0], xtest[:,1], ytest, s=0.05, alpha=1, c="brown") 
plt.show()

# Necessary for this implementation to have data set with 
# shape = (number of features, number of samples)
xtest = xtest.T
ytest = ytest.T
xtrain = xtrain.T
ytrain = ytrain.T

#%%
np.random.seed(0)

""" Subtask 2.3 """
print("Subtask 2.3\n")

# Parameters for ANN
n_hidden = 2
n_neurons = 10

ann = NeuralNetwork(xtrain, ytrain, n_hidden, n_neurons)
ann.fit(xtest, ytest, max_iter=100, lr=0.15, optimizer="Adam", method="Finite Difference",
        epsilon=0.001)
loss1 = ann.loss_test
loss1_train = ann.loss_train

try:
    # Calculating the MSEs
    mse_test = mean_squared_error(ytest, ann.predict(xtest))
    mse_train = mean_squared_error(ytrain, ann.predict(xtrain))
    
    print('MSE of training set: ', np.round(mse_train, 5),
    '\nMSE of test set: ', np.round(mse_test, 5), 
    '\n --> Using Adam with Finite Difference Approximation and Relu activation. \n')
except:
    print('ERROR: MSEs are diverging!\n')
    
#%%
def relu(Z):
    return np.maximum(0, Z)

def derivate_relu(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return x
np.random.seed(0)

""" Subtask 2.4 """
print("Subtask 2.4\n")

# Parameters for ANN
n_hidden = 2
n_neurons = 10

# Initializing the network
ann = NeuralNetwork(xtrain, ytrain, n_hidden, n_neurons)

# Training the network
ann = NeuralNetwork(xtrain, ytrain, 2, 10)
ann.fit(xtest, ytest, max_iter=100, lr=0.15, optimizer="Adam", method="Gradient")
loss2 = ann.loss_test
loss2_train = ann.loss_train

try:
    # Calculating the MSEs
    mse_test = mean_squared_error(ytest, ann.predict(xtest))
    mse_train = mean_squared_error(ytrain, ann.predict(xtrain))
    
    print('MSE of training set: ', np.round(mse_train, 5),
    '\nMSE of test set: ', np.round(mse_test, 5), 
    '\n --> Using Adam with gradients and relu activation function. \n')
except:
    print('ERROR: MSEs are diverging!\n')
    
# %%
""" Subtask 2.5 """
print("Subtask 2.5\n")

def custom(z):
    return z**2

def d_custom(z):
    return 2*z

np.random.seed(0)

# Parameters for ANN
n_hidden = 2
n_neurons = 10

# Initializing the network
ann = NeuralNetwork(xtrain, ytrain, n_hidden, n_neurons)

# Training the network
ann.fit(xtest, ytest, max_iter=100, lr=0.15, optimizer="Adam", method="Gradient",
        activation1=custom, d_activation1=d_custom)
loss3 = ann.loss_test
loss3_train = ann.loss_train

try:
    # Calculating the MSEs
    mse_test = mean_squared_error(ytest, ann.predict(xtest, activation1=custom, activation2=relu))
    mse_train = mean_squared_error(ytrain, ann.predict(xtrain, activation1=custom, activation2=relu))
    
    print('MSE of training set: ', np.round(mse_train, 5),
        '\nMSE of test set: ', np.round(mse_test, 5), 
        '\n--> Using Adam with gradients and custom activation function on first layer. \n')
except:
    print('ERROR: MSEs are diverging!\n')
    
#%%
sns.set_palette("rocket_r")

# Comparison
fig = plt.figure()

plt.grid(True, linestyle="dotted", linewidth=1)
plt.title('MSE Comparison - ANN', fontsize=12 ,
            fontweight='bold')
plt.plot(range(len(loss1[:100])), loss1[:100], lw=2.5, 
         label="Adam+FD+Relu (Test)", ls="-", alpha=0.7)
plt.plot(range(len(loss1[:100])), loss1_train[:100], lw=1, 
         label="Adam+FD+Relu (Train)", c='black', ls="dotted")

plt.plot(range(len(loss2[:100])), loss2[:100], lw=2.5,
                    ls='-',  label='Adam+GD+Relu (Test)', alpha=0.7)
plt.plot(range(len(loss2[:100])), loss2_train[:100], lw=0.7, 
         label="Adam+GD+Relu (Train)", c='black', ls="dashed")

plt.plot(range(len(loss3[:100])), loss3[:100], lw=2.5,
                    ls='-',  label='Adam+GD+Custom (Test)',  alpha=0.7)
plt.plot(range(len(loss3[:100])), loss3_train[:100], lw=0.7, 
         label="Adam+GD+Custom (Train)", c='black', ls="dashdot")

plt.xlabel("Number of Iterations", fontweight='bold')
plt.ylabel("Mean Squared Error (MSE)" , fontweight='bold')
plt.legend()

plt.savefig('2.png', dpi=1200)
plt.show()
# %%