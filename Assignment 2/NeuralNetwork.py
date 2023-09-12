#%%
import numpy as np
import matplotlib.pyplot as plt # %matplotlib qt
import datetime
from sklearn.metrics import mean_squared_error
import copy 
from _stochastic_optimizers import AdamOptimizer
import seaborn as sns
sns.set_palette("rocket_r")

def Relu(Z):
    return np.maximum(0, Z)

def d_Relu(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return x

class NeuralNetwork:
    """
    Initializes Arificial Neural Network (ANN) with the specified parameters. 
    Can be used to approximate any given linear- or non-linear data by learning via 
    backpropagation and predicting via forward propagation.
    
    Parameters
    ----------
    xtrain : array of shape (n_features, n_samples)
            Training data.

    ytrain : array of shape (n_targets, n_samples)
            Training targets.
            
    n_hidden : int, default=2
        Number of hidden layers.
        
    n_neurons : int, default=10
        Number of neurons per hidden layer. In this configuration every hidden layer has 
        an equal number of neurons.

    Attributes
    ----------
    W : dictionary containing all weight matrices.

    b : dictionary containing all bias matrices.
    """

    def __init__(self, xtrain, ytrain, n_hidden=2, n_neurons=10):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        self.W = {}
        self.b = {}
        self.initilize_weights_biases()
        self.loss_test = []
        self.loss_train = []

    def initilize_weights_biases(self):
        """
        Builds the weights and biases for every layer of the network. 
        Shapes of the matricies depend on their location.
        """

        for i in range(1, self.n_hidden + 2):
            if i == 1: # first layer
                self.W["W{}".format(i)] = np.random.normal(0, 1, (self.n_neurons, self.xtrain.shape[0]))
                self.b["b{}".format(i)] = np.zeros((self.n_neurons, 1))
                
            elif i == self.n_hidden + 1: # last layer
                self.W["W{}".format(i)] = np.random.normal(0, 1, (self.ytrain.shape[0], self.n_neurons))
                self.b["b{}".format(i)] = np.zeros((self.ytrain.shape[0], 1))
                
            else: # every layer in between
                self.W["W{}".format(i)] = np.random.normal(0, 1, (self.n_neurons, self.n_neurons))
                self.b["b{}".format(i)] = np.zeros((self.n_neurons, 1))
            
    def update_parameters(self, dws, dbs, lr):
        """ Updates for weights and biases of the given layer """
        
        for i in range(self.n_hidden + 1, 0, -1):
            self.W["W{}".format(i)] -= lr * dws["dw{}".format(i)] / self.xtrain.shape[1]
            self.b["b{}".format(i)] -= lr * dbs["db{}".format(i)] / self.xtrain.shape[1]

    def update_parameters_adam(self, gradients):
        """ Updates the weights and biases via the corrected adam gradients """
        
        # From last to first layer goes list gradients, reshapes lists to original shape,
        # updates the parameters and deletes them from the list
        for i in range(self.n_hidden + 1, 0, -1):
            length = self.W["W{}".format(i)].shape[0] * self.W["W{}".format(i)].shape[1]
            self.W["W{}".format(i)] += np.reshape(gradients[: length], self.W["W{}".format(i)].shape)
            
            del gradients[: length]
        
        for i in range(self.n_hidden + 1, 0, -1):
            length = self.b["b{}".format(i)].shape[0] * self.b["b{}".format(i)].shape[1]
            self.b["b{}".format(i)] += np.reshape(gradients[: length], self.b["b{}".format(i)].shape)
            
            del gradients[: length]   

    def forward_propagation(self, X, W, b, activation1=Relu, activation2=Relu):
        """
        Predicts the outcome of the model by using the calculated weights and biases. 
        """
        
        z = {}; v = {}
        for i in range(1, self.n_hidden + 2):
            if i == 1: # first layer
                z["z{}".format(i)] = np.dot(np.hstack((b["b{}".format(i)], 
                                                    W["W{}".format(i)])), 
                    np.vstack((np.ones((1, X.shape[1])), X))
                    )
                # Applies the activation function
                v["v{}".format(i)] = activation1(z["z{}".format(i)])
                
            elif i < self.n_hidden + 1: # every layer in between
                z["z{}".format(i)] = np.dot(np.hstack((b["b{}".format(i)], 
                                                    W["W{}".format(i)])), 
                    np.vstack((np.ones((1, X.shape[1])), 
                            v["v{}".format(i-1)])),
                    )
                # Applies the activation function
                v["v{}".format(i)] = activation2(z["z{}".format(i)])
                
            else: # last layer
                f = np.dot(np.hstack((b["b{}".format(i)], W["W{}".format(i)])), 
                np.vstack((np.ones((1, X.shape[1])), v["v{}".format(i-1)])), 
                )

        return f, z, v
    
    def back_propagation(self, X, Y, activation1, d_activation1,
                         activation2, d_activation2):
        """
        Trains the model by calculating the gradients w.r.t. weights and biases and 
        returns the dictionaries of gradients from weights and biases.
        """
        
        # Output of the model via FP        
        f, z, v = self.forward_propagation(X, self.W, self.b, 
                                           activation1, activation2)
        
        # Containers for gradients
        dws = {}; dbs = {}
        
        # Iterates from last- to first hidden layer 
        for i in range(self.n_hidden + 1, 0, -1):
            if i == self.n_hidden + 1: # last layer
                
                delta = 2 * (f - Y)
                dw = np.dot(delta, v["v{}".format(i-1)].T)
                # Adding up the rows of the delta array
                db = np.sum(delta, axis=1, keepdims=True)
                
                # Adding calculated gradients to dictionary
                dws["dw{}".format(i)] = dw
                dbs["db{}".format(i)] = db

            elif (i < self.n_hidden + 1) & (i != 1): # every layer in between
                
                delta = np.multiply(d_activation2(z["z{}".format(i)]), 
                            np.dot(self.W['W{}'.format(i+1)].T, delta))
                dw = np.dot(delta, v["v{}".format(i-1)].T)
                # Adding up the rows of the delta array
                db = np.sum(delta, axis=1, keepdims=True)
                
                # Adding calculated gradients to dictionary
                dws["dw{}".format(i)] = dw
                dbs["db{}".format(i)] = db
                                
            elif i == 1: # first layer
                
                delta = np.multiply(d_activation1(z["z{}".format(i)]), 
                            np.dot(self.W['W{}'.format(i+1)].T, delta))
                dw = np.dot(delta, X.T)
                db = np.sum(delta, axis=1, keepdims=True)
                
                # Adding calculated gradients to dictionary                
                dws["dw{}".format(i)] = dw
                dbs["db{}".format(i)] = db
        
        return dws, dbs
                
    def l2_loss(self, X, y_true, W, b, activation1=Relu, activation2=Relu):
        """ Calculates the predicted values via forward propagation and computes 
            the L2-Loss """
        f, *dummy = self.forward_propagation(X, W, b, activation1, activation2)
            
        return np.sum((f - y_true)**2)
    
    def back_propagation_FD(self, X, Y, activation1=Relu, d_activation1=d_Relu,
                         activation2=Relu, d_activation2=d_Relu, epsilon=0.1):
        """
        Trains the model by calculating the finite difference approximations and 
        returns the dictionaries of gradients from weights and biases.
        """
        
        # Containers for the results
        dws = {}; dbs = {}
        # Iterates from last- to first hidden layer
        for i in range(self.n_hidden + 1, 0, -1):
            
            """ Iteration through the entire weights/bias array of the given layer and adds/subtracts
            epsilon to one element at a time to calculate the respective loss.
            For every loss-computation a copy of the current weight and bias arrays are 
            created via deepcopy. """
             
            dw = np.zeros_like(self.W["W{}".format(i)])    
            for j in range(self.W["W{}".format(i)].shape[0]): # row
                for k in range(self.W["W{}".format(i)].shape[1]): # column
                        
                    W_prime = copy.deepcopy(self.W)                           
                    W_prime["W{}".format(i)][j,k] += epsilon
                    loss_prime = self.l2_loss(X, Y, W_prime, self.b)
                    
                    W_2prime = copy.deepcopy(self.W)
                    W_2prime["W{}".format(i)][j,k] -= epsilon
                    loss_2prime = self.l2_loss(X, Y, W_2prime, self.b)
                    
                    dl_dw_jk = (loss_prime - loss_2prime) / (2*epsilon)
                    
                    # Overwrites the zero value in dw with the result
                    dw[j,k] = dl_dw_jk
                    
            # Adding the gradient to the dictionary
            dws["dw{}".format(i)] = dw
            
            db = np.zeros_like(self.b["b{}".format(i)])    
            for j in range(self.b["b{}".format(i)].shape[0]): # row
                for k in range(self.b["b{}".format(i)].shape[1]): # column
                        
                    b_prime = copy.deepcopy(self.b)
                    b_prime["b{}".format(i)][j,k] += epsilon                
                    loss_prime = self.l2_loss(X, Y, self.W, b_prime)
                    
                    b_2prime = copy.deepcopy(self.b)
                    b_2prime["b{}".format(i)][j,k] -= epsilon                
                    loss_2prime = self.l2_loss(X, Y, self.W, b_2prime)                
                    
                    dl_db_jk = (loss_prime - loss_2prime) / (2*epsilon)
                    
                    # Overwrites the zero value in db with the result
                    db[j,k] = dl_db_jk

            # Adding the gradient to the dictionary
            dbs["db{}".format(i)] = db
            
        return dws, dbs

    def fit(self, xtest, ytest, batch_size=64, lr=0.0001, max_iter=1000, method="Gradient",
            optimizer="SGD", activation1=Relu, d_activation1=d_Relu, activation2=Relu, 
            d_activation2=d_Relu, epsilon=0.1):
        """
        Fit neural network.

        Parameters
        ----------
        xtest : array of shape (n_features, n_samples)
            Test data. Necessary to compute the MSE curves of training vs. test data.

        ytest : array of shape (n_targets, n_samples)
            Test targets.
    
        batch_size : int, default=64
            Batch size for Stochastic Gradient Descent (SGD).

        lr : float, default=0.001
            Learning rate for SGD.

        method : string, default="gradient"
            Method for calculating the updates of the weights and the biases.
            
            "Gradient":          Uses the standard gradient updates
            "Finite Difference": Uses the finite difference approximation instead of the 
                                 gradient updates
                                 
        optimizer : string, default="SGD"
            Determines whether to use the standard SGD for optimization or to include 
            additional schemes.
            
            "SGD": Uses the standard SGD 
            "Adam": Uses the Adam optimizer additionally for correcting the gradiant-updates
        
        epsilon : float, default=0.1
            Tuning parameter for the finite differences approximation.
            
        activation : function, default=Relu
            Activation functions for the first layer (activation1) and all other layers (activation2).
        
        d_activation : function, default=d_Relu 
            Partial derivatives of the activation functions.
             
        Returns
        -------
        self : returns an instance of self.
        """
        
        if optimizer == "Adam": 
                  
            # sum of elements of last layer to first layer
            W_sum = [self.W["W{}".format(i)].flatten().tolist() for i in range(self.n_hidden + 1, 0, -1)]
            b_sum = [self.b["b{}".format(i)].flatten().tolist() for i in range(self.n_hidden + 1, 0, -1)]

            # Flattens a list of lists
            flatten = lambda t: [item for sublist in t for item in sublist]

            # Instantiates the adam optimizer 
            opt = AdamOptimizer(flatten(W_sum)+flatten(b_sum), lr, 0.9, 0.999, 1e-08)
        
        start = datetime.datetime.now()
        
        # Main Calculation
        for t in range(1,max_iter+1):

            # Pick k random samples from training set
            try:
                ran = np.random.choice(range(0, self.xtrain.shape[1]), batch_size, replace=False)
            except: # if batch_size > number of samples
                ran = np.random.choice(range(0, self.xtrain.shape[1]), batch_size, replace=True)

            if optimizer == "Adam":
            
                # Train the model via backpropagation    
                if method == "Finite Difference":
                    dws, dbs = self.back_propagation_FD(self.xtrain[:,ran], self.ytrain[:,ran], activation1, 
                                        d_activation1, activation2, d_activation2, 
                                        epsilon)
                else: # method = "Gradient"
                    dws, dbs = self.back_propagation(self.xtrain[:,ran], self.ytrain[:,ran], activation1, 
                                        d_activation1, activation2, d_activation2)
            
                # Converts dictionaries into lists
                dw_sum = [dws["dw{}".format(i)].flatten().tolist() for i in range(self.n_hidden + 1, 0, -1)]
                db_sum = [dbs["db{}".format(i)].flatten().tolist() for i in range(self.n_hidden + 1, 0, -1)]

                # Get respective updates for gradients
                gradients = opt._get_updates(flatten(dw_sum)+flatten(db_sum)).copy()

                # Perfom updates using corrected gradients
                self.update_parameters_adam(gradients)
            
            else: # normal SGD
                if method == "Finite Difference":
                    dws, dbs = self.back_propagation_FD(self.xtrain[:,ran], self.ytrain[:,ran], activation1, 
                                    d_activation1, activation2, d_activation2, 
                                    epsilon)
                else: # method = "Gradient"
                    dws, dbs = self.back_propagation(self.xtrain[:,ran], self.ytrain[:,ran], activation1, 
                                        d_activation1, activation2, d_activation2)
                
                # Update weights and biases
                self.update_parameters(dws, dbs, lr)
                
            # Prediction for training set and MSE
            f, *cache = self.forward_propagation(self.xtrain, self.W, self.b, 
                                                 activation1, activation2)    
            try:
                loss = mean_squared_error(self.ytrain, f)
                self.loss_train.append(loss)
            except:
                print('ERROR: Trainig MSE is diverging! Try using a different learning rate or '
                      'a different batch size.\n ')
                return self
            
            # predict the values for test set and MSE
            f, *cache = self.forward_propagation(xtest, self.W, self.b, 
                                                 activation1, activation2)
            try:
                self.loss_test.append(mean_squared_error(ytest, f))
            except:
                print('ERROR: Test MSE is diverging! Try using a different learning rate or '
                      'a different batch size.\n ')                
                return self
        
        # MSE plot
        fig = plt.figure()
        runtime = (datetime.datetime.now() - start).total_seconds() 
        print('Runtime of computation {}s using {} iterations.'.format(runtime, t))
        plt.plot(range(len(self.loss_train)), self.loss_train, label="training loss", ls="dotted", lw=2, alpha=0.7)
        plt.plot(range(len(self.loss_test)), self.loss_test, label="test loss", lw=2.5, alpha=0.5)
        plt.xlabel("Number of iterations", fontweight='bold')
        plt.ylabel("Mean Squared Error (MSE)" , fontweight='bold')
        plt.legend()
        plt.savefig('MSE.png', dpi=600)
        plt.show()

        # 3d-scatterplot of the true and the predicted target values
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(xtest[0,:], xtest[1,:], ytest, s=0.05, alpha=1, c="brown", label='ytest') 
        ax.scatter(xtest[0,], xtest[1,], f, s=0.05, alpha=1, c="g", label='Predicted values') 
        plt.xlabel("feature 1", fontweight='bold')
        plt.ylabel("feature 2" , fontweight='bold')
        ax.set_zlabel("regression value" , fontweight='bold')
        lgnd = plt.legend(prop={'size': 12})
        lgnd.legendHandles[0]._sizes = [30]
        lgnd.legendHandles[1]._sizes = [30]
        plt.savefig('Plot.png', dpi=600)
        
        return self                                 
    
    def predict(self, X, activation1=Relu, activation2=Relu):
        """
        Returns the target values for the given test set. 
        """
        
        f, *z = self.forward_propagation(X, self.W, self.b, activation1, activation2)
        return f
# %%
