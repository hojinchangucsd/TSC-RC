import numpy as np
import math as math

from load import load_data

# X=[N,T,V] Y=[N], Z is for averaging across time
def prep_data(X,Y,Z=8):
    # shuffle X and Y
    p = np.random.permutation(len(Y))
    X = X[p]
    Y = Y[p]

    # one hot encode Y
    Y = np.eye(2)[Y]

    # Average X across dim T per Z timepoints
    N = len(X)
    averaged_X = np.empty((X.shape[0], math.ceil(X.shape[1] // Z), X.shape[2]))
    for i in range(0, averaged_X.shape[1]): 
        averaged_X[:,i,:] = np.mean(X[:,i*Z:(i+1)*Z,:], keepdims=True, axis=1).squeeze()

    # Z-score normalization
    X_std = np.std(X, axis=1, keepdims=True)
    X_mean = np.mean(X, axis=1, keepdims=True)
    normalized_X = (averaged_X - X_mean) / X_std

    return normalized_X, Y

# Return: X_train, X_test, Y_train, Y_test
def split_data(X,Y,train_split=0.8): 
    split_index = int(X.shape[0]*train_split)
    return X[:split_index,:,:], X[split_index:,:,:], Y[:split_index], Y[split_index:]