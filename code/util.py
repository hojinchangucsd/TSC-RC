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
    aX = np.empty((X.shape[0], math.ceil(X.shape[1] // Z), X.shape[2]))
    for i in range(0, aX.shape[1]): 
        aX[:,i,:] = np.mean(X[:,i*Z:(i+1)*Z,:], keepdims=True, axis=1).squeeze()

    '''
    # Z-score normalization
    aX_std = np.std(aX, axis=1, keepdims=True)
    aX_mean = np.mean(aX, axis=1, keepdims=True)
    nX = (aX - aX_mean) / aX_std
    '''
    '''
    # Min-max normalization
    aX_min = aX.min(axis=1, keepdims=True)
    aX_max = aX.max(axis=1, keepdims=True)
    nX = (aX - aX_min) / (aX_max - aX_min)
    '''

    return aX, Y

# Return: X_train, X_test, Y_train, Y_test
def split_data(X,Y,train_split=0.8): 
    split_index = int(X.shape[0]*train_split)
    return X[:split_index,:,:], X[split_index:,:,:], Y[:split_index], Y[split_index:]