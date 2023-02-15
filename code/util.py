import numpy as np

from load import load_data

# X=[N,T,V] Y=[N], Z is for averaging across time
def prep_data(X,Y,Z=8):
    T = 1 # time axis

    # shuffle X and Y
    p = np.random.permutation(len(Y))
    X = X[p]
    Y = Y[p]

    # one hot encode Y
    Y = np.eye(2)[Y]

    # Average X across dim T per Z timepoints
    N = len(X)
    averaged_X = np.empty((X.shape[0], X.shape[1] // 3, X.shape[2]))
    for i in range(0, averaged_X.shape[1]): 
        averaged_X[:,i,:] = np.mean(X[:,i:i+Z,:], keepdims=True)

    print("X shape: ", X.shape)
    print("aX shape: ", averaged_X.shape)

    X_std = np.std(X, axis=T, keepdims=True)
    #X_mean = ?
    X_scaler = StandardScaler()
    normalized_X = X_scaler.fit_transform(averaged_X)

    return normalized_X, Y

# Return: X_train, X_test, Y_train, Y_test
def split_data(X,Y,train_split=0.8): 
    split_index = int(N*train_split)
    return X[:split_index,:,:], X[split_index:,:,:], Y[:split_index], Y[split_index:]