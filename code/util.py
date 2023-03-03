import numpy as np
import math as math
from sklearn.metrics import confusion_matrix

from load import load_data

# X=[N,T,V] Y=[N], Z is for averaging across time
def prep_data(X,Y,Z=8):

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

def shuffle_data(X,Y): 
    # shuffle X and Y
    p = np.random.permutation(len(Y))
    X = X[p]
    Y = Y[p]
    return X, Y

def generate_k_fold_set(dataset,k=5): 
    X,Y = dataset

    if k == 1:
        yield (X, Y), (X[len(X):], Y[len(Y):])
        return

    X,Y = shuffle_data(X,Y)
    
    fold_width = len(X) // k

    l_idx, r_idx = 0, fold_width
    for i in range(k):
        Xtrain = np.concatenate([X[:l_idx], X[r_idx:]])
        Ytrain = np.concatenate([Y[:l_idx], Y[r_idx:]])
        train = shuffle_data(Xtrain,Ytrain)

        Xval = X[l_idx:r_idx]
        Yval = Y[l_idx:r_idx]
        val = shuffle_data(Xval,Yval)

        yield train, val

        l_idx, r_idx = r_idx, r_idx + fold_width

def cm_values(pred_class, Yte): 
    true_class = np.argmax(Yte, axis=1)
    return confusion_matrix(true_class, pred_class).ravel()

# Return: X_train, X_test, Y_train, Y_test
def split_data(X,Y,train_split=0.8): 
    split_index = int(X.shape[0]*train_split)
    return X[:split_index,:,:], X[split_index:,:,:], Y[:split_index], Y[split_index:]

# Compute various metrics based solely on one model's predictions
def compute_test_scores(pred_class, Yte):
    """
    Wrapper to compute classification accuracy and F1 score
    """
    
    true_class = np.argmax(Yte, axis=1)
    
    accuracy = accuracy_score(true_class, pred_class)
    mcc = matthews_corrcoef(true_class, pred_class)
    tn, fp, fn, tp = confusion_matrix(true_class, pred_class).ravel()
    if Yte.shape[1] > 2:
        f1 = f1_score(true_class, pred_class, average='weighted')
    else:
        f1 = f1_score(true_class, pred_class, average='binary')
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return accuracy, f1, mcc, sensitivity, specificity

# Compute metrics using confusion matrix values
def compute_metrics_cm(values): 
    tn, fp, fn, tp = values
    acc = (tp+tn)/(tp+tn+fp+fn)
    mcc = (tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    f1 = 2*tp/(2*tp+fp+fn)
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    metrics = acc, f1, mcc, sens, spec
    return metrics