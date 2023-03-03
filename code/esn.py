# General imports
import numpy as np
import scipy.io
import time

# Custom imports
from modules import RC_model
from util import *

# ============ RC model configuration and hyperparameter values ============
config = {}
config['dataset_name'] = 'veriskin'

config['seed'] = int(time.time())
np.random.seed(config['seed'])

# Hyperarameters of the reservoir
config['n_internal_units'] = 450        # size of the reservoir
config['spectral_radius'] = 0.59        # largest eigenvalue of the reservoir
config['leak'] = None                   # amount of leakage in the reservoir state update (None or 1.0 --> no leakage)
config['connectivity'] = 0.01           # percentage of nonzero connections in the reservoir
config['input_scaling'] = 0.5           # scaling of the input weights
config['noise_level'] = 0.01            # noise in the reservoir state update
config['n_drop'] = 5                    # transient states to be dropped
config['bidir'] = False                 # if True, use bidirectional reservoir
config['circ'] = False                  # use reservoir with circle topology

# Dimensionality reduction hyperparameters
config['dimred_method'] ='pca'       # options: {None (no dimensionality reduction), 'pca', 'tenpca'}
config['n_dim'] = 40                    # number of resulting dimensions after the dimensionality reduction procedure

# Type of MTS representation
config['mts_rep'] = 'mean'              # MTS representation:  {'last', 'mean', 'output', 'reservoir'}
config['w_ridge_embedding'] = 10.0      # regularization parameter of the ridge regression

# Type of readout
config['readout_type'] = 'lin'          # readout used for classification: {'lin', 'mlp', 'svm'}

# Linear readout hyperparameters
config['w_ridge'] = 10.0                # regularization of the ridge regression readout

# SVM readout hyperparameters
config['svm_gamma'] = 0.005             # bandwith of the RBF kernel
config['svm_C'] = 5.0                   # regularization for SVM hyperplane

# MLP readout hyperparameters
config['mlp_layout'] = (100,50,10)      # neurons in each MLP layer
config['num_epochs'] = 2000             # number of epochs 
config['w_l2'] = 0.001                  # weight of the L2 regularization
config['nonlinearity'] = 'relu'         # type of activation function {'relu', 'tanh', 'logistic', 'identity'}

print(config)

# ============ k-Fold Cross-validation ============
folds = 7
print("Number of folds: %d" % (folds))
print("Validation set percentage: %.2f%%" % (100/folds))

# ============ Load dataset ============
print("Loading data...")
X, Y = load_data('C:\\Users\\Hojin\\Documents\\Career\\Veriskin\\dataset\\')
print("Preparing data...")
X, Y = prep_data(X, Y)

# ============ Initialize, train and evaluate the RC model ============
fold = 0 # current fold
TN, FP, FN, TP = 0, 0, 0, 0
for train, val in generate_k_fold_set((X,Y), k=folds): 
    fold = fold + 1
    print("Generating fold %d..."%(fold))
    Xtrain, Ytrain = train
    Xval, Yval = val
    print("Creating model...")
    classifier = RC_model(  reservoir=None,     
                            n_internal_units=config['n_internal_units'],
                            spectral_radius=config['spectral_radius'],
                            leak=config['leak'],
                            connectivity=config['connectivity'],
                            input_scaling=config['input_scaling'],
                            noise_level=config['noise_level'],
                            circle=config['circ'],
                            n_drop=config['n_drop'],
                            bidir=config['bidir'],
                            dimred_method=config['dimred_method'], 
                            n_dim=config['n_dim'],
                            mts_rep=config['mts_rep'],
                            w_ridge_embedding=config['w_ridge_embedding'],
                            readout_type=config['readout_type'],            
                            w_ridge=config['w_ridge'],              
                            mlp_layout=config['mlp_layout'],
                            num_epochs=config['num_epochs'],
                            w_l2=config['w_l2'],
                            nonlinearity=config['nonlinearity'], 
                            svm_gamma=config['svm_gamma'],
                            svm_C=config['svm_C'])
    print("Training...")
    tr_time = classifier.train(Xtrain, Ytrain)
    print('Training time = %.2f seconds'%tr_time)
    print("Testing...")
    Ypred = classifier.test(Xval, Yval)
    tn, fp, fn, tp = cm_values(Ypred, Yval)
    TN, FP, FN, TP = TN+tn, FP+fp, FN+fn, TP+tn
    print('Accumulated confusion matrix values: ')
    print('TN: %d  FP: %d  FN: %d  TP: %d' % (TN,FP,FN,TP))
CM_vals = (TN, FP, FN, TP)
acc, f1, mcc, sens, spec = compute_metrics_cm(CM_vals)
print('=== %d fold cross validation metrics ==='%(folds))
print('Accuracy: %.2f'%(acc))
print('F1: %.2f'%(f1))
print('MCC: %.2f'%(mcc))
print('Sensitivity: %.2f'%(sens))
print('Specificity: %.2f'%(spec))