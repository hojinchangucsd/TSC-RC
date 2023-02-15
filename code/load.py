import glob
import ntpath
import random
import pandas as pd
import numpy as np

# load all data and return as a 3D numpy array
# stack data so that the samples are in the 1st-, the time in the 2nd-, and the features (ch0 and ch1) in the 3rd dimension
def load_data(prefix):
    #global filelist
    loaded = list()
    led_data = list()
    #load cancers
    filenames = sorted( glob.glob(prefix + 'cancers\\' + "*.csv") )    #filenames are sorted by name
    Ncancer = len(filenames)
    for name in filenames:
        data, led = load_file(name)
        loaded.append(data)
        led_data.append(led)
        #filelist.append(name)   #store to global file list
    #load benigns
    filenames = sorted( glob.glob(prefix + 'benigns\\' + "*.csv") )
    Nbenigns = len(filenames)
    for name in filenames:
        data, led = load_file(name)
        loaded.append(data)
        led_data.append(led)
        #filelist.append(name)   #store to global file list
    #load normals
    filenames = sorted( glob.glob(prefix + 'normalskin\\' + "*.csv") )
    Nnormals = len(filenames)
    for name in filenames:
        data, led = load_file(name)
        loaded.append(data)
        led_data.append(led)
        #filelist.append(name)   #store to global file list
    loaded = np.stack(loaded)
    led_cur = np.stack(led_data)
    #Setup class Array
    y = np.zeros(Ncancer + Nbenigns + Nnormals, dtype=np.int8)
    y[0:Ncancer] = 1
    #y[Ncancer:Ncancer + Nbenigns + Nnormals] = 1  #use this if representing cancer as class 0
    #y = to_categorical(y) #needed only if softmax is used for output layer: i.e. more than 2 classes

    return loaded, y #, led_cur, Ncancer, Nbenigns, Nnormals

# loads a single TruScore xxxx.csv file as a numpy array, 
#columns 1 and 2 coresponding to analog channels ch0 and ch1
def load_file(filepath):
    myfile = open(filepath, "r")
    line = myfile.readline()
    led_cur = float( line.split()[2] )   # reads led current form the line "LED current: xxx"
    line = myfile.readline() 
    T = float( line.split()[1] ) # reads Temperature form the line "Temperature: xxx"
    myfile.close()    
    dataframe = pd.read_csv(filepath, header=None, skiprows=2, usecols=[1,2])  #reads signal values from each channel
    return dataframe.values, led_cur

# Original Fillipo code for loading dataset
'''
data = scipy.io.loadmat('../dataset/'+config['dataset_name']+'.mat')
Xtr = data['X']  # shape is [N,T,V]
if len(Xtr.shape) < 3:
    Xtr = np.atleast_3d(Xtr)
Ytr = data['Y']  # shape is [N,1]
Xte = data['Xte']
if len(Xte.shape) < 3:
    Xte = np.atleast_3d(Xte)
Yte = data['Yte']

print('Loaded '+config['dataset_name']+' - Tr: '+ str(Xtr.shape)+', Te: '+str(Xte.shape))

# One-hot encoding for labels
onehot_encoder = OneHotEncoder(sparse=False)
Ytr = onehot_encoder.fit_transform(Ytr)
Yte = onehot_encoder.transform(Yte)
'''