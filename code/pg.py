import numpy as np

X = np.random.randint(low=0,high=4,size=(5,7,2))

Z = 3
N = len(X)
averaged_X = np.empty((X.shape[0], (X.shape[1] // 3) + 1, X.shape[2]))
for i in range(0, averaged_X.shape[1]): 
    averaged_X[:,i,:] = np.mean(X[:,i*Z:(i+1)*Z,:], keepdims=True, axis=1).squeeze()

for n in range(0,N):
    print(f'\nX[{n:d}]\n',X[n])
    print(f'\naX[{n:d}]\n',averaged_X[n])