import numpy as np

X = np.random.randint(low=0,high=4,size=(5,7,2))

Z = 3
N = len(X)
aX = np.empty((X.shape[0], (X.shape[1] // 3) + 1, X.shape[2]))
for i in range(0, aX.shape[1]): 
    aX[:,i,:] = np.mean(X[:,i*Z:(i+1)*Z,:], keepdims=True, axis=1).squeeze()

aX_std = np.std(aX, axis=1, keepdims=True)
aX_mean = np.mean(aX, axis=1, keepdims=True)
nX = (aX - aX_mean) / aX_std

for n in range(0,N):
    print(f'\nX[{n:d}]\n',X[n])
    print(f'\naX[{n:d}]\n',aX[n])
    print(f'\naX[{n:d}]\n',nX[n])
