import os.path
import numpy as np

def build_path(path):
    for i in path:
        if not os.path.exists(i):
            os.makedirs(i)
    
def calc_normalized(X:np.ndarray, axis:int, savefile:str):
    Xmean = np.fromfile(savefile+'mean.bin', dtype=np.float32)
    Xstd = np.fromfile(savefile+'std.bin', dtype=np.float32)
    return (X-Xmean) / Xstd

def Normalize(X:np.ndarray, axis:int, savefile:str=None):
    Xmean, Xstd = X.mean(axis=axis), X.std(axis=axis)
    for i in range(Xstd.size):
        if (Xstd[i]==0):
            Xstd[i]=1
    X = (X - Xmean) / Xstd
    if savefile != None:
        Xmean.tofile(savefile+'mean.bin')
        Xstd.tofile(savefile+'std.bin')
    return X, Xmean, Xstd
