# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from scipy import linalg

class LinearRegression:
    def __init__(self) :
        self.w_ = None
    
    def fit(self, X, y):
        Xtil = np.c_[np.ones((X.shape[0])),X] #切片項追加
        A = np.dot(Xtil.T,Xtil)
        b = np.dot(Xtil.T, y)
        self.w_ = linalg.solve(A,b)
    
    def predict(self,X):
        if X.ndim == 1 :
            X = X.reshape(1,-1)  #1行○列
        Xtill = np.c_[np.ones(X.shape[0],X)]
        return np.dot(Xtill,self.w_)   


