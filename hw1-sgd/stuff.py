import numpy as np
import pandas as pd


df = pd.read_csv('data.csv', delimiter=',')
X = df.values[:,:-1]
y = df.values[:,-1]
print(X.shape)
min, max = np.min(X,axis=0), np.max(X, axis=0)
print(max[:5])
print(min[:5])
diff = max - min
diff = np.array([[0,1,2,0], [0,2,2,1]])

remcols = np.where(np.max(diff,axis=0)==np.min(diff,axis=0))
remcols
Xtemp = np.delete(X,remcols,axis=1)
Xtemp.shape

#%%
theta = np.random.randn(X.shape[1]+1)
intercept = np.ones((X.shape[0],1))
design = np.hstack((intercept,X))
print(theta.shape)
print(design.shape)
preds = design@theta
preds
loss = (1/X.shape[0])*np.sum(((preds-y)**2))
print(loss)
losses = (preds - y)**2
np.sum(losses) / X.shape[0]
