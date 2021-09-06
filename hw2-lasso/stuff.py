import numpy as np
import matplotlib.pyplot as plt

def l1_loss(X, W, y):
    preds = X.T@W
    loss = ((preds-y)**2)/X.shape[0] + np.sum(abs(W),axis=1)
    return loss

def l1_prime(X, W, y):
    

w = np.arange(-10,10)
print(w)

np.sum(abs(w))
