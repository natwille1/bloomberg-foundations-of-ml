import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    # TODO
    # check if values are constant in the array
    np.all(X_train == X_train[0, :], axis=0)

    train_max, train_min = (np.max(train), np.min(train))
    train_normalised = (train - train_min) / (train_max - train_min)
    test_normalised = (test - train_min) / (train_max - train_min)
    return train_normalised, test_normalised


df = pd.read_csv('data.csv', delimiter=',')
print(df.head(4))
cols = df.shape[1]
X = df.iloc[:, 0:cols-1]
y = df.iloc[:, 48]
print(X.head(2))
print(y.head(2))


#%%
## Data analysis

plt.plot(X)
plt.show()

#%%
## Identify features of constant value

it = 0
it2 = 1

const_features = []
features = X.columns.tolist()

for feat in range(len(features)):
    print("Feature: ", feat)
    if len(X.iloc[:,feat].value_counts()) == 1:
        print("constant feature: ", feat)
    else:
        print(len(X.iloc[:,feat].value_counts()))
        print("not constant feature")


#%%
X = df.values[:,:-1]
y = df.values[:,-1]
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 100, random_state=32)

print(X_train.shape, X_test.shape)

checks = np.all(X_train == X_train[0, :], axis=0)
print(checks)


X_train_scaled, X_test_scaled = feature_normalization(X_train, X_test)

print("train max: {}, train min: {}, test max: {}, test min: {}".format(X_train_scaled.max(), X_train_scaled.min(), X_test_scaled.max(), X_test_scaled.min()))

plt.hist(X_train)
plt.show()
plt.hist(X_train_scaled)
plt.show()


#%%
