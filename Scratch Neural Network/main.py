import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("Scratch Neural Network/data/test.csv")
data = np.array(df)
m, n = (
    data.shape
)  # m = number of examples we have | n = features +1 w;hcih is the "index" because.csv, but ever feature is how activated a pixel is
np.random.shuffle(data)

# the data we will test via cross validation to avoid over fitting
# they are transposed to make matrix transformations easier, so now every column is an example
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[:n]

# data we will train and get our weights from
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[:n]

print(X_train[:, 0].shape)


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 784) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(0, Z)


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(A2)
    return Z1, A1, Z2, A2


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1


def back_prop(Z1, A1, Z2, A2, W2, Y):
    pass
