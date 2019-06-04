import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    f = sigmoid(x)
    return f*(1-f)

    
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.square(np.tanh(x))


def relu(x):
    return np.maximum(x, 0.0)

def relu_prime(x):
    return np.where(x>0, 1.0, 0.0)


def softmax(x):
    f = np.exp(x - np.max(x))
    return f / f.sum(axis=0)
    