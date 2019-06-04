import numpy as np

def cross_entropy(AL, Y):
        m = Y.shape[1]
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        cost = np.squeeze(cost)
        return cost
    
def cross_entropy_prime(AL, Y):
    return - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))