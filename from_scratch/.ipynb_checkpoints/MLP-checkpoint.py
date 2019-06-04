import numpy as np
from activations import *
from loss import *

class Perceptron:
    
    def __init__(self, layer, activation):
        if(len(layer)<3): raise ValueError('Not enough layers')
        if(len(activation)!=len(layer)-1): raise ValueError('Activations don\'t match layers')
        for a in activation: 
            if a not in ['sigmoid','relu','tanh']: raise ValueError('Activation {} not implemented'.format(a))
                    
        self.L = len(layer); self.cost = []
        
        self.W = {}; self.B = {}; self.g = {}
        for l in range(1, self.L):     
            self.W[l]= np.random.randn(layer[l],layer[l-1])*0.01
            self.B[l] = np.zeros((layer[l], 1))
            self.g[l] = activation[l-1]
    
    def model(self, X, Y, iterations, learningRate):
        if(X.shape[0]!=self.W[1].shape[1]): raise ValueError('X must be {} input nodes by m samples'.format(self.W[1].shape[1]))
        m = X.shape[1]
        
        for i in range(iterations):
            pred, A, Z = self.forward(X)
            dAL = self.compute_cost(pred, Y)
            dW, dB = self.backprop(m, A, Z, dAL)
            self.update(dW, dB, learningRate)
            
    def predict(self, X):
        pred, A, Z = self.forward(X)
        return np.where(pred>=0.5, 1, 0)
            
    def forward(self, X):
        Z = {0: X}; A = {0: X}
        for l in range(1,self.L):
            Z[l] = np.dot(self.W[l], A[l-1])+self.B[l]
            if (self.g[l] == 'relu'):
                A[l] = relu(Z[l])
            elif (self.g[l] == 'sigmoid'):
                A[l] = sigmoid(Z[l])
            elif (self.g[l] == 'tanh'):
                A[l] = tanh(Z[l])
        return A[self.L-1], A, Z
    
    def compute_cost(self, AL, Y):
        self.cost.append(cross_entropy(AL, Y))
        dAL = cross_entropy_prime(AL, Y)
        return dAL    
        
    def backprop(self, m, A, Z, dAL):
        dZ = {} ;  dW = {} ; dB = {}
        dA = {self.L-1: dAL}
        
        for l in range(self.L-1, 0, -1):

            # backwards through activation
            if self.g[l]=='relu':
                dZ[l] = dA[l] * relu_prime(Z[l])
            elif self.g[l]=='sigmoid':
                dZ[l] = dA[l] * sigmoid_prime(Z[l])
            elif self.g[l]=='tanh':
                dZ[l] = dA[l] * tanh_prime(Z[l])

            # backwards linearly & calculate deltas for parameters
            dW[l] = (1/m) * np.dot(dZ[l], A[l-1].T)
            dB[l] = (1/m) * np.sum(dZ[l], axis=1, keepdims=True)
            dA[l-1] = np.dot(self.W[l].T, dZ[l])
            
        return dW, dB
    
    def update(self, dW, dB, alpha):
        for l in range(1,self.L):
            self.W[l] -= alpha * dW[l]
            self.B[l] -= alpha * dB[l]
        
