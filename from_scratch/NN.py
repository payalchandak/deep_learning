import numpy as np
import scipy.special 



class NN_Classifier:

    
    sigmoid = lambda self, x: scipy.special.expit(x)
    softmax = lambda self, x: np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum(axis=0)
    
    
    def __init__(self, layers, learningRate):
        if(len(layers)<3): raise ValueError('Enter nodes as: [input, hidden_1, ... hidden_n, output]')    
                    
        self.lr = learningRate
        self.size = len(layers); self.out = self.size-1
        
        self.nodes = {}; self.weights = {}; self.err = {}
        
        for i in range(0, self.size):    
            self.nodes[i] = np.zeros((layers[i],1))               # columns of nodes init to 0
            if(i>0):
                self.err[i] = np.zeros((layers[i],1))             # errors init to 0
            if(i+1<self.size): 
                self.weights[(i, i+1)] = np.random.rand(layers[i+1],layers[i])-0.5      # wt(i,j) from layer i to j of shape (j, i)
                                                                                        # init bet -0.5 & 0.5
    
    
    # puts MSE in last error layer                
    def set_loss(self, targets):                                                        
        targets = np.array(targets, ndmin=2).reshape(self.nodes[self.out].shape)
        self.err[self.out] = np.square(np.abs(np.subtract(self.nodes[self.out], targets)))
    
    def get_loss(self, targets=None):
        if targets is not None:
            self.set_loss(targets)
        return sum(self.err[self.out])[0]

    
    
    def train(self, data, targets):
        if(len(self.nodes[0])!=len(data)): raise ValueError('Data doesn\'t match input layer')
        if(len(self.nodes[self.out])!=len(targets)): raise ValueError('Targets don\'t match output layer')
        
        # forward pass
        self.nodes[0] = np.array(data, ndmin=2).reshape(self.nodes[0].shape)
        for i in range(self.out):
            self.nodes[i+1] = np.dot(self.weights[(i,i+1)],self.nodes[i])
            if(i+1<self.out):
                self.nodes[i+1] = self.sigmoid(self.nodes[i+1])
            else:
                self.nodes[i+1] = self.softmax(self.nodes[i+1])
        
        # output layer error
        self.set_loss(targets)
        
        # hidden layer error 
        for i in range(self.out-1,0,-1):
            self.err[i] = np.dot(self.weights[(i,i+1)].T, self.err[i+1])
            
        # back propagation 
        # gradient of edges(i,i+1) = -1 * error_i+1 * output_i+1 * (1 - output_i+1) * output_i                      
        for i in range(self.out):
            delta = -1 * np.dot( (self.err[i+1]*self.nodes[i+1]*(1-self.nodes[i+1])),  self.nodes[i].T)    
            self.weights[(i,i+1)] -= self.lr * delta
    
    
    
    def test(self, data):
        if(len(self.nodes[0])!=len(data)): raise ValueError('Data does not match input layer')
            
        # forward pass
        self.nodes[0] = np.array(data, ndmin=2).T
        for i in range(self.out):
            self.nodes[i+1] = np.dot(self.weights[(i,i+1)],self.nodes[i])
            if(i+1<self.out):
                self.nodes[i+1] = self.sigmoid(self.nodes[i+1])
            else:
                self.nodes[i+1] = self.softmax(self.nodes[i+1])
       
        return self.nodes[self.out]