"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np
from functools import partial

class Optimizer:
    def __init__(self, optimizer_name ,layers_list, lr=None, 
                 gamma = 0.9, beta=0.9, beta1 = 0.9, beta2 = 0.999):

        self.layers = layers_list
        self.lr = lr
        self.g = gamma
        self.b = beta
        self.b1 = beta1; self.b2 = beta2
        

        # momentum
        self.v_W = [np.zeros_like(l.W) for l in layers_list]
        self.v_b = [np.zeros_like(l.b) for l in layers_list]

        # scaling
        self.s_W = [np.zeros_like(l.W) for l in layers_list]
        self.s_b = [np.zeros_like(l.b) for l in layers_list]

        self.t = 0 # time or number of mini-batches
        self.adam = partial(self.dam, 'adam')
        self.nadam = partial(self.dam, 'nadam')

        OPTIMIZERS = {
            'sgd': self.sgd,
            'momentum': self.momentum,
            'nag' : self.nag,
            'rmsprop' : self.rmsprop,
            'adam' : self.adam,
            'nadam': self.nadam,
        }
        self.optimiser = OPTIMIZERS[optimizer_name]


    def step(self):
        self.optimiser()

    def sgd(self):

        for layer in self.layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


    def momentum(self):

        for i,layer in enumerate(self.layers):
            # update momentum for each layer
            self.v_W[i] = self.g * self.v_W[i] + self.lr * layer.grad_W
            self.v_b[i] = self.g * self.v_b[i] + self.lr * layer.grad_b

            # update params
            layer.W -= self.v_W[i]
            layer.b -= self.v_b[i]


    def nag(self):
        
        for i,layer in enumerate(self.layers):
            # update momentum for each layer
            self.v_W[i] = self.g * self.v_W[i] + self.lr * layer.grad_W
            self.v_b[i] = self.g * self.v_b[i] + self.lr * layer.grad_b

            # update params
            layer.W -= self.g * self.v_W[i] + self.lr * layer.grad_W
            layer.b -= self.g * self.v_b[i] + self.lr * layer.grad_b
    

    def rmsprop(self):

        epsilon = 1e-8
        for i,layer in enumerate(self.layers):
            # update scaling for each layer
            self.s_W[i] = self.b * self.s_W[i] + (1-self.b) * (layer.grad_W**2)
            self.s_b[i] = self.b * self.s_b[i] + (1-self.b) * (layer.grad_b**2)
        
            # update params
            layer.W -= (self.lr/np.sqrt(self.s_W[i]+ epsilon)) * layer.grad_W
            layer.b -= (self.lr/np.sqrt(self.s_b[i]+ epsilon)) * layer.grad_b
    

    def dam(self, subtype):
        self.t += 1
        epsilon = 1e-8
        for i,layer in enumerate(self.layers):
            # update momentum
            self.v_W[i] = self.b1 * self.v_W[i] + (1-self.b1) * layer.grad_W
            self.v_b[i] = self.b1 * self.v_b[i] + (1-self.b1) * layer.grad_b

            # update scaling
            self.s_W[i] = self.b2 * self.s_W[i] + (1-self.b2) * (layer.grad_W**2)
            self.s_b[i] = self.b2 * self.s_b[i] + (1-self.b2) * (layer.grad_b**2)

            # correct bias
            if subtype == 'adam':
                v_W_corr = self.v_W[i]/(1-self.b1**self.t)
                v_b_corr = self.v_b[i]/(1-self.b1**self.t)
            elif subtype =='nadam':
                v_W_corr = self.v_W[i]*(self.b1/(1-self.b1**(self.t+1))) + \
                           layer.grad_W *((1-self.b1)/(1-self.b1**self.t))
                
                v_b_corr = self.v_b[i]*(self.b1/(1-self.b1**(self.t+1))) + \
                           layer.grad_b *((1-self.b1)/(1-self.b1**self.t))
            
            s_W_corr = self.s_W[i]/(1-self.b2**self.t)
            s_b_corr = self.s_b[i]/(1-self.b2**self.t)

            # update parameters
            layer.W -= (self.lr/np.sqrt(s_W_corr+epsilon))*v_W_corr
            layer.b -= (self.lr/np.sqrt(s_b_corr+epsilon))*v_b_corr