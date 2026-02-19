"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from .activations import ACTIVATION_MAP

class NeuaralLayer:
    """
        Neural layer class that has forward and back propagation functionality
        Implements the basic forward and backward propagation equations taught 
        in class 'adjusted for batch processing'.
        """

    def __init__(self, batch_size, input_size, output_size , weight_init, activation_name):

        self.input_size = input_size
        self.output_size = output_size

        """
        Note: In slides, the shape of W is (o,i), but here it is (i,o)
        It is done to write the code elegantly for a batch of x's (i.e X)
        dimension of X : (n,d) (n is batch size)
        dimension of W : (d,h) (d is input, h is output (the current layer size))
        dimension of b : (1,h)
        With this we can write the book equation : z = Wx+b for multiple x as:
        Z = XW + b or WA+b , where each row of Z (n,h) contains one z. 
        b broadcasts to multiple rows here.

        This is called "batch-first" format as n is the starting dimension
        """

        self.Z = np.zeros((batch_size,output_size))
        self.A = np.zeros((batch_size,output_size))
        self.activation_fn, self.activation_der = ACTIVATION_MAP[activation_name]

        if weight_init == 'xavier':
            std = np.sqrt(2/(input_size + output_size))
            self.W = np.random.normal(0,std, (input_size,output_size))
        else:
            self.W = np.random.rand(input_size,output_size)*0.01
        
        self.b = np.zeros((1,output_size))
        self.grad_W = None 
        self.grad_b = None

        self.dZ_prev = None
        ###### DOUBT: for evaluation should they be list of None or only None?
        ###### CAREFUL: ensure that last batch_size is not given are bs param

    def forward(self, A_prev):
        """
        Forward Propagation: Takes the activations from previous layer and
        updates the activations of this layer
        """

        self.Z = A_prev @ self.W + self.b
        self.A =  self.activation_fn(self.Z)

        return self.A
    
    def backward(self,A_prev,Z_prev,dZ):
        """
        Backward Propagation: Implements the backpropagation equations
        Takes A_(l-1), Z_(l-1), delta_(l) (called as dZ) from previous layer
        and computes dW and db. Also computes delta_(l-1) to be fed to the
        previous layer in back propagation
        """

        n = A_prev.shape[0] # samples per batch

        # Note that we are maintaining delta_l for each of the input in the batch
        # and summing it up when computing dW, db (This is the mini-batch GD!)
        self.grad_W = (A_prev.T @ dZ)/n
        self.grad_b = np.sum(dZ, axis = 0, keepdims=True)/n

        self.dZ_next = (dZ @ self.W.T) * self.activation_der(Z_prev)

        return self.grad_W, self.grad_b
