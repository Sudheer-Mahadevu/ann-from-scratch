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

    def __init__(self, input_size, output_size , weight_init, activation_name,
                 verbose = False):

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

        if weight_init == 'xavier':
            std = np.sqrt(2/(input_size + output_size))
            self.W = np.random.normal(0,std, (input_size,output_size))
        else:
            self.W = np.random.rand(input_size,output_size)*0.01
        
        self.b = np.zeros((1,output_size))
        self.grad_W = None 
        self.grad_b = None

        self.Z = None
        self.A = None
        self.activation_fn, self.activation_der = ACTIVATION_MAP[activation_name]

        # cache for backpropagation
        self.A_prev = None
        self.Z_prev = None

        self.verbose = verbose
        if self.verbose:
            print(f"""Neural Layer created:
                  number of neurons: {output_size},
                  weight init: {weight_init},
                  activation: {activation_name},
                  """)
            
        ###### DOUBT: for evaluation should they be list of None or only None?
        ###### TODO: For ReLU, HE initialization is recommended.

    def forward(self, Z_prev ,A_prev):
        """
        Forward Propagation: Takes the activations from previous layer and
        updates the activations of this layer
        """

        self.Z = A_prev @ self.W + self.b
        self.A =  self.activation_fn(self.Z)

        self.Z_prev = Z_prev; self.A_prev = A_prev

        return self.Z, self.A
    
    def backward(self,dZ):
        """
        Backward Propagation: Implements the backpropagation equations
        Takes A_(l-1), Z_(l-1), delta_(l) (called as dZ) from previous layer
        and computes dW and db. Also computes delta_(l-1) to be fed to the
        previous layer in back propagation
        """

        n = self.A_prev.shape[0] # samples per batch

        # Note that we are maintaining delta_l for each of the input in the batch
        # and summing it up when computing dW, db (This is the mini-batch GD!)
        self.grad_W = (self.A_prev.T @ dZ)/n
        self.grad_b = np.sum(dZ, axis = 0, keepdims=True)/n

        dZ_prev = (dZ @ self.W.T) * self.activation_der(self.Z_prev)

        return dZ_prev

    def zero_grad(self):
        """
        Sets gradients to None
        """

        """
        This seems to be redundant as we are directly replacing grad_W, grad_b
        with new values every mini-batch/ backprop iteration. But this will be
        a useful method when gradient accumulation is used and it is recommened
        to put it.
        """

        self.grad_W = None
        self.grad_b = None