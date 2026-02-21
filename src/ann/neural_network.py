"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

from .neural_layer import NeuaralLayer
from .activations import softmax
from.objective_functions import LOSS_FUNCTIONS

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, hidden_size, weight_init, activation_name, 
                 loss_func_name, verbose = False):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        self.layers = []

        # Assume input has dimension 784 (28*28) and output has dimension 10
        self.input_size = 2  # d
        self.output_size = 1  # k

        self.loss_func, self.loss_delta = LOSS_FUNCTIONS[loss_func_name]

        prev_dim = self.input_size
        for h_size in hidden_size:
            self.layers.append(
            NeuaralLayer(prev_dim,h_size,weight_init,activation_name, 
                         verbose=verbose))
            prev_dim = h_size

        self.layers.append(NeuaralLayer(prev_dim,self.output_size, weight_init,
                                    activation_name='identity', verbose=verbose))
        
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """

        if X.shape[1] != self.input_size:
            raise ValueError(f"""Neural Network expects input dimension 
                             {self.input_size} got {X.shape[1]}""")
        
        A_prev = X; Z_prev = X
        for layer in self.layers:
            Z_curr , A_curr = layer.forward(Z_prev, A_prev)
            Z_prev = Z_curr; A_prev = A_curr
        
        logits = Z_prev
        
        return logits


    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """

        dZ = self.loss_delta(y_true, y_pred)
        for layer in self.layers:
            dZ = layer.backward(dZ)

        #### DOUBT: should it return grads or neural layer is fine?
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        pass
    
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        pass
    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        pass
