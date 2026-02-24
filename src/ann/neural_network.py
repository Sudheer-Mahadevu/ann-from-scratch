"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

from .neural_layer import NeuaralLayer
from .activations import softmax
from .objective_functions import LOSS_FUNCTIONS
from .optimizers import Optimizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from functools import partial
import numpy as np
import time

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
        self.output_size = 2  # k

        prev_dim = self.input_size
        for h_size in hidden_size:
            self.layers.append(
            NeuaralLayer(prev_dim,h_size,weight_init,activation_name, 
                         verbose=verbose))
            prev_dim = h_size

        self.layers.append(NeuaralLayer(prev_dim,self.output_size, weight_init,
                                    activation_name='identity', verbose=verbose))
        
        self.loss_func, self.loss_delta = LOSS_FUNCTIONS[loss_func_name]

        self.recorder = {}

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
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)

        #### DOUBT: should it return grads or neural layer is fine?
        
    
    def train_minibatch(self, X, y, optimizer):

        logits = self.forward(X)
        y_pred = softmax(logits)
        loss = self.loss_func(y,y_pred)
        self.backward(y, y_pred)
        optimizer.step()
        self.zero_grad()

        return loss


    def train_epoch(self, X, y, X_valid, y_valid, bs, optimizer, metric_names):
        n = X.shape[0]
        train_loss = 0
        for b in range(0,n,bs):
            batch_loss = self.train_minibatch(X[b:b+bs], y[b:b+bs], optimizer)
            train_loss += y[b:b+bs].shape[0] * batch_loss
            
            self.recorder['raw_loss'].append(y[b:b+bs].shape[0] * batch_loss/bs)
        
        # TODO: If possible try to validate in batches
        train_loss /= n
        val_metrics = self.evaluate(X_valid, y_valid, metric_names)

        self.recorder['train_loss'].append[train_loss]
        for name in val_metrics:
            self.recorder[name].append[val_metrics[name]]

        return


    def train(self,X_train, y_train, X_valid, y_valid, batch_size, 
              optimizer, epochs, learning_rate, metric_names):
        """
        Train the network for specified epochs.
        """
        # better to create a dataloader that shuffles data, splits into
        #  train, valid sets

        optimizer.layers = self.layers,
        optimizer.lr = learning_rate
        
        self.init_recorder(metric_names)
        for e in range(epochs):
            if(e%5 == 0):
                print(f"Training Epoch {e} ...")
                
            start_time = time.perf_counter()
            self.train_epoch(X_train, y_train,X_valid,y_valid,
                             batch_size, optimizer, metric_names)
            end_time = time.perf_counter()
            duration = end_time-start_time
            self.recorder['time'] = round(duration,2)
        
        self.pretty_print_train()
        return self.recorder
    
    
    def evaluate(self, X, y, metrics = ['accuracy','f1_macro']):
        """
        Evaluate the network on given data.
        """

        logits = self.forward(X)
        y_pred = softmax(logits)
        valid_loss = self.loss_func(y,y_pred)

        y_pred_cls = np.argmax(y_pred, axis=1)
        y_true_cls = np.argmax(y,axis=1)

        METRICS_MAP = {
            'accuracy': accuracy_score,
            'f1_macro' : partial(f1_score,average='macro'),
            'precision': precision_score,
            'recall' : recall_score
        }

        metrics = {}
        metrics['valid_loss'] = valid_loss

        for m_name in metrics:
            metrics[m_name] = METRICS_MAP[m_name](y_true_cls,y_pred_cls)

        return metrics
    
    
    def zero_grad(self):

        for layer in self.layers:
            layer.zero_grad()
    

    def init_recorder(self, metric_names):

        self.recorder['train_loss'] = []
        self.recorder['valid_loss'] = []
        self.recorder['time'] = []
        self.recorder['raw_loss'] = []

        for name in metric_names:
            self.recorder[name] = []
    

    def pretty_print_train(self):
        
        print("\n" + "="*70)
        print("TRAINIG LOG")
        print("="*70)
        
        fields = ["train_loss", "valid_loss", "accuracy", "f1_macro", "time"]
        print(" Epochs | T-Loss | V-Loss |  Accu  |F1-Score| Time")
        epochs = len(self.recorder['train_loss'])

        for e in range(epochs):
            print(f"{e}"+" "*7, end="|")
            for f in fields:
                print(f"{self.recorder[f]:.4f}  ", end="|")
            print("\n")