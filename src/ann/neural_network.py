"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

from .neural_layer import NeuaralLayer
from .activations import softmax
from .objective_functions import LOSS_FUNCTIONS
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from functools import partial
import numpy as np
import time
from .optimizers import Optimizer


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args, verbose = False):
        """
        Initialize the neural network.
        With hidden and output layers and loss function
        """
        self.layers = []
        
        hidden_size = cli_args.hidden_size
        weight_init = cli_args.weight_init
        activation_name = cli_args.activation
        loss_func_name = cli_args.loss

        # Assume input has dimension 784 (28*28) and output has dimension 10
        self.input_size = 784  # d
        self.output_size = 10  # k

        prev_dim = self.input_size
        for h_size in hidden_size:
            self.layers.append(
            NeuaralLayer(prev_dim,h_size,weight_init,activation_name, 
                         verbose=verbose))
            prev_dim = h_size

        self.layers.append(NeuaralLayer(prev_dim,self.output_size, weight_init,
                                    activation_name='identity', verbose=verbose))
        
        self.loss_func, self.loss_delta = LOSS_FUNCTIONS[loss_func_name]
        self.optimizer = cli_args.optimizer
        self.lr = cli_args.learning_rate
        self.grad_W = [None]*(len(hidden_size)+1); self.grad_b = [None]*(len(hidden_size)+1)

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
            return (grad_w, grad_b) tuple for each of the layers
        """

        dZ = self.loss_delta(y_true, y_pred)
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)

        for i,layer in enumerate(reversed(self.layers)):
            self.grad_W[i] = layer.grad_W; self.grad_b[i] = layer.grad_b

        return (self.grad_W, self.grad_b)
    
    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d
    

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()


    def train(self, X_train, y_train, epochs=10, batch_size = 64):
        """This function is added only for evaluation, actual set of 
        functions are train_custom ->train_epoch -> train_minibatch. They offer
        tracking the metrics, evaluation on validation set, pretty print, etc.
        See main branch code to remove submission clutter.
        """
        n_samples = X_train.shape[0]
        if(len(y_train.shape) == 1):
            y_train = np.eye(10)[y_train] # convert to one-hot encoded vectors
        indices = np.arange(n_samples)
        optimizer = Optimizer(self.optimizer, self.layers)
        optimizer.lr = self.lr

        for e in range(epochs):
            np.random.shuffle(indices)
            for i in range(0,n_samples,batch_size):
                batch_indices = indices[i:i+batch_size]
                self.train_minibatch(X_train[batch_indices], 
                                     y_train[batch_indices], optimizer)


        return
    
    
    def train_minibatch(self, X, y, optimizer):
        """
        Trains mini-batch of X,y and return the train loss for the batch
        """
        logits = self.forward(X)
        y_pred = softmax(logits)
        loss = self.loss_func(y,y_pred)
        self.backward(y, y_pred)
        optimizer.step()
        self.zero_grad()

        return loss



    def train_epoch(self,dls,optimizer, metric_names):
        """
        Trains one epoch in mini-batches, validates and updates the training logs
        """
        
        # 1. Train
        train_loss = 0
        batches = dls.get_batches('train')
        for x,y in batches:
            batch_loss = self.train_minibatch(x, y, optimizer)
            train_loss += y.shape[0] * batch_loss
            
            # update mini-batch wise train loss
            self.recorder['raw_loss'].append(y.shape[0] * batch_loss/dls.bs)
        train_loss /= dls.x_train.shape[0]
        
        # 2. Validate
        val_metrics = self.evaluate_dls(dls, metric_names=metric_names)

        # 3. Update the training logs
        self.recorder['train_loss'].append(train_loss)
        for name in val_metrics:
            self.recorder[name].append(val_metrics[name])

        return



    def train_custom(self,dls, optimizer, epochs, learning_rate, 
              metric_names = ['accuracy','f1_macro']):
        """
        Train the network for specified epochs.

        Args:
        dls : MINSTLoader object
        optimizer: Optimizer object
        epochs: number of epochs
        metric_names: train, valid loss are always logged. Other metircs:
        'accuracy', 'f1_macro', 'precision', 'recall' 
        """

        # set optimizer configuration
        optimizer.layers = self.layers
        optimizer.lr = learning_rate
        
        # Train epochs
        self.init_recorder(metric_names)
        print("Training Epochs ... ", end="")
        for e in range(epochs):
            print(f"{e}", end=" ", flush=True)
                
            start_time = time.perf_counter()
            self.train_epoch(dls, optimizer, metric_names)
            end_time = time.perf_counter()
            duration = end_time-start_time
            self.recorder['time'].append(round(duration,2))
        
        self.pretty_print_train()

        return self.recorder


    
    def evaluate(self, X, y, metric_names=[]):
        """
        Evaluate the network on given data.

        Args:
        X : (n,d) shaped data matrix
        y : (n,k) shaped one-hot label matrix
        metric names: list of metrics. 'accuracy', 'f1_macro', 'precision', 'recall' .
        validation loss is always logged
        """

        if(len(y.shape) == 1):
            y = np.eye(10)[y] # convert to one-hot encoded vectors


        logits = self.forward(X)
        y_pred = softmax(logits)
        valid_loss = self.loss_func(y,y_pred)

        y_pred_cls = np.argmax(y_pred, axis=1)
        y_true_cls = np.argmax(y,axis=1)

        METRICS_MAP = {
            'accuracy': accuracy_score,
            'f1_macro' : partial(f1_score,average='macro'),
            'precision': partial(precision_score, average = 'macro'),
            'recall' : partial(recall_score, average = 'macro'),
        }

        metrics = {}
        metrics['valid_loss'] = valid_loss

        for m_name in metric_names:
            metrics[m_name] = METRICS_MAP[m_name](y_true_cls,y_pred_cls)

        return metrics
    
    def evaluate_dls(self,dls, metric_names, type='valid'):
        """
        Evaluates the model in batches, given dls.

        Args: 
        dls: MNISTLoader object
        metric_names: list of metrics. 'accuracy', 'f1_macro', 'precision',
        'recall' key-words. validation loss is always logged
        type: 'valid', 'test', 'train' key-words. The data in the dls to be evaluated
        """

        batches = dls.get_batches(type)

        preds = np.array([]); targs =np.array([])
        
        valid_loss = 0
        for x,y in batches:
            logits = self.forward(x)
            y_pred = softmax(logits)
            valid_loss += self.loss_func(y,y_pred)*y.shape[0]
            y_pred_cls = np.argmax(y_pred,axis=1)
            y_true_cls = np.argmax(y, axis=1)
            preds = np.concatenate((preds,y_pred_cls))
            targs = np.concatenate((targs,y_true_cls))
        
        valid_loss /= len(preds)

        METRICS_MAP = {
            'accuracy': accuracy_score,
            'f1_macro' : partial(f1_score,average='macro'),
            'precision': precision_score,
            'recall' : recall_score
        }

        metrics = {}
        metrics['valid_loss'] = valid_loss

        for m_name in metric_names:
            metrics[m_name] = METRICS_MAP[m_name](targs,preds)

        return metrics
    

    def zero_grad(self):
        """
        Sets the gradients of all layers to zero
        """
        for layer in self.layers:
            layer.zero_grad()
    


    def init_recorder(self, metric_names):
        """
        Initialize the train logs recorder
        """
        self.recorder['train_loss'] = []
        self.recorder['valid_loss'] = []
        self.recorder['time'] = []
        self.recorder['raw_loss'] = []

        for name in metric_names:
            self.recorder[name] = []
    
    """
    NOTE on recorder:
    The following metrics are always logged:
    train_loss, raw_loss, valid_loss, time
    The following metrics can be logged if arguments are provided:
    accuracy, f1_macro, precision, recall
    They are to be given as metric_names arguments in the exact same way
    """


    def pretty_print_train(self):
        """
        Prints the train logs at the end of training
        """
        print("\n"+"="*70)
        print(" "*27+"TRAINIG LOG")
        print("="*70)
        
        fields = ["train_loss", "valid_loss", "accuracy", "f1_macro", "time"]
        print(" Epochs | T-Loss | V-Loss |  Accu  |F1-Score| Time")
        epochs = len(self.recorder['train_loss'])

        # print(self.recorder)
        for e in range(epochs):
            print(f"\n{e}"+" "*7, end="|")
            for f in fields:
                print(f"{self.recorder[f][e]:.4f}  ", end="|")
        print("\n")