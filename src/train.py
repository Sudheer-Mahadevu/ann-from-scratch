"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
from ann import NeuralNetwork
import numpy as np
from utils import MNISTLoader
from ann import Optimizer
from types import SimpleNamespace
import wandb
from api_keys import WANDB_API_KEY, WANDB_ENTITY
from utils import train_with_wandb_sweep, sweep_config

def parse_arguments():
    """
    Parse command-line arguments.

    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('-d','--dataset', type=str, default='mnist',
                        choices=['mnist','fashion_mnist'], 
                        help='Dataset to use')
    
    parser.add_argument('-e','--epochs', type=int, default= 5,
                             help = "Number of epochs to train")
    
    parser.add_argument('-b', '--batch_size', type=int, default = 128,
                        help='Mini-batch size')
    
    parser.add_argument('-l', '--loss', type=str, default = 'cross_entropy',
                        choices=['mean_squared_error', 'cross_entropy'],
                        help='Loss function to use')
    
    parser.add_argument('-o', '--optimizer', type=str, default='momentum',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 
                                 'adam','nadam'], 
                        help='Optimizer for gradient descent')

    parser.add_argument('-lr', '--learning_rate', type=float, default = 0.001,
                      help = 'Initial learning rate')

    parser.add_argument('-wd', '--weight_decay', type=float, default = 0,
                        help="Weight decay for L2 regularization")

    parser.add_argument('-nhl','--num_layers', type=int, default = 1,
                        help='Number of hidden layers')

    parser.add_argument('-sz', '--hidden_size', type=int, 
                        nargs='+', default = [128, 128, 128],
                        help='List with number of neurons in each hidden layer')

    parser.add_argument('-a', '--activation', type=str, default='sigmoid',
                        choices=['sigmoid', 'tanh', 'relu'],
                    help='Activation function to use in each hidden layer')

    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier'], 
                        help='Weight initialization method')
    
    parser.add_argument('-wp','--wandb_project', type=str, 
                        default='me21b102_assignmnet1', help='W&B project name')
    
    parser.add_argument('-mp','--model_path', type=str, 
                        default='../models/model.pth', help='Path to save model')
    
    parser.add_argument('-v','--verbose',action='store_true',
                        help='Print debugging information')

    return parser.parse_args()


def main():
    """
    Main training function.
    """
    wandb.login(key = WANDB_API_KEY)
    sweep_id = wandb.sweep(sweep_config, entity=WANDB_ENTITY, 
                           project="da6401-assignment1")
    
    wandb.agent(sweep_id, function=train_with_wandb_sweep, count=2)

def train_model(args):
    
    dls = MNISTLoader(args.dataset,val_split=0.2, batch_size=args.batch_size)
    model = NeuralNetwork(args.hidden_size,args.weight_init,args.activation,
                          args.loss)
    optimizer = Optimizer(args.optimizer, model.layers)
    model.train(dls,optimizer,args.epochs,args.learning_rate)
    return






if __name__ == '__main__':
    main()
