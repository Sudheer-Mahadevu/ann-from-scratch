"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
from ann import NeuralNetwork
import numpy as np

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
    
    parser.add_argument('-b', '--batch_size', type=int, default = 32,
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
                        nargs='+', default = [32],
                        help='List with number of neurons in each hidden layer')

    parser.add_argument('-a', '--activation', type=str, default='sigmoid',
                        choices=['sigmoid', 'tanh', 'relu'],
                    help='Activation function to use in each hidden layer')

    parser.add_argument('-w_i', '--weight_init', type=str, default='random',
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
    args = parse_arguments()
    print(args)
    # check that hidden size list has len of num_layers
    model = NeuralNetwork(args.hidden_size,args.weight_init,args.activation,
                          args.loss, args.verbose)
    X = np.array([[1,2]])
    Y = np.array([[1,0]])
    
    model.train(X,Y,0,0)
    print("Training complete!")


if __name__ == '__main__':
    main()
