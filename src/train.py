"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
from ann import NeuralNetwork
import numpy as np
from utils import MNISTLoader
from ann import Optimizer

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
    
    parser.add_argument('-e','--epochs', type=int, default= 10,
                             help = "Number of epochs to train")
    
    parser.add_argument('-b', '--batch_size', type=int, default = 64,
                        help='Mini-batch size')
    
    parser.add_argument('-l', '--loss', type=str, default = 'mean_squared_error',
                        choices=['mean_squared_error', 'cross_entropy'],
                        help='Loss function to use')
    
    parser.add_argument('-o', '--optimizer', type=str, default='rmsprop',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 
                                 'adam','nadam'], 
                        help='Optimizer for gradient descent')

    parser.add_argument('-lr', '--learning_rate', type=float, default = 0.0025,
                      help = 'Initial learning rate')

    parser.add_argument('-wd', '--weight_decay', type=float, default = 0,
                        help="Weight decay for L2 regularization")

    parser.add_argument('-nhl','--num_layers', type=int, default = 3,
                        help='Number of hidden layers')

    parser.add_argument('-sz', '--hidden_size', type=int, 
                        nargs='+', default = [128, 128, 128],
                        help='List with number of neurons in each hidden layer')

    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['sigmoid', 'tanh', 'relu'],
                    help='Activation function to use in each hidden layer')

    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier'], 
                        help='Weight initialization method')
    
    parser.add_argument('-wp','--wandb_project', type=str, 
                        default='me21b102_assignmnet1', help='W&B project name')
    
    parser.add_argument('-mp','--model_path', type=str, 
                        default='best_model.npy', help='Path to save model')
    
    parser.add_argument('-v','--verbose',action='store_true',
                        help='Print debugging information')
    
    parser.add_argument('-r', '--runs', type=int, default=1, 
                        help="number of hyper parameter sweep runs")

    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()
    model = NeuralNetwork(args)
    # weights = load_model(args.model_path)
    # model.set_weights(weights)
    dls = MNISTLoader(args.dataset,val_split=0.2, batch_size=args.batch_size)
    model.train(dls.x_train,dls.y_train,epochs = 10, batch_size = 64)
    metrics = model.evaluate(dls.x_val,dls.y_val, metric_names=['accuracy','f1_macro'])
    print(metrics)
    # optimizer = Optimizer(args.optimizer, model.layers)
    # model.train_custom(dls,optimizer,args.epochs,args.learning_rate)

    # best_weights = model.get_weights()
    # np.save("best_model.npy", best_weights)
    

def load_model(model_path):
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def train_model(args):
    
    dls = MNISTLoader(args.dataset,val_split=0.2, batch_size=args.batch_size)
    model = NeuralNetwork(args.hidden_size,args.weight_init,args.activation,
                          args.loss, args.verbose)
    optimizer = Optimizer(args.optimizer, model.layers)
    model.train(dls,optimizer,args.epochs,args.learning_rate)
    return






if __name__ == '__main__':
    main()
