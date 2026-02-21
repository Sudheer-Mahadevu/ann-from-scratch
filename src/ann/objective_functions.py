"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Returns Cross Entropy Loss averaged across mini-batch

    shape of y_ture, y_pred : (n,k)
    n: mini-batch size,
    k: number of classes
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)

    loss = -np.sum(y_true * np.log(y_pred))/y_pred.shape[0]

    return loss

def cross_entropy_delta(y_true, y_pred):
    """
    Returns error in the last layer dL/dz for CE loss, softmax combination
    """
    return y_pred-y_true

def MSE_loss(y_true,y_pred):
    """
    Returns MSE loss averaged across mini-batch and number of classes
    """
    errors = np.square(y_pred-y_true)
    loss = np.sum(errors)/(y_pred.shape[0]*y_pred.shape[1])

    return loss

def MSE_delta(y_true, y_pred):
    """
    Returns error in the last layer dL/dz for MSE loss, 
    identiy activation combination
    """
    k = y_pred.shape[1] # number of classes
    return 2*(y_pred-y_true)/k

LOSS_FUNCTIONS = {
    'cross_entropy': (cross_entropy_loss,cross_entropy_delta),
    'mean_squared_error' : (MSE_loss, MSE_delta),
}