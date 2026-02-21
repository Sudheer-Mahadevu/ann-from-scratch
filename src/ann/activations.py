"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def sigmoid_derivative(Z):
    s = sigmoid(Z)
    return s*(1-s)

def relu(Z):
    return np.maximum(0,Z)

def relu_derivative(Z):
    return (Z > 0)

def tanh(Z):
    return np.tanh(Z)

def tanh_derivate(Z):
    t = tanh(Z)
    return 1-np.square(t)

def identity(Z): return Z

def identity_derivative(Z): return np.ones(Z.shape)

def softmax(Z):
    exp_Z = np.exp(Z)
    y_cap = exp_Z/ np.sum(exp_Z, axis=1, keepdims=True)
    return y_cap

ACTIVATION_MAP = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu' : (relu, relu_derivative),
    'tanh' : (tanh, tanh_derivate),
    'identity': (identity, identity_derivative),
    'softmax' : softmax,
}