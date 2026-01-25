import numpy as np

def categorical_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    m = y_true.shape[1]
    loss = -np.sum(y_true * np.log(y_pred)) / m
    return loss

def categorical_cross_entropy_prime(y_true, y_pred):
    m = y_true.shape[1]
    return (y_pred - y_true) / m

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def binary_cross_entropy(y_true, y_pred):
    m = y_true.shape[1]
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def binary_cross_entropy_prime(y_true, y_pred):
    m = y_true.shape[1]
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / m