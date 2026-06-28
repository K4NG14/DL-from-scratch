import numpy as np


def softmax(logits):
    shift = logits - np.max(logits, axis=0, keepdims=True)
    exps = np.exp(shift)
    return exps / np.sum(exps, axis=0, keepdims=True)


def softmax_cross_entropy(y_true, logits):
    """
    Combined Softmax + Cross-Entropy loss, operating directly on raw
    logits (the output of a Dense layer with no activation). Fusing
    both steps into one function avoids having a Softmax layer assume
    it's always followed by cross-entropy, and is the standard approach
    used by frameworks like PyTorch (CrossEntropyLoss).
    """
    y_pred = softmax(logits)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    m = y_true.shape[1]
    return -np.sum(y_true * np.log(y_pred)) / m


def softmax_cross_entropy_prime(y_true, logits):
    """Gradient w.r.t. the logits: (softmax(logits) - y_true) / m."""
    y_pred = softmax(logits)
    m = y_true.shape[1]
    return (y_pred - y_true) / m


def categorical_cross_entropy(y_true, y_pred):
    """
    Expects y_pred to already be probabilities (e.g. the output of a
    standalone Softmax layer). For classification training where the
    model ends in raw logits, use softmax_cross_entropy instead.
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    m = y_true.shape[1]
    loss = -np.sum(y_true * np.log(y_pred)) / m
    return loss


def categorical_cross_entropy_prime(y_true, y_pred):
    """
    Only valid when y_pred comes from a Softmax whose Jacobian is being
    simplified to (pred - true) by the loss itself. If the Softmax
    layer computes its real Jacobian (see activations.py), use
    softmax_cross_entropy_prime on raw logits instead.
    """
    m = y_true.shape[1]
    return (y_pred - y_true) / m


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def binary_cross_entropy_prime(y_true, y_pred):
    m = y_true.shape[1]
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / m