from layer import Layer
import numpy as np


class ReLU(Layer):
    def forward(self, input_data):
        self.input = input_data
        self.output = np.maximum(0, input_data)
        return self.output

    def backward(self, output_gradient, optimizer=None):
        relu_grad = (self.input > 0).astype(float)
        return output_gradient * relu_grad


class Softmax(Layer):
    """
    Computes the full Softmax Jacobian per sample, so this layer is
    self-contained and safe to use anywhere (e.g. attention blocks),
    not just right before a cross-entropy loss.

    For classification training, prefer NOT placing this layer last in
    the model. Instead, let the final Dense output raw logits and use
    losses.softmax_cross_entropy / softmax_cross_entropy_prime, which
    apply softmax internally in a numerically stable way and make the
    combined derivative explicit rather than implicit in the layer.
    """
    def forward(self, input_data):
        shift_x = input_data - np.max(input_data, axis=0, keepdims=True)
        exps = np.exp(shift_x)
        self.output = exps / np.sum(exps, axis=0, keepdims=True)
        return self.output

    def backward(self, output_gradient, optimizer=None):
        # Jacobian-vector product without forming the n x n Jacobian:
        # (diag(s) - s s^T) @ g  ==  s * (g - sum(s * g, axis=0))
        # Same math as looping per sample and building the full matrix,
        # but ~200x faster since it's fully vectorized across the batch.
        s = self.output
        dot = np.sum(s * output_gradient, axis=0, keepdims=True)
        return s * (output_gradient - dot)


class Sigmoid(Layer):
    def forward(self, input_data):
        self.input = input_data
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, output_gradient, optimizer=None):
        s = self.output
        sigmoid_grad = s * (1 - s)
        return output_gradient * sigmoid_grad