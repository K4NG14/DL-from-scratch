from layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        std = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(output_size, input_size) * std
        self.bias = np.zeros((output_size, 1))

    def forward(self, input_data):
        self.input = input_data
        # Z = W * X + b
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):

        # Gradients
        weights_gradient = np.dot(output_gradient, self.input.T) 
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True) 
        input_gradient = np.dot(self.weights.T, output_gradient)

        # Gradient Descent
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient