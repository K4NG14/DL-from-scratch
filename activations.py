from layer import Layer
import numpy as np

class ReLU(Layer):
    def forward(self, input_data):
        self.input = input_data
        self.output = np.maximum(0, input_data)
        return self.output

    def backward(self, output_gradient, learning_rate=None):
        # ReLU': 1 if x > 0, else 0
        relu_grad = (self.input > 0).astype(float)
        return output_gradient * relu_grad


class Softmax(Layer):
    def forward(self, input_data):
        shift_x = input_data - np.max(input_data, axis=0, keepdims=True)
        exps = np.exp(shift_x)
        self.output = exps / np.sum(exps, axis=0, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate=None):
        """
        IMPORTANT:
        The gradient of Softmax is a complex Jacobian matrix. 
        But when combined with Categorical Cross-Entropy, the gradient 
        simplifies to: (Predictions - True_Labels).
        This layer acts assumes the Loss function has 
        already handled the combined derivative.
        """
        return output_gradient
    
class Sigmoid(Layer):
    def forward(self, input_data):
        self.input = input_data
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Sigmoid': s * (1 - s)
        s = self.output
        sigmoid_grad = s * (1 - s)
        return output_gradient * sigmoid_grad