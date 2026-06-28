from layer import Layer
import numpy as np


class Dense(Layer):
    def __init__(self, input_size, output_size, init_mode="he"):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Initialization scale depends on the activation that follows,
        # because each one has a different sensitivity to input variance:
        # He -> ReLU and variants (compensates for ReLU zeroing out
        #       roughly half the activations).
        # Glorot/Xavier -> sigmoid, tanh, softmax, linear (symmetric
        #       activations, no ReLU-style bias).
        # LeCun -> SELU (designed for self-normalizing networks).
        if init_mode == "he":
            std = np.sqrt(2.0 / input_size)
        elif init_mode == "glorot":
            std = np.sqrt(2.0 / (input_size + output_size))
        elif init_mode == "lecun":
            std = np.sqrt(1.0 / input_size)
        else:
            std = 0.01

        self.params["weights"] = np.random.randn(output_size, input_size) * std
        self.params["bias"] = np.zeros((output_size, 1))

    def forward(self, input_data):
        # Fail loudly with a clear message instead of letting a shape
        # mismatch surface as a cryptic numpy error several layers later.
        assert input_data.shape[0] == self.input_size, (
            f"Dense: expected input_size={self.input_size}, "
            f"got input with shape[0]={input_data.shape[0]}."
        )

        self.input = input_data
        # Z = W * X + b
        self.output = np.dot(self.params["weights"], self.input) + self.params["bias"]
        return self.output

    def backward(self, output_gradient, optimizer=None):
        # The layer only computes gradients here; how they're applied
        # to the weights is entirely the optimizer's responsibility.
        self.grads["weights"] = np.dot(output_gradient, self.input.T)
        self.grads["bias"] = np.sum(output_gradient, axis=1, keepdims=True)
        input_gradient = np.dot(self.params["weights"].T, output_gradient)

        if optimizer is not None:
            optimizer.update(self)

        return input_gradient