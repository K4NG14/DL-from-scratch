import numpy as np
from losses import *

class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output, learning_rate):
        grad = grad_output
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def train(self, x_train, y_train, loss_func, loss_prime, epochs, learning_rate, batch_size=None, verbose=True):
        if batch_size is None:
            batch_size = x_train.shape[1]
        
        n_samples = x_train.shape[1]
        
        print(f"Training on {n_samples} examples with batch_size={batch_size}...")
        
        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            x_shuffled = x_train[:, indices]
            y_shuffled = y_train[:, indices]

            loss = 0
            
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[:, i : i + batch_size]
                y_batch = y_shuffled[:, i : i + batch_size]
                
                output = self.forward(x_batch)
        
                loss += loss_func(y_batch, output)

                grad = loss_prime(y_batch, output)
                self.backward(grad, learning_rate)
            
            loss /= (n_samples / batch_size)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

    def predict(self, input_data):
        output = self.forward(input_data)
        return np.argmax(output, axis=0)


