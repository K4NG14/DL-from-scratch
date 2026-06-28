import numpy as np


class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output, optimizer):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad, optimizer)

    def set_mode(self, training: bool):
        """
        Switch every layer between train and eval mode. Layers like
        Dropout or BatchNorm rely on this to behave differently during
        training versus inference (e.g. Dropout must not zero out
        activations at inference time).
        """
        for layer in self.layers:
            layer.set_mode(training)

    def train(self, x_train, y_train, loss_func, loss_prime, optimizer,
               epochs, batch_size=None, verbose=True):
        self.set_mode(True)

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
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[:, i:i + batch_size]
                y_batch = y_shuffled[:, i:i + batch_size]

                output = self.forward(x_batch)
                loss += loss_func(y_batch, output)
                n_batches += 1

                grad = loss_prime(y_batch, output)
                self.backward(grad, optimizer)

            loss /= n_batches

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

    def predict(self, input_data):
        """Raw model output, computed in eval mode."""
        self.set_mode(False)
        return self.forward(input_data)

    def predict_classes(self, input_data):
        """Index of the most likely class, for classification tasks."""
        output = self.predict(input_data)
        return np.argmax(output, axis=0)

    def save_weights(self, filepath):
        """
        Saves every layer's trainable parameters to a single .npz file.
        Layers without parameters (activations) are skipped automatically.
        """
        to_save = {}
        for idx, layer in enumerate(self.layers):
            for key, value in getattr(layer, "params", {}).items():
                to_save[f"layer{idx}__{key}"] = value
        np.savez(filepath, **to_save)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath):
        """
        Loads parameters saved by save_weights. The layer list must
        match the architecture used when saving (same order, same
        shapes) — this restores values, it doesn't reconstruct layers.
        """
        data = np.load(filepath)
        for idx, layer in enumerate(self.layers):
            for key in getattr(layer, "params", {}):
                npz_key = f"layer{idx}__{key}"
                if npz_key not in data:
                    raise KeyError(
                        f"'{npz_key}' not found in {filepath}. "
                        "Does the architecture match the one you saved?"
                    )
                expected_shape = layer.params[key].shape
                loaded = data[npz_key]
                assert loaded.shape == expected_shape, (
                    f"Shape mismatch for {npz_key}: expected "
                    f"{expected_shape}, got {loaded.shape}"
                )
                layer.params[key] = loaded
        print(f"Weights loaded from {filepath}")