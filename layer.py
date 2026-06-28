class Layer:
    """
    Base class for every layer in the network.

    Design:
    - `params` / `grads`: layers with trainable weights (e.g. Dense) expose
      them here as dicts. An external Optimizer reads `grads` and writes
      `params`. The layer itself never decides HOW a weight is updated —
      it only computes gradients. This keeps optimizers (SGD, Adam, ...)
      fully decoupled from layer implementations.
    - `training`: train/eval flag. Layers like Dropout or BatchNorm read
      this to behave differently at train time vs inference time.
    """

    def __init__(self):
        self.input = None
        self.output = None
        self.training = True

        # Subclasses with trainable parameters populate these dicts.
        # Example in Dense: self.params = {"weights": W, "bias": b}
        self.params = {}
        self.grads = {}

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_gradient, optimizer=None):
        """
        Computes the gradient w.r.t. the layer's input. If the layer has
        trainable parameters, it must store gradients in self.grads and
        call optimizer.update(self) when an optimizer is provided.
        """
        raise NotImplementedError

    def set_mode(self, training: bool):
        """Toggle train/eval mode. Composite layers (e.g. ResidualBlock)
        override this to propagate the flag to their sublayers."""
        self.training = training