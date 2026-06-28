from layer import Layer


class ResidualBlock(Layer):
    """
    Residual block: output = F(x) + x, where F(x) is the sub-network
    defined by `sublayers`. The block behaves as a single Layer, so it
    can be dropped directly into a Network's layer list — this avoids
    turning Network into a general computation graph while still
    supporting skip connections, the core building block of ResNet-style
    architectures.

    F(x)'s output must match the input's shape, exactly as in a real
    residual connection, since the two are summed elementwise.
    """
    def __init__(self, sublayers):
        super().__init__()
        self.sublayers = sublayers

    def forward(self, input_data):
        self.input = input_data
        out = input_data
        for layer in self.sublayers:
            out = layer.forward(out)

        assert out.shape == input_data.shape, (
            "ResidualBlock requires F(x) to match the input's shape. "
            f"Input: {input_data.shape}, F(x) output: {out.shape}"
        )

        self.output = out + input_data
        return self.output

    def backward(self, output_gradient, optimizer=None):
        grad_through_block = output_gradient
        for layer in reversed(self.sublayers):
            grad_through_block = layer.backward(grad_through_block, optimizer)

        # The skip connection bypasses every sublayer, so its gradient
        # is just output_gradient passed through unchanged — mirroring
        # the forward pass (out = F(x) + x).
        return grad_through_block + output_gradient

    def set_mode(self, training: bool):
        self.training = training
        for layer in self.sublayers:
            layer.set_mode(training)