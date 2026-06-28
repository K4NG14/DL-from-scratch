import numpy as np


class Optimizer:
    """
    Interface every optimizer must implement. update(layer) reads
    layer.grads and writes layer.params. Keeping this separate from
    Layer means any layer can be paired with any optimizer.
    """
    def update(self, layer):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic gradient descent with optional momentum. Momentum
    smooths the update direction using a running velocity per
    parameter, which helps escape shallow local minima and reduces
    oscillation.
    """
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self._velocity = {}  # keyed by (id(layer), param_name)

    def update(self, layer):
        for key, param in layer.params.items():
            grad = layer.grads[key]
            v_key = (id(layer), key)

            if v_key not in self._velocity:
                self._velocity[v_key] = np.zeros_like(param)

            # In-place updates (*=, -=, +=) avoid allocating a new array
            # on every step. With reassignment (v = m*v - lr*g) numpy
            # creates a fresh array each call; in-place mutates the
            # existing buffer, which matters once weight matrices get
            # large or training runs for many steps.
            v = self._velocity[v_key]
            v *= self.momentum
            v -= self.lr * grad
            param += v


class Adam(Optimizer):
    """
    Adam optimizer: tracks per-parameter running averages of the
    gradient (m) and its square (v), with bias correction for the
    early steps. Adapts the effective learning rate per parameter,
    which is why it's the default choice for most modern architectures
    (Transformers, ResNets, etc.).
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = {}
        self._v = {}
        self._t = {}  # step counter, per parameter

    def update(self, layer):
        for key, param in layer.params.items():
            grad = layer.grads[key]
            p_key = (id(layer), key)

            if p_key not in self._m:
                self._m[p_key] = np.zeros_like(param)
                self._v[p_key] = np.zeros_like(param)
                self._t[p_key] = 0

            self._t[p_key] += 1
            t = self._t[p_key]

            m, v = self._m[p_key], self._v[p_key]
            # In-place updates: same reasoning as in SGD above.
            m *= self.beta1
            m += (1 - self.beta1) * grad
            v *= self.beta2
            v += (1 - self.beta2) * (grad ** 2)

            # Bias correction matters most in the first few steps,
            # when m and v are still biased toward zero.
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)