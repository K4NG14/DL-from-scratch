# DNN Framework - Architecture and design decisions

Minimal neural network framework in pure NumPy, designed to replicate known architectures (MLPs, ResNets, classifiers) without relying on PyTorch/TensorFlow. It prioritizes transparent internal mechanisms over execution speed.

## File structure

| File | Responsibility |
| --- | --- |
| `layer.py` | Base `Layer` class: `forward`/`backward`, params/grads, train/eval mode |
| `dense.py` | Fully-connected layer |
| `activations.py` | `ReLU`, `Sigmoid`, `Softmax` |
| `losses.py` | Loss functions and their derivatives |
| `optimizers.py` | `SGD` (with momentum), `Adam` |
| `blocks.py` | `ResidualBlock` for ResNet-like architectures |
| `network.py` | Orchestrates forward/backward passes, training, save/load |

## Data flow

```
X → [Layer 1] → [Layer 2] → ... → [Layer N] → loss_func(y_true, output)
                                                       ↓
grad_input ← backward ← backward ← ... ← backward ← loss_prime(y_true, output)
                ↑
          optimizer.update(layer)  (only in layers with params)

```

`Network` knows nothing about weights or how they are updated - it only iterates through layers forward and backward. This deliberate decoupling is what allows any combination of layers/optimizer/loss to work without modifying `Network`.

## Design decisions

### 1. Layers do not update themselves

Every `Layer` with weights exposes `self.params` (dict) and `self.grads` (dict). `backward()` only *computes* gradients; the one deciding how to apply them is an external `Optimizer` object, injected via `network.train(..., optimizer=...)`.

**Why:** if the optimizer were inside the layer (as in the initial version, which did `self.weights -= lr * grad` directly in `Dense`), switching from SGD to Adam would have required rewriting every single layer. With this separation, adding a new optimizer just means adding a new file in `optimizers.py`, without touching any existing layer.

### 2. Explicit train/eval mode

`Layer.training` (bool) is propagated using `Network.set_mode(bool)`. `predict()` sets it to `False` automatically; `train()` sets it to `True` at the start of each call.

**Why:** Dropout and BatchNorm (which are not implemented yet, but you will need them to replicate almost any modern architecture) behave differently during training than during inference. Without this flag, there is no way to implement them correctly later without refactoring `Network`.

### 3. `ResidualBlock` instead of a general computational graph

`Network` remains a sequential list of layers. To support skip connections (ResNet), `ResidualBlock` encapsulates a sub-list of layers and behaves just like another layer: `forward` computes `F(x) + x`, and `backward` propagates the gradient through `F` and adds the unmodified shortcut gradient to it.

**Why:** a full computational graph (like PyTorch autograd) is much more flexible, but it's a complexity you don't need yet if your goal is to replicate concrete architectures with branches known in advance (residual, not arbitrary). `ResidualBlock` covers that use case with zero changes to the `Network` engine.

### 4. Weight initialization according to activation

`Dense(init_mode=...)` supports `"he"`, `"glorot"`, `"lecun"`:

* **He** (√(2/fan_in)) → ReLU and variants (compensates for ReLU zeroing out ~half of the activations).
* **Glorot/Xavier** (√(2/(fan_in+fan_out))) → sigmoid, tanh, softmax, linear (symmetric activations).
* **LeCun** (√(1/fan_in)) → SELU.

**Why it matters:** if you are going to replicate a paper, its architecture assumes a specific initialization. Using the wrong one can cause the network not to converge the same way as the original, giving the false impression that the problem lies elsewhere.

### 5. Shape validation in `forward()`

`Dense.forward()` asserts that `input_data.shape[0] == self.input_size` before operating.

**Why:** in a multi-layer architecture, a shape mismatch in layer 3 usually manifests as a cryptic NumPy exception in layer 4. The assertion pinpoints the real origin of the problem.

### 6. Weight save/load

`Network.save_weights(path)` / `load_weights(path)` use `np.savez`, saving each parameter as `layer{idx}__{name}`. Only layers with `params` are saved/loaded (activations are ignored).

**Why:** replicating a known architecture often involves comparing against pre-trained weights or simply not losing hours of training progress. Without this, every Python session starts from scratch.

**Limitation to keep in mind:** `load_weights` does not reconstruct the architecture, it only restores values. The layer list must match the order and shape of the one used when saving.

### 7. `predict()` vs `predict_classes()`

`predict()` returns the raw model output (useful for regression and for obtaining probabilities). `predict_classes()` applies `argmax`.

**Why:** the previous version always applied `argmax` inside `predict()`, which is incorrect for regression tasks. Separating both methods prevents the framework from forcing a default task (classification).

## What's missing (next steps)

These pieces are not implemented yet because they weren't part of the prioritized changes, but the current design already accommodates them:

* **Dropout / BatchNorm**: the `training` flag already exists; these are the logical next step if you are going to replicate architectures with regularization.
* **Conv2D / MaxPool2D / Flatten**: necessary for CNNs (LeNet, VGG).
* **Weight decay (L2)** in the optimizers.
* **RNN/LSTM**: require recurrent state, which breaks the single-pass forward/backward pattern that the rest of the framework assumes.

## Minimal usage example

```python
from network import Network
from dense import Dense
from activations import ReLU
from losses import softmax_cross_entropy, softmax_cross_entropy_prime
from optimizers import Adam

model = Network([
    Dense(input_size=4, output_size=8, init_mode="he"),
    ReLU(),
    Dense(input_size=8, output_size=3, init_mode="glorot"),  # raw logits
])

model.train(
    X_train, y_train,
    loss_func=softmax_cross_entropy,
    loss_prime=softmax_cross_entropy_prime,
    optimizer=Adam(lr=0.001),
    epochs=100,
    batch_size=32,
)

predicted_classes = model.predict_classes(X_test)
model.save_weights("checkpoint.npz")

```

## How to extend the framework

* **New layer**: inherit from `Layer`, implement `forward` and `backward(output_gradient, optimizer=None)`. If it has weights, populate `self.params` in `__init__` and `self.grads` in `backward`.
* **New optimizer**: inherit from `Optimizer` in `optimizers.py`, implement `update(layer)` by reading `layer.grads` and writing to `layer.params`.
* **New loss**: a `loss(y_true, y_pred)` function and its derivative `loss_prime(y_true, y_pred)`, no class needed, pure functions like the ones in `losses.py` are enough.