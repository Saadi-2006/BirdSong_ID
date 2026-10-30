"""BirdSong ID - neural network written FROM SCRATCH in NumPy.

Implements the basic ANN concepts by hand (no Keras/TensorFlow):
  - Dense (fully-connected) layers:  z = X @ W + b
  - ReLU activation:                 a = max(0, z)
  - Softmax output:                  p_i = exp(z_i) / sum(exp(z_j))
  - Cross-entropy loss:              L = -mean(log(p_correct))
  - Backpropagation (chain rule) for gradients
  - Mini-batch gradient descent with learning rate
"""
import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_deriv(z):
    return (z > 0).astype(float)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y):
    """Mean cross-entropy. y: (N,) integer labels."""
    n = y.shape[0]
    return -np.log(probs[np.arange(n), y] + 1e-12).mean()


class Dense:
    """One fully-connected layer: forward + backward, holds W, b and grads."""

    def __init__(self, in_dim, out_dim, rng):
        # He initialisation (good for ReLU)
        self.W = rng.normal(0, np.sqrt(2.0 / in_dim), size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        return self.z

    def backward(self, dz, lr):
        """dz = dL/dz for this layer. Returns dL/dx. Updates W, b in place."""
        n = self.x.shape[0]
        dW = self.x.T @ dz / n
        db = dz.mean(axis=0)
        dx = dz @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dx


class MLP:
    """Feedforward network: Dense -> ReLU -> Dense -> ReLU -> ... -> Dense -> Softmax."""

    def __init__(self, layer_dims, seed=42):
        rng = np.random.default_rng(seed)
        self.layers = [Dense(layer_dims[i], layer_dims[i + 1], rng)
                       for i in range(len(layer_dims) - 1)]

    def forward(self, x, train=True):
        self.zs, self.a_in = [], x
        a = x
        for i, layer in enumerate(self.layers):
            z = layer.forward(a)
            self.zs.append(z)
            a = relu(z) if i < len(self.layers) - 1 else z
        self.logits = a
        return softmax(a)

    def backward(self, probs, y, lr):
        n = y.shape[0]
        # dL/dlogits for softmax + cross-entropy (the neat cancellation)
        dz = probs.copy()
        dz[np.arange(n), y] -= 1.0
        # backprop through layers (ReLU derivative on hidden layers)
        for i in reversed(range(len(self.layers))):
            if i < len(self.layers) - 1:
                dz = dz * relu_deriv(self.zs[i])
            dz = self.layers[i].backward(dz, lr)

    def predict(self, x):
        return self.forward(x, train=False).argmax(axis=1)

    def predict_proba(self, x):
        return self.forward(x, train=False)


def train_mlp(model, Xtr, ytr, Xva, yva, lr=0.05, epochs=120, batch=64,
              seed=42, verbose=True):
    """Mini-batch gradient descent. Returns history dict + best-val weights."""
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_W = -1, None

    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            probs = model.forward(Xtr[idx], train=True)
            model.backward(probs, ytr[idx], lr)

        # epoch metrics
        ptr = model.predict_proba(Xtr)
        pva = model.predict_proba(Xva)
        tr_loss, va_loss = cross_entropy(ptr, ytr), cross_entropy(pva, yva)
        tr_acc = (ptr.argmax(1) == ytr).mean()
        va_acc = (pva.argmax(1) == yva).mean()
        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(va_loss); hist["val_acc"].append(va_acc)
        if va_acc > best_acc:
            best_acc = va_acc
            best_W = [(l.W.copy(), l.b.copy()) for l in model.layers]
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"epoch {ep:3d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.3f}", flush=True)

    # restore best-validation weights
    for layer, (W, b) in zip(model.layers, best_W):
        layer.W, layer.b = W, b
    return hist, best_acc
