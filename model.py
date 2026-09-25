import numpy as np

class Parameter:
    """
    A wrapper class for model parameters to hold their values and gradients.
    """
    def __init__(self, value):
        self.val = np.array(value, dtype=np.float32)
        self.grad = np.zeros_like(self.val)

    def zero_grad(self):
        self.grad.fill(0.0)


class Layer:
    """
    Base class for all neural network layers.
    """
    def __init__(self):
        self.trainable = False
        self.params = []
        self.mode = 'train'  # 'train' or 'test'

    def forward(self, X):
        raise NotImplementedError

    def backward(self, dY):
        raise NotImplementedError

    def set_mode(self, mode):
        self.mode = mode


class Linear(Layer):
    """
    Fully connected layer: Y = XW + b
    """
    def __init__(self, in_features, out_features, init_type='he'):
        super().__init__()
        self.trainable = True
        
        # Initialization
        if init_type == 'he':
            # He initialization for ReLU
            std = np.sqrt(2.0 / in_features)
        elif init_type == 'xavier':
            # Xavier initialization for Sigmoid/Tanh
            std = np.sqrt(2.0 / (in_features + out_features))
        else:
            # Simple standard normal
            std = 0.01
            
        W_val = np.random.normal(0.0, std, (in_features, out_features))
        b_val = np.zeros((1, out_features))
        
        self.W = Parameter(W_val)
        self.b = Parameter(b_val)
        self.params = [self.W, self.b]
        
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W.val) + self.b.val

    def backward(self, dY):
        # dY shape: (batch_size, out_features)
        # X shape: (batch_size, in_features)
        # W shape: (in_features, out_features)
        # dW = X.T * dY
        self.W.grad += np.dot(self.X.T, dY)
        # db = sum(dY, axis=0)
        self.b.grad += np.sum(dY, axis=0, keepdims=True)
        # dX = dY * W.T
        return np.dot(dY, self.W.val.T)


class ReLU(Layer):
    """
    Rectified Linear Unit activation.
    """
    def __init__(self):
        super().__init__()
        self.X = None

    def forward(self, X):
        self.X = X
        return np.maximum(0.0, X)

    def backward(self, dY):
        return dY * (self.X > 0.0)


class LeakyReLU(Layer):
    """
    Leaky Rectified Linear Unit activation.
    """
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.X = None

    def forward(self, X):
        self.X = X
        return np.where(X > 0.0, X, X * self.alpha)

    def backward(self, dY):
        dx = np.ones_like(self.X)
        dx[self.X <= 0.0] = self.alpha
        return dY * dx


class Sigmoid(Layer):
    """
    Sigmoid activation function.
    """
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, X):
        # Clip X to prevent overflow in exp
        clipped_X = np.clip(X, -88.0, 88.0)
        self.out = 1.0 / (1.0 + np.exp(-clipped_X))
        return self.out

    def backward(self, dY):
        return dY * self.out * (1.0 - self.out)


class Tanh(Layer):
    """
    Hyperbolic tangent activation function.
    """
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, X):
        self.out = np.tanh(X)
        return self.out

    def backward(self, dY):
        return dY * (1.0 - self.out ** 2)


class Dropout(Layer):
    """
    Dropout layer for regularization.
    """
    def __init__(self, drop_rate=0.2):
        super().__init__()
        self.drop_rate = drop_rate
        self.mask = None

    def forward(self, X):
        if self.mode == 'train' and self.drop_rate > 0.0:
            # Keep probability is 1 - drop_rate
            keep_prob = 1.0 - self.drop_rate
            self.mask = (np.random.rand(*X.shape) < keep_prob) / keep_prob
            return X * self.mask
        else:
            self.mask = None
            return X

    def backward(self, dY):
        if self.mode == 'train' and self.drop_rate > 0.0 and self.mask is not None:
            return dY * self.mask
        else:
            return dY


class Sequential(Layer):
    """
    A container that sequences layers and propagates forward/backward.
    """
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.params = []
        for layer in self.layers:
            if layer.trainable:
                self.params.extend(layer.params)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dY):
        grad = dY
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def set_mode(self, mode):
        self.mode = mode
        for layer in self.layers:
            layer.set_mode(mode)


# Optimizers

class Optimizer:
    def __init__(self, params, lr=0.001, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with momentum.
    """
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.val) for p in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            grad = param.grad
            # Apply L2 regularization (weight decay)
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * param.val
                
            # Momentum update
            self.velocities[i] = self.momentum * self.velocities[i] + self.lr * grad
            param.val -= self.velocities[i]


class Adam(Optimizer):
    """
    Adam optimizer.
    """
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.val) for p in self.params]
        self.v = [np.zeros_like(p.val) for p in self.params]

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            grad = param.grad
            # Apply L2 regularization (weight decay)
            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * param.val

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            param.val -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
