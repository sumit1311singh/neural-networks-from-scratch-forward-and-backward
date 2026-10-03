"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    # TODO: Estimate the gradient of scalar f w.r.t. array x via central finite differences
    grad = np.zeros_like(x, dtype=float)

    for idx in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[idx] += eps
        x_minus[idx] -= eps

        grad[idx] = (f(x_plus) - f(x_minus))/(2*eps)

    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    # TODO: Return max relative error between analytic and numeric gradients.
    analytic_grad, numeric_grad = np.asarray(analytic_grad), np.asarray(numeric_grad)

    errors = np.abs(analytic_grad-numeric_grad)/np.maximum(np.abs(analytic_grad), np.maximum(np.abs(numeric_grad), tol))

    return float(np.max(errors))

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    # TODO: your approach here
    W, b = weight_init_fn(in_dim, out_dim)

    def forward_fn(x):
      y = x @ W + b
      cache = {
        'x': x,
        'W': W,
        'b': b
        }
      return y, cache

    def backward_fn(dout, cache):
      x = cache['x']
      W = cache['W']

      dx = dout @ W.T
      dW = x.T @ dout
      db = np.sum(dout, axis=0)

      grads = {
        'W': dW, 
        'b': db
        }
      return dx, grads

    layer = {
      'params': {
        'W': W, 
        'b': b
        },
      'forward': forward_fn,
      'backward': backward_fn
    }

    return layer

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    # TODO: your approach here
    if kind == 'relu':
      def forward_fn(x):
        y = np.maximum(x, 0)
        cache = {
          'x': x
        }
        return y, cache
      def backward_fn(dout, cache):
        x = cache['x']
        dx = dout * np.where(x<0, 0, 1)
        return dx, {}
    elif kind == 'tanh':
      def forward_fn(x):
        y = np.tanh(x)
        cache = {
          'tanh_x': y
        }
        return y, cache
      def backward_fn(dout, cache):
        tanh_x = cache['tanh_x']
        dx = dout * (1-tanh_x**2)
        return dx, {}
    elif kind == 'sigmoid':
      def forward_fn(x):
        exp = np.exp(x)
        y = exp/(1+exp)
        cache = {
          'sigmoid_x': y
        }
        return y, cache
      def backward_fn(dout, cache):
        sigmoid_x = cache['sigmoid_x']
        dx = dout * (sigmoid_x * (1-sigmoid_x))
        return dx, {}

    layer = {
        'params': {},
        'forward': forward_fn,
        'backward': backward_fn
      }
    return layer

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

