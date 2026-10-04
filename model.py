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

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    # TODO: your approach here
    if scheme == 'he':
      s = np.sqrt(2/in_dim)
    else:
      s = np.sqrt(2/(in_dim+out_dim))
    
    W = s * np.random.randn(in_dim, out_dim)
    b = np.zeros(out_dim)

    return W, b

# Step 6 - make_loss
import numpy as np

def make_loss(kind='cross_entropy'):
  """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

  Inputs to loss_fn:
    logits: (batch, C) float array of raw class scores
    labels: (batch,) int array of class indices in [0, C)
  Outputs:
    loss: Python float, mean scalar loss over the batch (finite)
    d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
  Must pass gradient_check, be minimized by confident correct predictions,
  and stay finite under saturated logits.
  """
  # TODO: your approach here
  if kind != 'cross_entropy':
    raise ValueError("Only 'cross_entropy' is supported.")
    
  def loss_fn(logits, labels):
    """
    logits: (batch, C)
    labels: (batch,) integer class indices
    Returns:
      loss: Python float
      d_logits: (batch, C)
    """

    # Numerical stability: subtract max per row
    shifted = logits - np.max(logits, axis=1, keepdims=True)

    # logsumexp
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))

    # correct-class logits
    batch = logits.shape[0]
    correct_logits = shifted[np.arange(batch), labels]

    # cross-entropy loss (mean)
    loss = np.mean(logsumexp.squeeze() - correct_logits)

    # softmax for gradient
    exp_shifted = np.exp(shifted)
    softmax = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    # gradient: (softmax - one_hot) / batch
    d_logits = softmax
    d_logits[np.arange(batch), labels] -= 1.0
    d_logits /= batch

    return float(loss), d_logits

  return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    # TODO: your approach here
    def forward_fn(x):
      current = x
      caches = []
      
      for layer in layers:
        y, cache = layer['forward'](current)
        current = y
        caches.append(cache)

      return current, caches

    def backward_fn(dout, caches):
      grad_list = []
      n = len(layers)
      for i, layer in enumerate(reversed(layers)):
        dx, grads_dict = layer['backward'](dout, caches[n-i-1])
        dout = dx
        grad_list.append(grads_dict)

      return dout, list(reversed(grad_list))

    params = []
    for layer in layers:
      params.append(layer['params'])

    return {
      'forward': forward_fn,
      'backward': backward_fn,
      'params': params
    }

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """
    # TODO: your approach here
    logits, caches = model['forward'](x)

    loss, d_logits = loss_fn(logits, y)

    dx, param_grads =  model['backward'](d_logits, caches)

    return loss, param_grads

# Step 9 - make_optimizer
def make_optimizer(params, lr=1e-2, kind='sgd'):
    """Build an optimizer that updates params in place.

    Inputs:
      params: arrays, possibly nested in lists/dicts (or dict of arrays) to optimize
      lr: float learning rate
      kind: str algorithm name (e.g. 'sgd')

    Returns:
      dict with key 'step'. step(grads) applies one in-place update
      using grads structured like params. Parameter shapes must stay
      unchanged. Repeated steps must reduce a simple convex objective
      within a modest fixed budget and keep values finite.
    """
    # TODO: your approach here
    if kind != 'sgd':
      raise ValueError("Only 'sgd' is supported.")
      
    def walk(x):
        if isinstance(x, np.ndarray):
            return
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
            return
        if isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
            return
        raise TypeError(
            "params must contain only numpy arrays, lists/tuples, and dicts"
        )

    walk(params)

    def step(grads):
        def update(p, g):
            if isinstance(p, np.ndarray):
                if not isinstance(g, np.ndarray):
                    raise TypeError("gradient structure does not match params")
                if p.shape != g.shape:
                    raise ValueError(
                        f"gradient shape {g.shape} does not match parameter shape {p.shape}"
                    )

                # In-place update: the ndarray object itself is preserved.
                p[...] -= lr * g
                return

            if isinstance(p, dict):
                if not isinstance(g, dict) or list(p.keys()) != list(g.keys()):
                    raise ValueError("gradient structure does not match params")
                for key in p:
                    update(p[key], g[key])
                return

            if isinstance(p, (list, tuple)):
                if not isinstance(g, type(p)) or len(p) != len(g):
                    raise ValueError("gradient structure does not match params")
                for pv, gv in zip(p, g):
                    update(pv, gv)
                return

            raise TypeError("invalid parameter structure")

        update(params, grads)

    return {'step': step}

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

