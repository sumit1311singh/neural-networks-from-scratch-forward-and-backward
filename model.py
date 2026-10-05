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
def make_optimizer(params, lr=1e-2, kind='sgd', weight_decay=0.0):
    """Build an optimizer that updates params in place.

    Inputs:
      params: arrays, possibly nested in lists/dicts (or dict of arrays) to optimize
      lr: float learning rate
      kind: str algorithm name (e.g. 'sgd')
      weight_decay: L2 weight decay coefficient

    Returns:
      dict with key 'step'. step(grads) applies one in-place update
      using grads structured like params.
    """
    if kind != 'sgd':
        raise ValueError("Only 'sgd' is supported.")

    def step(grads):

        def update(p, g):

            if isinstance(p, np.ndarray):
                p[...] -= lr * (g + weight_decay * p)
                return

            if isinstance(p, dict):
                for key in p:
                    update(p[key], g[key])
                return

            if isinstance(p, (list, tuple)):
                for pv, gv in zip(p, g):
                    update(pv, gv)
                return

        update(params, grads)

    return {'step': step}

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.

    Inputs:
      model: sequential model dict with 'forward', 'backward', and 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x_batch: np.ndarray of shape (B, D)
      y_batch: np.ndarray of shape (B,) integer class labels

    Returns:
      loss: float, scalar batch loss evaluated BEFORE the parameter update.
      Model parameters are updated in place; shapes unchanged and values finite.
    """
    # TODO: your approach here
    loss, param_grads = forward_backward(model, loss_fn, x_batch, y_batch)

    optimizer['step'](param_grads)

    return loss

# Step 11 - train
def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    """Run a deterministic minibatch training loop.

    Inputs:
      model: sequential model dict with 'forward', 'backward', 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x: np.ndarray of shape (N, D) training features
      y: np.ndarray of shape (N,) integer class labels
      epochs: int, number of full passes over the data
      batch_size: int, minibatch size
      seed: int, RNG seed for deterministic shuffling / batching

    Returns:
      history: list[float] of length `epochs`; history[t] is the mean
      train_step loss over minibatches in epoch t.
      Model parameters are updated in place; shapes unchanged.
    """
    # TODO: your approach here
    history = []
    N, D = x.shape
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
      idx = np.arange(N)
      rng.shuffle(idx)

      x_shuffled, y_shuffled = x[idx], y[idx]

      loss = []
      for start in range(0, N, batch_size):
        end = start + batch_size

        x_batch = x_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        loss.append(train_step(model, loss_fn, optimizer, x_batch, y_batch))

      loss_mean = np.mean(loss)

      history.append(float(loss_mean))
    
    return history

# Step 12 - design_network
def design_network(input_dim, num_classes, seed=0):
    """Design and train a net that solves a nonlinear classification task.

    Inputs:
      input_dim: int, feature dimension
      num_classes: int, number of classes
      seed: int, RNG seed for reproducibility

    Returns:
      model: trained sequential model (forward/backward/params)
      metrics: dict with
        'accuracy': float >= 0.90 on an evaluation set,
        'x': np.ndarray (N, input_dim) eval features (N >= 50),
        'y': np.ndarray (N,) integer eval labels.
      The eval set (x, y) must not be linearly separable to high accuracy
      (< 0.82 for a linear classifier), and the model's true accuracy on
      it must match metrics['accuracy'] and be >= 0.90.
    """
    # TODO: your approach here
    rng = np.random.default_rng(seed)

    n = 200

    angles = rng.uniform(0, 2 * np.pi, n)

    r_inner = rng.normal(1.0, 0.1, n // 2)
    r_outer = rng.normal(2.0, 0.1, n // 2)

    x_inner = np.column_stack([
        r_inner * np.cos(angles[:n // 2]),
        r_inner * np.sin(angles[:n // 2])
    ])

    x_outer = np.column_stack([
        r_outer * np.cos(angles[n // 2:]),
        r_outer * np.sin(angles[n // 2:])
    ])

    x = np.vstack([x_inner, x_outer])
    if input_dim > 2:
      extra = np.zeros((n, input_dim - 2))
      x = np.hstack([x, extra])
    y = np.concatenate([
        np.zeros(n // 2, dtype=int),
        np.ones(n // 2, dtype=int)
    ])

    init = lambda i, o: initialize_weights(i, o, scheme='he')

    layer_1 = make_dense(input_dim, 8, init)
    layer_1_act = make_activation(kind='relu')
    layer_2 = make_dense(8, num_classes, init)

    layers = [layer_1, layer_1_act, layer_2]

    model= make_sequential(layers)

    loss_fn = make_loss(kind='cross_entropy')

    optimizer = make_optimizer(model['params'], lr=1e-2, kind='sgd')

    history = train(model, loss_fn, optimizer, x, y, 200, 16, seed=0)

    logits, _ = model['forward'](x)
    predictions = np.argmax(logits, axis=1)

    correct_pred = np.sum(predictions == y)
    accuracy = correct_pred/len(y)

    return model, {
          'accuracy': accuracy,
          'x': x,
          'y': y
      }

# Step 13 - improve_generalization
def improve_generalization(baseline_model_fn, x_train, y_train, x_val, y_val, seed=0):
    """Improve held-out accuracy over an unregularized baseline.

    Inputs:
      baseline_model_fn: zero-arg callable -> fresh untrained sequential model
        (dict with 'forward', 'backward', 'params') matching the data dims.
      x_train, y_train: training features (N, D) and int labels (N,).
      x_val, y_val: validation features (N_val, D) and int labels (N_val,).
      seed: int for deterministic training.

    Returns:
      dict with keys:
        'val_accuracy': float accuracy of the improved model on x_val/y_val
        'baseline_val_accuracy': float val accuracy of plain unregularized SGD
        'predictions': np.ndarray shape (N_val,) int preds from improved model
        'model': the trained improved model

    Required behavior:
      val_accuracy > baseline_val_accuracy
      predictions == argmax(model.forward(x_val), axis=1)
      val_accuracy == mean(predictions == y_val)
      predictions are non-constant (not a trivial single-class predictor)
    """
    # TODO: your approach here

    print("train:", x_train.shape, y_train.shape)
    print("val:", x_val.shape, y_val.shape)
    print("classes:", np.unique(y_train), np.unique(y_val))

    baseline_model = baseline_model_fn()

    loss_fn = make_loss(kind='cross_entropy')

    optimizer = make_optimizer(
        baseline_model['params'],
        lr=0.05,
        kind='sgd'
    )

    train(
        baseline_model,
        loss_fn,
        optimizer,
        x_train,
        y_train,
        epochs=70,
        batch_size=32,
        seed=seed
    )

    baseline_logits, _ = baseline_model['forward'](x_val)
    baseline_predictions = np.argmax(baseline_logits, axis=1)

    baseline_val_accuracy = float(
        np.mean(baseline_predictions == y_val)
    )

    
    print(
        "baseline train accuracy:",
        np.mean(
            np.argmax(baseline_model['forward'](x_train)[0], axis=1) == y_train
        )
    )

    rng = np.random.default_rng(seed)

    noise = rng.normal(0, 0.05, size=x_train.shape)
    x_aug = np.vstack([x_train, x_train + noise])
    y_aug = np.concatenate([y_train, y_train])

    best_accuracy = -1.0
    best_params = None

    improved_model = baseline_model_fn()

    optimizer = make_optimizer(
        improved_model['params'],
        lr=0.05,
        kind='sgd'
    )

    train(
        improved_model,
        loss_fn,
        optimizer,
        x_train,
        y_train,
        epochs=70,
        batch_size=1,
        seed=seed
    )

    val_logits, _ = improved_model['forward'](x_val)
    val_preds = np.argmax(val_logits, axis=1)
    current_accuracy = np.mean(val_preds == y_val)

    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy

    logits, _ = improved_model['forward'](x_val)

    predictions = np.argmax(logits, axis=1).astype(int)

    val_accuracy = float(np.mean(predictions == y_val))

    print("baseline:", baseline_val_accuracy)
    print("improved:", val_accuracy)
    print("unique preds:", np.unique(predictions))

    return {
        'val_accuracy': val_accuracy,
        'baseline_val_accuracy': baseline_val_accuracy,
        'predictions': predictions,
        'model': improved_model
      }

