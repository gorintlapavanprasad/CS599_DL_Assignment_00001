
# utils/noise.py
import numpy as np
import tensorflow as tf
rng = np.random

def add_gaussian(x, std):
    if std <= 0: return x
    return x + rng.normal(0.0, std, size=x.shape).astype(x.dtype)

def add_uniform(x, low=-0.5, high=0.5, p=1.0):
    if p <= 0: return x
    mask = (rng.rand(*x.shape) < p).astype(x.dtype)
    return x + mask * rng.uniform(low, high, size=x.shape).astype(x.dtype)

def jitter_lr(lr, jitter=0.1):
    """Randomly jitter LR by ±jitter fraction per epoch."""
    if jitter <= 0: return lr
    return float(lr) * float(1.0 + rng.uniform(-jitter, jitter))

def noisy_weights(var, std=0.0, p=0.0, uniform=False):
    if std <= 0 and p <= 0: return var
    w = var.numpy()
    if uniform:
        w = add_uniform(w, -std, std, p=1.0 if p<=0 else p)
    else:
        w = add_gaussian(w, std)
    var.assign(w)

def set_seed(seed: int):
    import random, os
    import numpy as np
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
