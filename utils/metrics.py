
# utils/metrics.py
import numpy as np

def accuracy(logits, y):
    preds = (logits > 0).astype(np.int32)
    return (preds == y.astype(np.int32)).mean()

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)

def early_stop(history, patience=10, key='val_loss'):
    best = float('inf'); wait = 0; best_idx = 0
    for i, h in enumerate(history):
        if h[key] < best - 1e-8:
            best = h[key]; wait = 0; best_idx = i
        else:
            wait += 1
            if wait >= patience:
                return True, best_idx
    return False, best_idx
