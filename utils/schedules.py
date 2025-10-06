
# utils/schedules.py
# Small, human-readable LR schedulers (no Keras).
import math

class PlateauHalver:
    """Halve LR if metric hasn't improved by `tol` in `patience` epochs."""
    def __init__(self, lr, patience=5, tol=1e-6, min_lr=1e-6):
        self.lr = float(lr)
        self.best = None
        self.wait = 0
        self.patience = int(patience)
        self.tol = float(tol)
        self.min_lr = float(min_lr)

    def step(self, metric):
        improved = (self.best is None) or (self.best - metric > self.tol)
        if improved:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.lr = max(self.lr * 0.5, self.min_lr)
                self.wait = 0
        return self.lr

class WarmupThenDecay:
    """Linear warmup to lr over `warmup` epochs, then cosine decay to `min_lr` over `total` epochs."""
    def __init__(self, lr, total, warmup=5, min_lr=1e-6):
        self.base_lr = float(lr)
        self.total = int(total)
        self.warmup = int(warmup)
        self.min_lr = float(min_lr)
        self.epoch = 0

    def step(self, metric=None):
        if self.epoch < self.warmup:
            t = (self.epoch + 1) / max(1, self.warmup)
            lr = self.base_lr * t
        else:
            t = (self.epoch - self.warmup) / max(1, self.total - self.warmup)
            cos = 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
            lr = self.min_lr + (self.base_lr - self.min_lr) * cos
        self.epoch += 1
        return lr
