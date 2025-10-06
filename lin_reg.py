
# lin_reg.py
# CS 599 — Problem 1: Linear Regression with tf.GradientTape (no Keras)
# Author: Pavan Prasad Gorintla (fill your details)
#
# This script trains a linear model on synthetic data y = 3x + 2 + noise
# and explores: L1/L2/Huber/hybrid losses, LR scheduling with patience,
# noise in data/weights/LR, different inits, and timing (CPU vs GPU).
#
# Run:
#   python lin_reg.py --epochs 200 --loss l2 --lr 0.05 --noise_std 0.5 --seed 808
#
import argparse, json, time, os
import numpy as np
import tensorflow as tf

from utils.schedules import PlateauHalver, WarmupThenDecay
from utils.noise import add_gaussian, add_uniform, jitter_lr, noisy_weights, set_seed

def make_data(n=10_000, noise_std=0.5, noise_type='gaussian', seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-5, 5, size=(n, 1)).astype('float32')
    y_clean = 3.0 * x + 2.0
    if noise_std > 0:
        if noise_type == 'gaussian':
            y = y_clean + rng.normal(0, noise_std, size=y_clean.shape).astype('float32')
        elif noise_type == 'uniform':
            y = y_clean + rng.uniform(-noise_std, noise_std, size=y_clean.shape).astype('float32')
        elif noise_type == 'laplace':
            y = y_clean + rng.laplace(0, noise_std, size=y_clean.shape).astype('float32')
        else:
            y = y_clean
    else:
        y = y_clean
    return x, y, y_clean

def loss_fn(y, y_hat, kind='l2', huber_delta=1.0, alpha=0.5, lam=0.0, W=None, b=None):
    # kind: 'l2' (MSE), 'l1' (MAE), 'huber', 'hybrid' (alpha*L1 + (1-alpha)*L2)
    if kind == 'l2':
        loss = tf.reduce_mean(tf.square(y - y_hat))
    elif kind == 'l1':
        loss = tf.reduce_mean(tf.abs(y - y_hat))
    elif kind == 'huber':
        err = y - y_hat
        abs_err = tf.abs(err)
        quad = tf.minimum(abs_err, huber_delta)
        lin = abs_err - quad
        loss = tf.reduce_mean(0.5 * tf.square(quad) + huber_delta * lin)
    elif kind == 'hybrid':
        l1 = tf.reduce_mean(tf.abs(y - y_hat))
        l2 = tf.reduce_mean(tf.square(y - y_hat))
        loss = alpha * l1 + (1 - alpha) * l2
    else:
        raise ValueError("Unknown loss kind")
    # Optional L2 weight decay (elastic net could be added similarly)
    if lam > 0.0 and W is not None and b is not None:
        loss += lam * (tf.reduce_sum(tf.square(W)) + tf.reduce_sum(tf.square(b)))
    return loss

def train(args):
    set_seed(args.seed)
    x, y, y_clean = make_data(args.n, args.noise_std, args.noise_type, seed=args.seed)

    # Variables with configurable init
    W = tf.Variable(tf.constant(args.init_W, dtype=tf.float32, shape=(1,1)))
    b = tf.Variable(tf.constant(args.init_b, dtype=tf.float32, shape=(1,)))

    # Optimizer: simple SGD, Momentum, or Adam (raw TF API)
    if args.opt == 'sgd':
        opt = tf.optimizers.SGD(args.lr)
    elif args.opt == 'momentum':
        opt = tf.optimizers.SGD(args.lr, momentum=0.9)
    elif args.opt == 'adam':
        opt = tf.optimizers.Adam(args.lr)
    else:
        raise ValueError("opt must be sgd|momentum|adam")

    # Optional schedulers
    plateau = PlateauHalver(args.lr, patience=args.patience, tol=args.plateau_tol) if args.patience>0 else None
    warmcos = WarmupThenDecay(args.lr, total=args.epochs, warmup=args.warmup) if args.warmup>0 else None

    history = []
    device = tf.config.list_logical_devices('GPU')
    device_name = device[0].name if device else 'CPU'
    print(f"Training on: {device_name}")

    for epoch in range(1, args.epochs+1):
        t0 = time.time()

        # Optional per-epoch LR jitter
        if args.lr_jitter>0 and isinstance(opt, (tf.optimizers.SGD, tf.optimizers.Adam)):
            lr_now = jitter_lr(opt.learning_rate.numpy(), args.lr_jitter)
            opt.learning_rate.assign(lr_now)

        # Optional noisy weights
        if args.weight_noise_std>0 or args.weight_noise_p>0:
            noisy_weights(W, std=args.weight_noise_std, p=args.weight_noise_p, uniform=args.weight_noise_uniform)
            noisy_weights(b, std=args.weight_noise_std, p=args.weight_noise_p, uniform=args.weight_noise_uniform)

        with tf.GradientTape() as tape:
            y_hat = tf.matmul(x, W) + b  # linear model
            loss = loss_fn(y, y_hat, kind=args.loss, huber_delta=args.huber_delta,
                           alpha=args.alpha, lam=args.weight_decay, W=W, b=b)

        grads = tape.gradient(loss, [W, b])
        opt.apply_gradients(zip(grads, [W, b]))

        # Scheduler step (use current loss as metric)
        if warmcos:
            new_lr = warmcos.step()
            opt.learning_rate.assign(new_lr)
        if plateau:
            new_lr = plateau.step(float(loss.numpy()))
            opt.learning_rate.assign(new_lr)

        dt = time.time() - t0
        rec = dict(epoch=epoch, loss=float(loss.numpy()), W=float(W.numpy().squeeze()),
                   b=float(b.numpy().squeeze()), lr=float(opt.learning_rate.numpy()),
                   sec_per_epoch=dt)
        history.append(rec)
        if epoch % max(1, args.print_every) == 0:
            print(f"[{epoch:04d}] loss={rec['loss']:.6f}  W={rec['W']:.4f}  b={rec['b']:.4f}  lr={rec['lr']:.6f}  t={dt:.4f}s")

    # Save artifacts
    os.makedirs("results", exist_ok=True); os.makedirs("figs", exist_ok=True)
    tag = f"linreg_{args.loss}_lr{args.lr}_noise{args.noise_std}_seed{args.seed}"
    with open(f"results/{tag}.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot
    import matplotlib.pyplot as plt
    # scatter + learned line
    xs = np.linspace(x.min(), x.max(), 100).astype('float32').reshape(-1,1)
    ys_hat = (xs @ W.numpy()) + b.numpy()
    plt.figure()
    plt.scatter(x, y, s=5, alpha=0.3, label="noisy data")
    plt.plot(xs, 3*xs+2, label="true f(x)=3x+2")
    plt.plot(xs, ys_hat, label=f"learned y=W x + b (W={float(W.numpy()):.3f}, b={float(b.numpy()):.3f})")
    plt.title(f"Linear Regression — loss={args.loss.upper()}, noise={args.noise_std}")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/{tag}_fit.png", dpi=150)

    # loss curve
    plt.figure()
    plt.plot([h['epoch'] for h in history], [h['loss'] for h in history])
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(f"figs/{tag}_loss.png", dpi=150)

    # timing curve
    plt.figure()
    plt.plot([h['epoch'] for h in history], [h['sec_per_epoch'] for h in history])
    plt.xlabel("epoch"); plt.ylabel("sec/epoch"); plt.title("Timing (approx.)")
    plt.tight_layout()
    plt.savefig(f"figs/{tag}_time.png", dpi=150)

    print("Done. Plots at figs/, logs at results/")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--loss", type=str, default="l2", choices=["l1","l2","huber","hybrid"])
    p.add_argument("--alpha", type=float, default=0.5, help="for hybrid loss: alpha*L1 + (1-alpha)*L2")
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--opt", type=str, default="sgd", choices=["sgd","momentum","adam"])
    p.add_argument("--noise_std", type=float, default=0.5)
    p.add_argument("--noise_type", type=str, default="gaussian", choices=["gaussian","uniform","laplace"])
    p.add_argument("--weight_noise_std", type=float, default=0.0)
    p.add_argument("--weight_noise_p", type=float, default=0.0)
    p.add_argument("--weight_noise_uniform", action="store_true")
    p.add_argument("--lr_jitter", type=float, default=0.0, help="±fractional jitter per epoch")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--init_W", type=float, default=0.0)
    p.add_argument("--init_b", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--plateau_tol", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--seed", type=int, default=808, help="Use your first name converted to decimal")
    p.add_argument("--print_every", type=int, default=10)
    args = p.parse_args()
    train(args)
