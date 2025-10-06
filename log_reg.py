# log_reg.py — CS 599 Problem 2: Multiclass Logistic Regression on Fashion-MNIST
# TensorFlow 2 eager, NO Keras models (dataset loader only), manual training loop.
# Clean, minimal, and robust (no indentation traps; no H/W name collisions).

import argparse, os, json, time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# ----- Constants (image dims kept distinct from weights variable) -----
NUM_CLASSES = 10
IMG_H, IMG_W = 28, 28
D = IMG_H * IMG_W

LABELS = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
          "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# ----- Data -----
def load_data(val_split=0.1, seed=0):
    # Only for downloading data; NO Keras models are used.
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
    x_tr = (x_tr.astype("float32")/255.0).reshape(-1, D)
    x_te = (x_te.astype("float32")/255.0).reshape(-1, D)

    # Validation split
    n = x_tr.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_split)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    x_val, y_val = x_tr[val_idx], y_tr[val_idx]
    x_tr, y_tr = x_tr[tr_idx], y_tr[tr_idx]
    return (x_tr, y_tr), (x_val, y_val), (x_te, y_te)

# ----- Model (manual) -----
def one_hot(y, c=NUM_CLASSES):
    return tf.one_hot(y, c)

def logits_fn(W, b, X):
    return tf.matmul(X, W) + b  # logits

def cross_entropy(y_true_oh, logits):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_true_oh, logits=logits)
    )

def accuracy(logits, y_true):
    y_pred = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(y_pred == tf.cast(y_true, tf.int64), tf.float32))

# ----- Train -----
def train(args):
    # Seeds
    import random
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)

    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = load_data(args.val_split, seed=args.seed)

    # Variables (D x C weights and C biases). Small init to avoid saturation.
    W = tf.Variable(tf.random.normal([D, NUM_CLASSES], stddev=0.01))
    b = tf.Variable(tf.zeros([NUM_CLASSES]))

    # Optimizer
    if args.opt == 'sgd':
        opt = tf.optimizers.SGD(args.lr)
    elif args.opt == 'momentum':
        opt = tf.optimizers.SGD(args.lr, momentum=0.9)
    elif args.opt == 'adam':
        opt = tf.optimizers.Adam(args.lr)
    else:
        raise ValueError("opt must be sgd|momentum|adam")

    # Datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).shuffle(10000, seed=args.seed).batch(args.batch)
    val_ds   = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(1024)
    test_ds  = tf.data.Dataset.from_tensor_slices((x_te, y_te)).batch(1024)

    history = []
    device = tf.config.list_logical_devices('GPU')
    device_name = device[0].name if device else 'CPU'
    print(f"Training on: {device_name}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train epoch
        tr_loss = tf.keras.metrics.Mean()
        tr_acc  = tf.keras.metrics.Mean()
        for xb, yb in train_ds:
            with tf.GradientTape() as tape:
                logits = logits_fn(W, b, xb)
                yb_oh = one_hot(yb)
                loss = cross_entropy(yb_oh, logits)
                if args.weight_decay > 0:
                    loss += args.weight_decay * tf.reduce_sum(tf.square(W))
            grads = tape.gradient(loss, [W, b])
            opt.apply_gradients(zip(grads, [W, b]))
            tr_loss.update_state(loss)
            tr_acc.update_state(accuracy(logits, yb))

        # Val epoch
        val_loss = tf.keras.metrics.Mean()
        val_acc  = tf.keras.metrics.Mean()
        for xb, yb in val_ds:
            logits = logits_fn(W, b, xb)
            yb_oh = one_hot(yb)
            loss = cross_entropy(yb_oh, logits)
            val_loss.update_state(loss)
            val_acc.update_state(accuracy(logits, yb))

        dt = time.time() - t0
        rec = dict(
            epoch=epoch,
            train_loss=float(tr_loss.result().numpy()),
            train_acc=float(tr_acc.result().numpy()),
            val_loss=float(val_loss.result().numpy()),
            val_acc=float(val_acc.result().numpy()),
            lr=float(opt.learning_rate.numpy()),
            sec_per_epoch=dt
        )
        history.append(rec)

        if epoch % max(1, args.print_every) == 0:
            print(f"[{epoch:03d}] "
                  f"tr_loss={rec['train_loss']:.4f} tr_acc={rec['train_acc']:.3f}  "
                  f"val_loss={rec['val_loss']:.4f} val_acc={rec['val_acc']:.3f}  "
                  f"lr={rec['lr']:.5f} t={dt:.2f}s")

    # Test
    y_pred_all, y_true_all = [], []
    for xb, yb in test_ds:
        logits = logits_fn(W, b, xb)
        y_pred_all.append(tf.argmax(logits, axis=1).numpy())
        y_true_all.append(yb.numpy())
    y_pred_all = np.concatenate(y_pred_all)
    y_true_all = np.concatenate(y_true_all)
    test_acc = (y_pred_all == y_true_all).mean()

    # Save logs
    os.makedirs("results", exist_ok=True); os.makedirs("figs", exist_ok=True)
    tag = f"fashionmnist_{args.opt}_b{args.batch}_lr{args.lr}_val{args.val_split}_seed{args.seed}"
    with open(f"results/{tag}.json", "w") as f:
        json.dump(dict(history=history, test_acc=float(test_acc)), f, indent=2)

    # ----- Plots: curves -----
    epochs = [h['epoch'] for h in history]

    plt.figure()
    plt.plot(epochs, [h['train_loss'] for h in history], label="train")
    plt.plot(epochs, [h['val_loss']   for h in history], label="val")
    plt.title("Cross-Entropy Loss"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.tight_layout(); plt.savefig(f"figs/{tag}_loss.png", dpi=150)

    plt.figure()
    plt.plot(epochs, [h['train_acc'] for h in history], label="train")
    plt.plot(epochs, [h['val_acc']   for h in history], label="val")
    plt.title("Accuracy"); plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend()
    plt.tight_layout(); plt.savefig(f"figs/{tag}_acc.png", dpi=150)

    plt.figure()
    plt.plot(epochs, [h['sec_per_epoch'] for h in history])
    plt.title("Timing (sec/epoch)"); plt.xlabel("epoch"); plt.ylabel("sec")
    plt.tight_layout(); plt.savefig(f"figs/{tag}_time.png", dpi=150)

    # ----- Visualize learned weights per class -----
    W_np = W.numpy()  # (D, C)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for c in range(NUM_CLASSES):
        ax = axes[c // 5, c % 5]
        ax.imshow(W_np[:, c].reshape(IMG_H, IMG_W), cmap='viridis')
        ax.axis('off'); ax.set_title(LABELS[c], fontsize=8)
    plt.suptitle("Learned weight images by class (logistic regression)")
    plt.tight_layout(); plt.savefig(f"figs/{tag}_weights.png", dpi=150)

    # ----- t-SNE + k-means on class weight vectors -----
    km = KMeans(n_clusters=NUM_CLASSES, n_init=10, random_state=args.seed)
    km.fit(W_np.T)  # shape (C, D)
    tsne = TSNE(n_components=2, random_state=args.seed, init='pca', perplexity=5)
    emb = tsne.fit_transform(W_np.T)
    plt.figure()
    plt.scatter(emb[:, 0], emb[:, 1], c=km.labels_)
    for i, lbl in enumerate(LABELS):
        plt.text(emb[i, 0], emb[i, 1], lbl, fontsize=8)
    plt.title("t-SNE of class weight vectors with k-means coloring")
    plt.tight_layout(); plt.savefig(f"figs/{tag}_tsne_kmeans.png", dpi=150)

    print(f"Test accuracy (logistic regression): {test_acc:.4f}")
    print("Done. Plots at figs/, logs at results/")

# ----- CLI -----
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--opt", type=str, default="adam", choices=["sgd","momentum","adam"])
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=808)
    p.add_argument("--print_every", type=int, default=1)
    args = p.parse_args()
    train(args)
