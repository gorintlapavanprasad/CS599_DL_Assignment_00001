# CS 599: Foundations of Deep Learning — Assignment #00001

This repo contains **clean, simple, and well‑commented** TensorFlow 2 (eager) code (no Keras `Model`/`Layer`) for:

- **Problem 1:** Linear Regression with GradientTape; experiments with loss functions, noise, LR scheduling, seeds, and timing (CPU vs GPU).
- **Problem 2:** Logistic Regression on **Fashion‑MNIST** with a manual training loop (no Keras), t‑SNE/k‑means on weights, optimizer ablations, batch size study, overfitting checks, and timing.

> **Note:** Only the **dataset loader** uses `tf.keras.datasets` for Fashion‑MNIST. All modeling/training is done with raw `tf.Variable`, `tf.GradientTape`, and functional ops — **no Keras layers or fit()**.

## Quick start

```bash
python lin_reg.py --epochs 200 --loss l2 --lr 0.05 --noise_std 0.5 --seed 808  # linear regression toy
python log_reg.py --epochs 30 --batch 256 --opt adam --lr 1e-3 --val_split 0.1 --seed 808
```

- Results (plots + JSON logs) are written to `results/` and `figs/`.
- For patience scheduling (halve LR on plateau): `--patience 5 --plateau_tol 1e-5`.

## Files

- `lin_reg.py` — Problem 1 implementation + experiment switches
- `log_reg.py` — Problem 2 implementation + analyses
- `utils/schedules.py` — Learning‑rate schedulers (plateau, cosine, warmup)
- `utils/noise.py` — Reusable noise helpers (data, weights, LR jitter)
- `utils/metrics.py` — Accuracy, confusion matrix, and simple early‑stop
- `report.tex` — **NeurIPS**-style report template with TODO markers to paste your GitHub link and results
- `requirements.txt` — minimal dependencies
- `README.md` — this file

Remember to push to **GitHub** and paste the link into `report.tex` before compiling the PDF.

Good luck, and have fun!
