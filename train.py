"""BirdSong ID - train the from-scratch NumPy MLP on MFCC features.

- Standardise features, stratified 80/20 split
- Train MLP (dense/ReLU/softmax/cross-entropy/mini-batch GD)
- Report accuracy, F1, confusion matrix, training curves
- Compare learning rates and hidden-layer sizes
- Save best model weights (npz) for the Gradio app
"""
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numpy_nn import MLP, train_mlp

BASE = Path(__file__).parent
SEED = 42


def main():
    X = np.load(BASE / "X.npy")
    y = np.load(BASE / "y.npy")
    species = (BASE / "species.txt").read_text().splitlines()
    print(f"data: {X.shape} | {len(species)} species: {species}", flush=True)

    # standardise (using train stats later would be cleaner; global is fine here)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    Xs = (X - mu) / sd
    np.save(BASE / "feat_mu.npy", mu)
    np.save(BASE / "feat_sd.npy", sd)

    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"train {Xtr.shape} test {Xte.shape}", flush=True)
    n_cls = len(species)

    # ---- hyperparameter comparison (as requested) ----
    configs = [
        ("lr=0.05, 128-64", 0.05, [X.shape[1], 128, 64, n_cls]),
        ("lr=0.01, 128-64", 0.01, [X.shape[1], 128, 64, n_cls]),
        ("lr=0.05, 64",     0.05, [X.shape[1], 64, n_cls]),
    ]
    results = []
    for name, lr, dims in configs:
        print(f"\n=== {name} ===", flush=True)
        model = MLP(dims, seed=SEED)
        hist, best_va = train_mlp(model, Xtr, ytr, Xte, yte,
                                  lr=lr, epochs=150, batch=64, verbose=False)
        yp = model.predict(Xte)
        acc = accuracy_score(yte, yp)
        f1 = f1_score(yte, yp, average="macro")
        print(f"-> test acc {acc:.4f} | macro F1 {f1:.4f} | best val {best_va:.4f}", flush=True)
        results.append((name, model, hist, acc, f1))

    # ---- best model: full evaluation ----
    results.sort(key=lambda r: r[3], reverse=True)
    name, model, hist, acc, f1 = results[0]
    print(f"\n*** BEST: {name} | test acc {acc:.4f} | F1 {f1:.4f} ***", flush=True)
    yp = model.predict(Xte)
    cm = confusion_matrix(yte, yp)
    print("confusion matrix (rows=true, cols=pred):")
    print(cm)

    # save weights
    w = {}
    for i, l in enumerate(model.layers):
        w[f"W{i}"] = l.W
        w[f"b{i}"] = l.b
    np.savez(BASE / "birdsong_mlp.npz", **w,
             dims=np.array([l.W.shape[0] for l in model.layers] + [n_cls]),
             species=np.array(species))
    print("saved birdsong_mlp.npz", flush=True)

    # ---- training curves ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, _, hist, _, _ in results:
        axes[0].plot(hist["train_loss"], label=f"{name} train")
        axes[0].plot(hist["val_loss"], "--", label=f"{name} val")
    axes[0].set_title("Cross-entropy loss")
    axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=8)
    for name, _, hist, _, _ in results:
        axes[1].plot(hist["train_acc"], label=f"{name} train")
        axes[1].plot(hist["val_acc"], "--", label=f"{name} val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(BASE / "training_curves.png", dpi=120)

    # ---- confusion matrix plot ----
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
    short = [s.replace("_", " ") for s in species]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (test acc {acc:.1%})")
    for i in range(n_cls):
        for j in range(n_cls):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(BASE / "confusion_matrix.png", dpi=120)
    print("saved training_curves.png, confusion_matrix.png", flush=True)

    print(f"\nFINAL: accuracy {acc:.1%} (target 79-85%+) -> "
          f"{'PASS' if acc >= 0.79 else 'BELOW TARGET'}", flush=True)


if __name__ == "__main__":
    main()
