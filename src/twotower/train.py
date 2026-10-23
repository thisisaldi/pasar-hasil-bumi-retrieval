"""Train + evaluate. Usage: python -m twotower.train [--dim 64 --drop 0 ...]  (see --help)."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from . import data as D
from .metrics import KS, evaluate_baselines, hit_rate
from .model import TwoTower

REPORTS = D.ROOT / "reports"
ARTIFACTS = D.ROOT / "artifacts"


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tx-file", default="transactions.csv")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--no-logq", action="store_true", help="disable popularity (logQ) correction")
    p.add_argument("--cart", action="store_true", help="feed cart items to the user tower (overfits on this data)")
    p.add_argument("--dat", action="store_true", help="Dual Augmented Two-tower (mimic loss)")
    p.add_argument("--dat-gamma", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="run")
    p.add_argument("--save", action="store_true", help="save weights to artifacts/<tag>")
    return p.parse_args(argv)


class ValHit(tf.keras.callbacks.Callback):
    """Adds val_hit (mean cart-masked hit@1/3/5) to logs so EarlyStopping can monitor it."""
    def __init__(self, d): super().__init__(); self.d = d

    def on_epoch_end(self, epoch, logs=None):
        d = self.d
        h = hit_rate(self.model.scores(d.val, d.val_cart), d.targets(d.val), d.val_cart, True)
        logs["val_hit"] = float(np.mean([h["hit@1"], h["hit@3"], h["hit@5"]]))


def plot_curves(h, best, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(exist_ok=True)
    ep = np.arange(1, len(h["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, h["loss"], color="tab:blue")
    ax[0].set(title="Training loss (in-batch softmax)", xlabel="epoch", ylabel="loss")
    ax[1].plot(ep, h["val_hit"], color="tab:red")
    ax[1].axvline(best + 1, ls="--", color="gray", label=f"best epoch {best + 1} (restored)")
    ax[1].set(title="Validation hit (mean of hit@1/3/5, cart-masked)", xlabel="epoch", ylabel="hit rate")
    ax[1].legend()
    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    a = parse(argv)
    tf.keras.utils.set_random_seed(a.seed)
    d = D.load(tx_file=a.tx_file)
    print(f"train/val/test = {len(d.train)}/{len(d.val)}/{len(d.test)}")

    results = evaluate_baselines(d)

    model = TwoTower(d, dim=a.dim, hidden=a.hidden, drop=a.drop, use_cart=a.cart,
                     logq=not a.no_logq, temperature=a.temperature,
                     dat=a.dat, dat_gamma=a.dat_gamma)
    model.compile(optimizer=tf.keras.optimizers.Adam(a.lr))
    train_ds = D.to_tf_dataset(d, d.train, d.train_cart, shuffle=True, bs=a.batch)
    early = tf.keras.callbacks.EarlyStopping(monitor="val_hit", mode="max", patience=a.patience,
                                             restore_best_weights=True)
    t0 = time.time()
    hist = model.fit(train_ds, epochs=a.epochs, callbacks=[ValHit(d), early], verbose=0)
    best = int(np.argmax(hist.history["val_hit"]))
    print(f"stopped at epoch {len(hist.history['loss'])}, best epoch {best + 1}, "
          f"val_hit {max(hist.history['val_hit']):.4f}, {time.time() - t0:.0f}s")

    plot_curves(hist.history, best, REPORTS / f"{a.tag}_curves.png")

    S, y = model.scores(d.test, d.test_cart), d.targets(d.test)
    results["two_tower"] = hit_rate(S, y, d.test_cart, False)
    results["two_tower +cartmask"] = hit_rate(S, y, d.test_cart, True)

    table = pd.DataFrame(results).T.round(4)
    print(table)

    REPORTS.mkdir(exist_ok=True)
    row = {**vars(a), "best_epoch": best + 1, "val_hit": max(hist.history["val_hit"]),
           **{f"test_{k}": v for k, v in results["two_tower +cartmask"].items()}}
    runs = REPORTS / "runs.csv"
    new = pd.DataFrame([row])
    (pd.concat([pd.read_csv(runs), new]) if runs.exists() else new).to_csv(runs, index=False)  # survives added columns
    (REPORTS / f"{a.tag}.json").write_text(json.dumps(results, indent=1))
    if a.save:
        ARTIFACTS.mkdir(exist_ok=True)
        model.save_weights(str(ARTIFACTS / a.tag / "model"))
    return results


if __name__ == "__main__":
    main()
