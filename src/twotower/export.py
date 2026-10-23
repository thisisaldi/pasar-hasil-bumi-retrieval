"""Export a trained run to a TensorFlow-free serving bundle (serving/bundle.json).

    python -m twotower.export --tag final [--data-dir dataset --weights artifacts/final/model]

The saved `final` run has no cart input (cart=False), so a user's embedding depends only on the
user id. Every user and product embedding is precomputed here, so the API is a dot product in
numpy. Test hit@k is recomputed from the exported vectors and checked against reports/runs.csv.

Raw DB ids (users, shops, categories, products) are replaced by stable pseudonyms so the bundle
can be published; product names and prices are kept.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .metrics import evaluate_baselines, hit_rate

OUT = D.ROOT / "serving" / "bundle.json"


def run_config(tag: str) -> dict:
    runs = pd.read_csv(D.ROOT / "reports" / "runs.csv")
    row = runs[runs.tag == tag]
    if row.empty:
        raise SystemExit(f"tag {tag!r} not in reports/runs.csv")
    return row.iloc[-1].to_dict()


def pseudonyms(values, prefix: str) -> dict:
    uniq = list(pd.unique(pd.Series(values)))
    width = len(str(len(uniq)))
    return {v: f"{prefix}{i + 1:0{width}d}" for i, v in enumerate(uniq)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="final")
    p.add_argument("--data-dir", type=Path, default=D.DEFAULT_DATA_DIR)
    p.add_argument("--weights", type=Path, default=None, help="default: artifacts/<tag>/model")
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args(argv)

    import tensorflow as tf
    from .model import TwoTower

    cfg = run_config(a.tag)
    if cfg["cart"]:
        raise SystemExit("cart-aware runs need the TF user tower at request time; export supports cart=False only")
    tf.keras.utils.set_random_seed(int(cfg["seed"]))
    d = D.load(a.data_dir, cfg["tx_file"])
    model = TwoTower(d, dim=int(cfg["dim"]), hidden=int(cfg["hidden"]), drop=float(cfg["drop"]),
                     use_cart=False, logq=not cfg["no_logq"], dat=bool(cfg["dat"]),
                     dat_gamma=float(cfg["dat_gamma"]))
    model.scores(d.test.head(1), d.test_cart[:1])  # build variables before restoring
    weights = a.weights or D.ROOT / "artifacts" / a.tag / "model"
    model.load_weights(str(weights)).expect_partial()

    # embeddings for every user / product (cart is ignored when cart=False)
    users = d.users["_id"].to_numpy()
    U = model.user_model({"user_id": tf.constant(users),
                          "cart": tf.zeros((len(users), len(d.pids)))}).numpy()
    P = model.product_model(model.cand_x).numpy()
    U, P = U.round(6), P.round(6)

    # 1) restored weights reproduce the logged run, 2) exported vectors reproduce the same metrics
    y = d.targets(d.test)
    tf_hit = hit_rate(model.scores(d.test, d.test_cart), y, d.test_cart, True)
    np_hit = hit_rate(U[d.user_ids(d.test)] @ P.T, y, d.test_cart, True)
    logged = {k: cfg[f"test_{k}"] for k in tf_hit}
    for name, h in (("restored TF model", tf_hit), ("exported vectors", np_hit)):
        if not all(abs(h[k] - logged[k]) < 1e-6 for k in h):
            raise SystemExit(f"{name} does not reproduce runs.csv: {h} vs {logged}")
    print("verified test hit@k:", {k: round(v, 4) for k, v in np_hit.items()})

    uname = pseudonyms(users, "u")
    pname = pseudonyms(d.pids, "p")
    sname = pseudonyms(d.prod_feat["shop"], "shop")
    cname = pseudonyms(d.prod_feat["category"], "cat")

    C = np.zeros((len(users), len(d.pids)), dtype=int)
    np.add.at(C, (d.user_ids(d.train), d.targets(d.train)), 1)
    prods = d.products.set_index("_id")
    catalog = [{"id": pname[pid], "name": prods.at[pid, "name"], "price": int(prods.at[pid, "price"]),
                "category": cname[f.category], "shop": sname[f.shop], "train_count": int(C[:, i].sum())}
               for i, (pid, f) in enumerate(zip(d.pids, d.prod_feat.itertuples()))]

    # a few held-out test baskets: the demo hides one item and asks the model to find it
    examples = []
    for _, g in d.test.groupby("transaction_id", sort=True):
        if len(g) < 2 or uname[g.user_id.iloc[0]] in {e["user_id"] for e in examples}:
            continue
        examples.append({"user_id": uname[g.user_id.iloc[0]],
                         "cart": [pname[x] for x in g.product_id.iloc[1:]],
                         "held_out": pname[g.product_id.iloc[0]]})
        if len(examples) == 8:
            break

    base = evaluate_baselines(d)
    results = {"two_tower": np_hit, **{k.replace(" +cartmask", ""): v for k, v in base.items() if "+cartmask" in k}}
    bundle = {
        "model": {
            "tag": a.tag, "architecture": "two-tower (TFRS), in-batch softmax with logQ correction",
            "cart_aware": False,
            "config": {k: cfg[k] for k in ("dim", "hidden", "drop", "lr", "batch", "dat", "seed", "best_epoch")},
            "user_features": ["user_id"],
            "product_features": ["product_id", "category", "shop", "price_bucket"],
        },
        "evaluation": {
            "protocol": "time split 70/10/20 by transaction timestamp; each test row = (user, basket minus one "
                        "item) -> rank the held-out item among all products; items already in the basket are "
                        "masked; ties count as misses",
            "split_rows": {"train": len(d.train), "val": len(d.val), "test": len(d.test)},
            "metric": "hit@k (cart-masked)",
            "results": results,
            "data_note": "transactions are synthetic (80% of picks drawn from 3-6 per-user favourites); "
                         "users and catalogue come from the marketplace database",
        },
        "products": catalog,
        "users": [{"id": uname[u], "train_interactions": int(C[i].sum())} for i, u in enumerate(users)],
        "user_embeddings": U.tolist(),
        "product_embeddings": P.tolist(),
        "user_item_counts": C.tolist(),
        "examples": examples,
    }
    a.out.parent.mkdir(exist_ok=True)
    a.out.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {a.out} ({a.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
