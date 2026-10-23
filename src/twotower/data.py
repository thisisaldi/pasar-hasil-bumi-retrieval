"""Load transactions, split by time, build cart features."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "dataset"


@dataclass
class Data:
    users: pd.DataFrame
    products: pd.DataFrame
    prod_feat: pd.DataFrame  # product_id, category, shop, price_bucket, prob (train freq)
    pids: list
    uidx: dict
    pidx: dict
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_cart: np.ndarray
    val_cart: np.ndarray
    test_cart: np.ndarray

    def user_ids(self, df): return df.user_id.map(self.uidx).to_numpy()
    def targets(self, df): return df.product_id.map(self.pidx).to_numpy()


def cart_matrix(df: pd.DataFrame, pidx: dict) -> np.ndarray:
    """Row i = multi-hot of the OTHER products in row i's transaction (target excluded)."""
    pi = df.product_id.map(pidx).to_numpy()
    m = np.zeros((len(df), len(pidx)), dtype="float32")
    for ix in df.groupby("transaction_id").indices.values():
        for i in ix:
            m[i, pi[ix]] = 1
            m[i, pi[i]] = 0
    return m


def load(data_dir: Path = DEFAULT_DATA_DIR, tx_file: str = "transactions.csv") -> Data:
    users = pd.read_csv(data_dir / "prod.users.csv")
    products = pd.read_csv(data_dir / "prod.products.csv")
    tx = pd.read_csv(data_dir / tx_file)
    products["price_bucket"] = pd.qcut(
        np.log1p(products["price"]), 8, labels=False, duplicates="drop").astype(int)

    # time split 70/10/20; rows of one transaction share a timestamp, so baskets never straddle
    t70, t80 = np.percentile(tx["timestamp"], [70, 80])
    train = tx[tx.timestamp <= t70].reset_index(drop=True)
    val = tx[(tx.timestamp > t70) & (tx.timestamp <= t80)].reset_index(drop=True)
    test = tx[tx.timestamp > t80].reset_index(drop=True)

    pids = products["_id"].tolist()
    pidx = {p: i for i, p in enumerate(pids)}
    uidx = {u: i for i, u in enumerate(users["_id"])}

    freq = train.product_id.value_counts(normalize=True)
    prod_feat = products.rename(columns={"_id": "product_id", "categories[0]": "category"})[
        ["product_id", "category", "shop", "price_bucket"]].copy()
    prod_feat["prob"] = prod_feat.product_id.map(freq).fillna(1e-6).astype("float32")

    return Data(users, products, prod_feat, pids, uidx, pidx, train, val, test,
                *(cart_matrix(d, pidx) for d in (train, val, test)))


def to_tf_dataset(d: Data, df: pd.DataFrame, cart: np.ndarray, shuffle: bool = False, bs: int = 256):
    import tensorflow as tf
    m = df[["user_id", "product_id"]].merge(d.prod_feat, on="product_id", how="left")
    feats = {c: m[c].to_numpy() for c in m.columns}
    feats["prob"] = feats["prob"].astype("float32")
    feats["cart"] = cart
    ds = tf.data.Dataset.from_tensor_slices(feats)
    if shuffle:
        ds = ds.shuffle(len(m), seed=42, reshuffle_each_iteration=True)
    return ds.batch(bs).cache()
