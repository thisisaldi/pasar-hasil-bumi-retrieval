"""User tower (user id + cart) and product tower (id, category, shop, price bucket)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs


def str_embed(vocab, dim):
    v = list(pd.unique(vocab))
    return tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=v, mask_token=None),
        tf.keras.layers.Embedding(len(v) + 1, dim)])


def mlp(dim, hidden, drop):
    layers = [tf.keras.layers.Dense(hidden, activation="relu")]
    if drop:
        layers.append(tf.keras.layers.Dropout(drop))
    layers.append(tf.keras.layers.Dense(dim))
    return tf.keras.Sequential(layers)


class UserTower(tf.keras.Model):
    def __init__(self, users, dim, hidden, drop, use_cart=True, dat=False):
        super().__init__()
        self.use_cart, self.dat = use_cart, dat
        self.emb = str_embed(users["_id"], dim)
        self.cart = tf.keras.layers.Dense(dim, use_bias=False)  # sum of learned item vectors in cart
        self.aug = str_embed(users["_id"], dim) if dat else None  # DAT augmented vector (mimics the item side)
        self.mlp = mlp(dim, hidden, drop)

    def call(self, x):
        parts = [self.emb(x["user_id"])]
        if self.use_cart:
            parts.append(self.cart(x["cart"]))
        if self.dat:
            parts.append(self.aug(x["user_id"]))
        return self.mlp(tf.concat(parts, axis=1))


class ProductTower(tf.keras.Model):
    def __init__(self, products, prod_feat, dim, hidden, drop, dat=False):
        super().__init__()
        self.dat = dat
        self.pid = str_embed(products["_id"], dim)
        self.cat = str_embed(prod_feat["category"], dim // 2)
        self.shop = str_embed(prod_feat["shop"], dim // 2)
        self.price = tf.keras.layers.Embedding(int(prod_feat.price_bucket.max()) + 2, dim // 2)
        self.aug = str_embed(products["_id"], dim) if dat else None  # DAT augmented vector (mimics the user side)
        self.mlp = mlp(dim, hidden, drop)

    def call(self, x):
        parts = [self.pid(x["product_id"]), self.cat(x["category"]),
                 self.shop(x["shop"]), self.price(x["price_bucket"])]
        if self.dat:
            parts.append(self.aug(x["product_id"]))
        return self.mlp(tf.concat(parts, axis=1))


class TwoTower(tfrs.Model):
    def __init__(self, d, dim=32, hidden=64, drop=0.2, use_cart=True, logq=True, temperature=None,
                 dat=False, dat_gamma=1.0):
        super().__init__()
        self.logq, self.dat, self.dat_gamma = logq, dat, dat_gamma
        self.user_model = UserTower(d.users, dim, hidden, drop, use_cart, dat)
        self.product_model = ProductTower(d.products, d.prod_feat, dim, hidden, drop, dat)
        self.task = tfrs.tasks.Retrieval(temperature=temperature)
        self.cand_x = {k: tf.constant(d.prod_feat[k].to_numpy())
                       for k in ["product_id", "category", "shop", "price_bucket"]}

    def compute_loss(self, f, training=False):
        q = self.user_model(f, training=training)      # training flag: else dropout stays off
        c = self.product_model(f, training=training)
        loss = self.task(
            q, c,
            candidate_sampling_probability=f["prob"] if self.logq else None,
            candidate_ids=f["product_id"],
            compute_metrics=False)  # scored in train.py (needs cart masking)
        if self.dat and training:
            # Adaptive-Mimic: each augmented vector regresses onto the (stop-grad) opposite embedding
            # of the positive pair, so it carries "what the other side looks like" at inference time.
            mimic = lambda a, t: tf.reduce_mean(tf.square(a - tf.stop_gradient(t)))
            loss += self.dat_gamma * (mimic(self.user_model.aug(f["user_id"]), c)
                                      + mimic(self.product_model.aug(f["product_id"]), q))
        return loss

    def scores(self, df, cart) -> np.ndarray:
        """(n, P) dot-product scores; columns follow Data.pids order."""
        q = self.user_model({"user_id": tf.constant(df.user_id.values), "cart": tf.constant(cart)}).numpy()
        c = self.product_model(self.cand_x).numpy()
        return q @ c.T
