"""Full-catalog hit-rate@k and non-learned baselines (no TensorFlow needed)."""
from __future__ import annotations

import numpy as np

KS = (1, 3, 5, 10)


def hit_rate(S: np.ndarray, y: np.ndarray, cart: np.ndarray, mask: bool, ks=KS) -> dict:
    """S: (n, P) scores, y: (n,) target index, cart: (n, P) multi-hot. Ties are NOT counted as hits."""
    S = S.copy()
    if mask:
        S[cart > 0] = -1e9  # a basket never repeats a product
    rank = (S >= S[np.arange(len(y)), y][:, None]).sum(1) - 1  # pessimistic on ties
    return {f"hit@{k}": float((rank < k).mean()) for k in ks}


def baseline_scores(train_u: np.ndarray, train_y: np.ndarray, n_users: int, n_items: int,
                    test_u: np.ndarray, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    C = np.zeros((n_users, n_items))
    np.add.at(C, (train_u, train_y), 1)
    pop = C.sum(0)
    n = len(test_u)
    noise = lambda: rng.random((n, n_items)) * 1e-9  # random tie-break
    return {
        "random": rng.random((n, n_items)),
        "popularity": np.tile(pop, (n, 1)) + noise(),
        "personal_freq": C[test_u] + pop / pop.sum() * 1e-3 + noise(),
    }


def evaluate_baselines(d) -> dict:
    scores = baseline_scores(d.user_ids(d.train), d.targets(d.train), len(d.uidx), len(d.pidx),
                             d.user_ids(d.test))
    y = d.targets(d.test)
    out = {}
    for name, S in scores.items():
        out[name] = hit_rate(S, y, d.test_cart, False)
        out[name + " +cartmask"] = hit_rate(S, y, d.test_cart, True)
    return out
