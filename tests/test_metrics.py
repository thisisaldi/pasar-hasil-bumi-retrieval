import numpy as np

from twotower.data import cart_matrix
from twotower.metrics import hit_rate

import pandas as pd


def test_hit_rate_ties_are_not_hits():
    S = np.zeros((2, 4))
    assert hit_rate(S, np.array([0, 1]), np.zeros((2, 4)), False, ks=(1,))["hit@1"] == 0.0


def test_cart_mask_removes_cart_items():
    S = np.array([[5.0, 4.0, 3.0]])
    cart = np.array([[1.0, 0.0, 0.0]])
    y = np.array([1])
    assert hit_rate(S, y, cart, False, ks=(1,))["hit@1"] == 0.0
    assert hit_rate(S, y, cart, True, ks=(1,))["hit@1"] == 1.0


def test_cart_matrix_excludes_target():
    df = pd.DataFrame({"transaction_id": [1, 1, 2], "product_id": ["a", "b", "c"]})
    m = cart_matrix(df, {"a": 0, "b": 1, "c": 2})
    assert m.tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
