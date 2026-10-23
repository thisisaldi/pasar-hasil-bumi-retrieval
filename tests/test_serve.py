import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from twotower import serve


@pytest.fixture
def tiny(tmp_path):
    """2 users, 4 products. u1's embedding points at p3 > p2 > p1 > p4."""
    b = {
        "model": {"tag": "tiny"}, "evaluation": {}, "examples": [{"user_id": "u1", "cart": ["p3"], "held_out": "p2"}],
        "products": [{"id": f"p{i}", "name": f"item {i}", "price": 1000 * i, "category": "c", "shop": "s",
                      "train_count": 0} for i in range(1, 5)],
        "users": [{"id": "u1", "train_interactions": 3}, {"id": "u2", "train_interactions": 1}],
        "user_embeddings": [[1.0, 0.0], [0.0, 1.0]],
        "product_embeddings": [[0.2, 0.0], [0.5, 0.0], [0.9, 0.0], [-1.0, 0.0]],
        "user_item_counts": [[0, 0, 0, 3], [1, 0, 0, 0]],
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(b))
    return path


@pytest.fixture
def client(tiny, monkeypatch):
    monkeypatch.setenv("BUNDLE_PATH", str(tiny))
    with TestClient(serve.app) as c:
        yield c


def test_top_k_orders_by_score_and_masks_cart(tiny):
    b = serve.Bundle(tiny)
    assert [i for i, _ in b.top_k(0, [], 4)] == [2, 1, 0, 3]
    assert [i for i, _ in b.top_k(0, [2], 2)] == [1, 0]
    assert [i for i, _ in b.top_k(0, [], 1, "personal_frequency")] == [3]
    assert len(b.top_k(0, [0, 1, 2], 4)) == 1  # never pads with masked items


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model": "tiny", "users": 2, "products": 4}


def test_recommend_with_baseline(client):
    r = client.post("/recommend", json={"user_id": "u1", "cart": ["p3", "p3"], "k": 2, "baseline": "personal_frequency"})
    assert r.status_code == 200
    body = r.json()
    assert body["cart"] == ["p3"]
    assert [x["product"]["id"] for x in body["items"]] == ["p2", "p1"]
    assert body["items"][0] == {"rank": 1, "score": 0.5, "product": body["items"][0]["product"]}
    assert body["baseline"]["items"][0]["product"]["id"] == "p4"


@pytest.mark.parametrize("payload, code", [
    ({"user_id": "nobody"}, 404),
    ({"user_id": "u1", "cart": ["p9"]}, 422),
    ({"user_id": "u1", "k": 0}, 422),
    ({"user_id": "u1", "k": 46}, 422),
    ({"user_id": "u1", "baseline": "magic"}, 422),
    ({"cart": []}, 422),
])
def test_recommend_rejects_bad_input(client, payload, code):
    assert client.post("/recommend", json=payload).status_code == code


@pytest.mark.parametrize("origin, allowed", [
    ("https://nfaldi.dev", True), ("https://naufaldi.vercel.app", True), ("https://evil.example", False)])
def test_cors_default_origins(client, origin, allowed):
    r = client.options("/recommend", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
    assert (r.headers.get("access-control-allow-origin") == origin) == allowed


def test_published_bundle_is_consistent():
    if not serve.DEFAULT_BUNDLE.exists():
        pytest.skip("serving/bundle.json not exported")
    b = serve.Bundle(serve.DEFAULT_BUNDLE)
    assert b.U.shape == (len(b.users), b.P.shape[1]) and b.P.shape[0] == len(b.products)
    assert b.counts.shape == (len(b.users), len(b.products))
    assert all(e["user_id"] in b.uidx and {e["held_out"], *e["cart"]} <= b.pidx.keys() for e in b.examples)
    assert np.isfinite(b.U).all() and np.isfinite(b.P).all()
