"""Retrieval API over the exported bundle (numpy only, no TensorFlow).

    uvicorn twotower.serve:app --app-dir src        # BUNDLE_PATH, CORS_ORIGINS env vars

The bundle is loaded once at startup; requests only do a (1 x 32) @ (32 x 45) product and a sort.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DEFAULT_BUNDLE = Path(__file__).resolve().parents[2] / "serving" / "bundle.json"
Baseline = Literal["personal_frequency", "popularity"]


class Bundle:
    def __init__(self, path: Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.model, self.evaluation, self.examples = raw["model"], raw["evaluation"], raw["examples"]
        self.products, self.users = raw["products"], raw["users"]
        self.U = np.asarray(raw["user_embeddings"], dtype=np.float32)
        self.P = np.asarray(raw["product_embeddings"], dtype=np.float32)
        self.counts = np.asarray(raw["user_item_counts"], dtype=np.float64)
        self.uidx = {u["id"]: i for i, u in enumerate(self.users)}
        self.pidx = {p["id"]: i for i, p in enumerate(self.products)}
        pop = self.counts.sum(0)
        self.tiebreak = pop / max(pop.sum(), 1) * 1e-3  # baselines break ties by popularity

    def scores(self, user: int, method: str) -> np.ndarray:
        if method == "two_tower":
            return self.P @ self.U[user]
        if method == "personal_frequency":
            return self.counts[user]
        return self.counts.sum(0)  # popularity

    def top_k(self, user: int, cart: list[int], k: int, method: str = "two_tower") -> list[tuple[int, float]]:
        s = self.scores(user, method).astype(np.float64)
        s[cart] = -np.inf  # a basket never repeats a product
        rank_by = s if method == "two_tower" else s + self.tiebreak
        order = np.argsort(-rank_by, kind="stable")[:k]
        return [(int(i), float(s[i])) for i in order if np.isfinite(s[i])]


# ---------- schemas
class Product(BaseModel):
    id: str
    name: str
    price: int = Field(description="IDR")
    category: str
    shop: str
    train_count: int = Field(description="times bought in the training split")


class User(BaseModel):
    id: str
    train_interactions: int


class RecommendRequest(BaseModel):
    user_id: str = Field(examples=["u39"])
    cart: list[str] = Field(default=[], max_length=45, description="product ids already in the basket; excluded from results",
                            examples=[["p32", "p31", "p40"]])
    k: int = Field(default=10, ge=1, le=45)
    baseline: Baseline | None = Field(default=None, description="also return this baseline's top-k for comparison")


class Recommendation(BaseModel):
    rank: int
    score: float = Field(description="dot product for two_tower; training purchase count for baselines")
    product: Product


class BaselineResult(BaseModel):
    name: Baseline
    items: list[Recommendation]


class RecommendResponse(BaseModel):
    user_id: str
    cart: list[str]
    k: int
    items: list[Recommendation]
    baseline: BaselineResult | None = None


class Health(BaseModel):
    status: Literal["ok"]
    model: str
    users: int
    products: int


class Example(BaseModel):
    user_id: str
    cart: list[str]
    held_out: str = Field(description="the test-set item removed from this basket")


class Error(BaseModel):
    detail: str


# ---------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = Bundle(os.environ.get("BUNDLE_PATH", DEFAULT_BUNDLE))  # fail fast if missing
    yield


app = FastAPI(title="Pasar Hasil Bumi retrieval API", version="0.1.0", lifespan=lifespan,
              description="Two-tower product retrieval for an agricultural marketplace (undergraduate thesis project).")
app.add_middleware(CORSMiddleware, allow_methods=["GET", "POST"], allow_headers=["content-type"],
                   allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()])


def bundle(request: Request) -> Bundle:
    return request.app.state.bundle


@app.get("/health", response_model=Health)
def health(request: Request):
    b = bundle(request)
    return Health(status="ok", model=b.model["tag"], users=len(b.users), products=len(b.products))


@app.get("/model")
def model_info(request: Request) -> dict:
    """Model config and the offline test results the bundle was verified against."""
    b = bundle(request)
    return {"model": b.model, "evaluation": b.evaluation}


@app.get("/products", response_model=list[Product])
def products(request: Request):
    return bundle(request).products


@app.get("/users", response_model=list[User])
def users(request: Request):
    """Known (pseudonymised) users. The model has no cold-start path: unknown users get 404."""
    return bundle(request).users


@app.get("/examples", response_model=list[Example])
def examples(request: Request):
    """Held-out test baskets, usable as realistic demo inputs."""
    return bundle(request).examples


@app.post("/recommend", response_model=RecommendResponse,
          responses={404: {"model": Error, "description": "unknown user_id"},
                     422: {"description": "invalid body or unknown product id in cart"}})
def recommend(req: RecommendRequest, request: Request):
    b = bundle(request)
    if req.user_id not in b.uidx:
        raise HTTPException(404, f"unknown user_id {req.user_id!r}")
    unknown = [p for p in req.cart if p not in b.pidx]
    if unknown:
        raise HTTPException(422, f"unknown product ids in cart: {unknown}")
    u, cart = b.uidx[req.user_id], [b.pidx[p] for p in dict.fromkeys(req.cart)]

    def recs(method):
        return [Recommendation(rank=r + 1, score=round(s, 6), product=b.products[i])
                for r, (i, s) in enumerate(b.top_k(u, cart, req.k, method))]

    return RecommendResponse(user_id=req.user_id, cart=list(dict.fromkeys(req.cart)), k=req.k,
                             items=recs("two_tower"),
                             baseline=BaselineResult(name=req.baseline, items=recs(req.baseline)) if req.baseline else None)
