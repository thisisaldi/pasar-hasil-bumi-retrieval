# Deployment and portfolio integration

## Runtime profile (measured locally)
| | |
|---|---|
| Runtime deps | `fastapi`, `uvicorn`, `numpy` (`requirements-serve.txt`). No TensorFlow. |
| Model artifact | `serving/bundle.json`, 106 KB, committed |
| Memory | about 30 MB peak while loading the bundle, plus about 60 MB for Python and FastAPI |
| Inference | about 0.25 ms per `/recommend` (45 dot products and a sort), in-process |
| State | none. Read-only and horizontally scalable. No secrets. |

## Configuration
| Env var | Default | Purpose |
|---|---|---|
| `BUNDLE_PATH` | `serving/bundle.json` | exported model bundle. The app fails at startup if it is missing. |
| `CORS_ORIGINS` | `http://localhost:3000` | comma-separated allowed origins, e.g. `https://<portfolio>.vercel.app,https://<custom-domain>` |
| `PORT` | `8000` | Docker only |

The API has no secrets. If you add a key later (rate limiting, analytics), keep it in the host's env
settings and call the API from a Next.js route handler, not from the browser.

## Local
    uv pip install --python .venv-serve/Scripts/python.exe -r requirements-serve.txt pytest httpx pandas
    PYTHONPATH=src .venv-serve/Scripts/python.exe -m pytest              # serve tests run without TensorFlow
    PYTHONPATH=src .venv-serve/Scripts/python.exe -m uvicorn twotower.serve:app --reload

## Docker
    docker build -t phb-retrieval .
    docker run -p 8000:8000 -e CORS_ORIGINS=https://<portfolio-domain> phb-retrieval

The image contains only `serve.py`, `__init__.py` and the bundle (see `.dockerignore`).

## Regenerating the artifact
`serving/bundle.json` is built from the private dataset and the saved weights:
`python -m twotower.train --tag final --save`, then `python -m twotower.export --tag final`.
The export stops if the restored model or the exported vectors do not reproduce the `runs.csv` test
metrics. It also replaces user, shop, category and product database ids with pseudonyms (`u01`,
`shop1`, `cat1`, `p01`). Product names and prices are kept.

## Hosting
The workload is a tiny stateless numpy service, so any container host works:
- **Google Cloud Run** or **Render / Fly.io**: deploy the Dockerfile as-is. Scale to zero is fine
  because startup is under a few seconds. This is the recommended option.
- **Vercel Python Functions**: also feasible, because the runtime has no TensorFlow and the bundle is
  106 KB. You would need a `vercel.json` / `api/` entrypoint, which is not included. Keep the API as its
  own project rather than inside the Next.js app, so the two deploy independently.
- Not needed: GPU, a database, a vector store.

Production URL: `https://<RETRIEVAL_API_URL>` (placeholder, not deployed yet).

## API contract
Interactive docs live at `/docs` (Swagger) and `/openapi.json`. TypeScript types can be generated from
the OpenAPI spec, e.g. `npx openapi-typescript https://<RETRIEVAL_API_URL>/openapi.json -o retrieval.d.ts`.

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{status:"ok", model, users, products}` |
| GET | `/model` | model config and the offline evaluation table (`evaluation.results[name]["hit@k"]`) |
| GET | `/products` | `Product[]`: `{id, name, price (IDR), category, shop, train_count}` |
| GET | `/users` | `{id, train_interactions}[]` (pseudonymised; the only users the model knows) |
| GET | `/examples` | `{user_id, cart[], held_out}[]`: real held-out test baskets for the demo |
| POST | `/recommend` | see below |

`POST /recommend` request:
```json
{ "user_id": "u39", "cart": ["p32", "p31", "p40"], "k": 5, "baseline": "personal_frequency" }
```
- `user_id` is required. `cart` is optional (up to 45 ids, duplicates are dropped, and cart items are
  never returned). `k` is 1-45 (default 10). `baseline` is `"personal_frequency"`, `"popularity"` or null.

Response:
```json
{ "user_id": "u39", "cart": ["p32","p31","p40"], "k": 5,
  "items": [ { "rank": 1, "score": 3.662868,
               "product": { "id": "p17", "name": "Daun Bawang - 250 gr", "price": 2000,
                            "category": "cat4", "shop": "shop2", "train_count": 262 } } ],
  "baseline": { "name": "personal_frequency", "items": [ ... ] } }
```
`score` is the user-product dot product for the two-tower model. For baselines it is the training
purchase count (for the user, or overall). The two kinds of score are not comparable; compare ranks.

Errors (FastAPI `{"detail": ...}` body):
- `404`: unknown `user_id`. There is no cold-start path.
- `422`: validation failure (missing `user_id`, `k` out of range, unknown baseline) or an unknown
  product id in `cart`.

## Suggested portfolio demo (built in the Next.js app, not in this repo)
1. **Retrieval explorer.** Load `/examples`, `/users` and `/products`. Let the visitor pick a user and
   basket, then show the two-tower top-k next to a baseline's top-k. For an example basket, highlight
   where its `held_out` item ranks.
2. **Evaluation panel.** Chart `GET /model` -> `evaluation.results` (hit@1/3/5/10 for each method).
   Show `evaluation.data_note` next to it: the data is synthetic, and the two-tower only matches
   personal frequency.
3. **Architecture.** Use `reports/architecture.png` and the flow diagram in the README.

If the API is down, fall back to a clearly labelled "recorded example" mode that uses a saved
`/recommend` response. Do not use made-up scores.
