# Portfolio demo brief (for the agent building the portfolio page)

Hand this file to an AI agent working in the **portfolio repo** (Next.js + TypeScript on Vercel,
https://naufaldi.vercel.app / https://nfaldi.dev). It describes what to build and every fact the page may
claim. Each fact was checked against the live API.

## Goal
Add a project page that lets a recruiter use a two-tower recommendation model built for an
undergraduate thesis, understand what it returns, and inspect its measured results.

- Backend repo: https://github.com/thisisaldi/pasar-hasil-bumi-retrieval
- Live API: **https://pasarhasilbumi-api.vercel.app**. Read it from an env var such as
  `NEXT_PUBLIC_RETRIEVAL_API_URL`; don't hard-code it.
- Swagger UI: https://pasarhasilbumi-api.vercel.app/docs. OpenAPI: https://pasarhasilbumi-api.vercel.app/openapi.json
- Generate types with `npx openapi-typescript https://pasarhasilbumi-api.vercel.app/openapi.json -o <types path>`
- CORS allows `https://naufaldi.vercel.app`, `https://nfaldi.dev` and `http://localhost:3000`, so the
  browser can call the API directly. No API key or secrets. Any other origin, including Vercel preview
  URLs of the portfolio, is blocked. For previews, proxy the calls through a Next.js route handler or
  ask for the origin to be added.

## Before writing code
Inspect the portfolio repo first: router (app vs pages), how existing project pages are built,
styling system, component library, data-fetching patterns and env handling. Follow its conventions
and reuse its components. Don't add a UI or charting library if something already installed covers
the need.

## What the system does (don't overstate it)
- Agricultural marketplace with **45 products** (vegetables, fruit, spices) and **96 known users**.
- Given a user and their current basket, it ranks the products they are likely to add next.
- Two-tower model (TensorFlow Recommenders). The user tower uses only the user id. The product tower
  uses product id, category, shop and price bucket. Score = dot product of the two 32-d embeddings.
  Training uses in-batch softmax with logQ popularity correction.
- **The served model is not cart-aware.** Basket items are only excluded from results; they don't
  change the ranking otherwise.
- **No cold start.** Unknown users return 404.
- Serving uses precomputed embeddings with numpy + FastAPI on a Vercel Function. Inference is under
  1 ms. Measured round trips are about 0.3-0.5 s, and the first request after idle can take longer
  (cold start).
- User, shop and category ids are pseudonyms (`u39`, `shop2`, `cat4`). Product names and prices (IDR)
  are real catalogue values.
- **The transactions are synthetic.** 80% of picks come from 3-6 favourites per user. The two-tower
  model roughly matches the "personal frequency" baseline and does not beat it. State this on the page.

## API
| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{status: "ok", model: "final", users: 96, products: 45}` |
| GET | `/model` | `{model: {...config}, evaluation: {protocol, split_rows, metric, results, data_note}}` |
| GET | `/products` | `[{id, name, price, category, shop, train_count}]` |
| GET | `/users` | `[{id, train_interactions}]` |
| GET | `/examples` | `[{user_id, cart: string[], held_out}]`: 8 real held-out test baskets |
| POST | `/recommend` | see below |

Request body: `{user_id: string, cart?: string[] (≤45), k?: 1-45 (default 10), baseline?: "personal_frequency" | "popularity" | null}`

Response: `{user_id, cart, k, items: [{rank, score, product}], baseline: {name, items: [{rank, score, product}]} | null}`

- `score` is a dot product for the two-tower model and a training purchase count for baselines. The
  two are not comparable, so compare **ranks**.
- `cart` in the response is de-duplicated. Basket items never appear in results.
- Errors use the body `{detail}`. **404** means an unknown `user_id`. **422** means a validation error:
  missing `user_id`, `k` out of range, unknown baseline, or an unknown product id in `cart`.

Live check, verified:

    POST /recommend {"user_id":"u39","cart":["p32","p31","p40"],"k":3,"baseline":"personal_frequency"}
    two-tower: 1 p17 Daun Bawang - 250 gr (3.66) · 2 p20 Paprika Merah | 500 gr · 3 p11 Cabe Merah Besar - 250 gr
    baseline:  1 p17 · 2 p20 · 3 p14 Kangkung - 500 gr

That basket is `/examples[0]`, and its hidden item `p17` is at rank 1 in both lists.

Measured test results (cart-masked hit@k), served by `GET /model`. Render them from the API; don't
copy them by hand.

| | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| two-tower | 0.387 | 0.755 | 0.841 | 0.892 |
| personal frequency | 0.381 | 0.769 | 0.860 | 0.912 |
| popularity | 0.098 | 0.236 | 0.358 | 0.584 |
| random | 0.017 | 0.076 | 0.120 | 0.233 |

Protocol: transactions are split by time, 70% train, 10% validation, 20% test (4236 / 605 / 1210 rows).
The held-out item is ranked against all 45 products, with the rest of the basket masked. Ties count
as misses.

## What to build
1. **Project page** (e.g. `/projects/pasar-hasil-bumi`). Cover the problem in 2-3 sentences, the
   undergraduate thesis context and the stack (TensorFlow Recommenders, numpy, FastAPI, Vercel). Link
   the GitHub repo and the Swagger docs.
2. **Retrieval explorer** (client component) with two modes:
   - **Test baskets**: pick one of the 8 baskets from `/examples` (default to the first one). This shows
     a real held-out evaluation case.
   - **Free play**: the visitor picks **any user** from `/users` and **any products** from `/products`
     as their basket, then sees the live ranking.
   - The **user** is what drives the ranking, so make the user picker the primary control. Show users
     with their `train_interactions`, and add a "random user" button.
   - Basket products are **excluded from results only**; they don't change the scores. Label the product
     picker accordingly (e.g. "Already in basket (excluded)"), and add a one-line explanation:
     "This model ranks by shopper history. Basket items are removed from results, not used as signals."
     Without this, a visitor who adds an item and sees the list barely change will think it's broken.
     Verified live: u39 with an empty basket gives p32 p40 p31 p17 p20 p11. With [p17, p01] it gives
     p32 p40 p31 p20 p11 p38. u05 with the same basket gives p40 p14 p12 p42 p22 p32.
   - Re-run the request on every change (debounced). Group products by `category` in the picker.
   - Controls: k (default 5), and a baseline (personal frequency / popularity / none).
   - Show the two-tower top-k and the baseline top-k side by side: rank, product name, price as IDR
     (`Intl.NumberFormat("id-ID", {style: "currency", currency: "IDR"})`), score, and a muted
     "bought N times in training" line from `train_count`.
   - For example baskets, highlight the `held_out` item and state its rank in each list. If it isn't
     in the top-k, say so plainly.
   - Show the basket as chips, with the note "basket items are excluded from results".
3. **Evaluation panel.** Fetch `/model` server-side with revalidation (e.g. 1 h). Show
   `evaluation.results` as a grouped bar chart or table. Show `protocol`, `split_rows` and `data_note`.
4. **Architecture.** Show these images, or copy them into `public/`:
   - https://raw.githubusercontent.com/thisisaldi/pasar-hasil-bumi-retrieval/main/reports/architecture.png
   - https://raw.githubusercontent.com/thisisaldi/pasar-hasil-bumi-retrieval/main/reports/final_curves.png

   Add the flow: offline training → export with metric re-check → JSON bundle → FastAPI → dot
   product, mask basket, top-k.
5. **Limitations:** synthetic data, no cold start, the served model ignores cart contents, and a tiny
   catalogue (brute-force scoring, not ANN). Future work: real orders, user features, a cart-aware
   tower, an ANN index.

## Reliability and honesty rules
- Show loading states, e.g. "Waking up the API…" for a cold start. Show a status dot driven by `/health`.
- If the API is unreachable, switch to a clearly labelled **"Recorded example (API offline)"** mode.
  It must use real responses captured from the live API: a script saves `/model`, `/examples`,
  `/products` and `/recommend` for each example basket into a committed JSON file. **Never generate or
  invent scores, rankings or metrics.**
- Don't describe the model as cart-personalised, learning in real time, or validated on real orders.
- The layout must work at phone width. Keep it accessible: labelled controls, keyboard navigable,
  and highlighting that doesn't rely on colour alone.

## Done when
- The page works locally against the live API.
- `npm run build` passes, plus lint and typecheck if configured.
- The u39 example shows p17 at rank 1.
- The offline fallback is tested by pointing the env var at a bad URL.
- The agent reports which files it added or changed, and anything it couldn't verify.
