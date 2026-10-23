"""Draw the architecture diagram -> reports/architecture.png.  python -m twotower.diagram"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .data import ROOT

BLUE, ORANGE, GREEN, GRAY = "#dbe9f6", "#fde7c8", "#d9f0d9", "#eeeeee"


def draw(path=ROOT / "reports" / "architecture.png"):
    fig, ax = plt.subplots(figsize=(16, 9.5))
    ax.set(xlim=(0, 16), ylim=(-1.2, 11.4))
    ax.axis("off")

    def box(x, y, text, w=1.9, h=0.95, color=GRAY, dashed=False, fs=9):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=color, ec="#444", lw=1.2, ls="--" if dashed else "-"))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, dashed=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.1,
                                     color="#444", ls="--" if dashed else "-", shrinkA=0, shrinkB=0))

    L = dict(inp=0.6, emb=2.2, cat=3.8, d1=5.2, d2=6.4, out=7.7)

    # ---- user tower
    cx = 3.3
    ax.text(cx - 1.2, 8.7, "USER TOWER (query)", ha="center", fontsize=12, weight="bold")
    box(cx - 1.05, L["inp"], "user_id\n(string)", color=BLUE)
    box(cx + 2.1, L["inp"], "cart multi-hot\n(45 items)\n[--cart]", color=BLUE, dashed=True)
    box(cx - 2.1, L["emb"], "StringLookup\n+ Embedding(32)", color=ORANGE)
    box(cx, L["emb"], "aug Embedding(32)\n[--dat]", color=ORANGE, dashed=True)
    box(cx + 2.1, L["emb"], "Dense(32)\nno bias", color=ORANGE, dashed=True)
    box(cx, L["cat"], "Concat  (32 | +32 cart | +32 aug)", w=5.9, h=0.7)
    box(cx, L["d1"], "Dense 64 + ReLU  (+ Dropout)", w=4.2, h=0.7, color=GREEN)
    box(cx, L["d2"], "Dense 32", w=4.2, h=0.7, color=GREEN)
    box(cx, L["out"], "user embedding q  (32)", w=3.4, h=0.8, color=BLUE)
    top_in, bot_emb = L["inp"] + 0.475, L["emb"] - 0.475
    arrow(cx - 1.05, top_in, cx - 2.1, bot_emb)
    arrow(cx - 1.05, top_in, cx, bot_emb, dashed=True)
    arrow(cx + 2.1, top_in, cx + 2.1, bot_emb, dashed=True)
    for x in (cx - 2.1, cx, cx + 2.1):
        arrow(x, L["emb"] + 0.475, x, L["cat"] - 0.35, dashed=(x != cx - 2.1))
    arrow(cx, L["cat"] + 0.35, cx, L["d1"] - 0.35)
    arrow(cx, L["d1"] + 0.35, cx, L["d2"] - 0.35)
    arrow(cx, L["d2"] + 0.35, cx, L["out"] - 0.4)

    # ---- product tower
    px = 11.4
    ax.text(px + 1.0, 8.7, "PRODUCT TOWER (candidate)", ha="center", fontsize=12, weight="bold")
    cols = [px - 3.3, px - 1.65, px, px + 1.65, px + 3.3]
    ins = ["product_id", "category", "shop", "price_bucket"]
    embs = ["StringLookup\n+ Embedding(32)", "Embedding(16)", "Embedding(16)", "Embedding(16)"]
    for x, t, e in zip(cols[:4], ins, embs):
        box(x, L["inp"], t, w=1.55, color=BLUE, fs=8.5)
        box(x, L["emb"], e, w=1.55, color=ORANGE, fs=8.5)
        arrow(x, L["inp"] + 0.475, x, L["emb"] - 0.475)
        arrow(x, L["emb"] + 0.475, x, L["cat"] - 0.35)
    box(cols[4], L["emb"], "aug Embedding(32)\n[--dat]", w=1.55, color=ORANGE, dashed=True, fs=8.5)
    arrow(cols[0] + 0.3, L["inp"] + 0.475, cols[4], L["emb"] - 0.475, dashed=True)
    arrow(cols[4], L["emb"] + 0.475, cols[4], L["cat"] - 0.35, dashed=True)
    box(px, L["cat"], "Concat  (32 + 16 + 16 + 16 = 80  | +32 aug)", w=8.0, h=0.7)
    box(px, L["d1"], "Dense 64 + ReLU  (+ Dropout)", w=4.2, h=0.7, color=GREEN)
    box(px, L["d2"], "Dense 32", w=4.2, h=0.7, color=GREEN)
    box(px, L["out"], "product embedding c  (32)", w=3.4, h=0.8, color=BLUE)
    arrow(px, L["cat"] + 0.35, px, L["d1"] - 0.35)
    arrow(px, L["d1"] + 0.35, px, L["d2"] - 0.35)
    arrow(px, L["d2"] + 0.35, px, L["out"] - 0.4)

    # ---- loss / serving
    box(7.1, 9.6, "score = q · c   (dot product)", w=4.2, h=0.7, color=ORANGE)
    box(7.1, 10.7, "in-batch softmax loss  (logQ popularity correction, accidental hits removed)",
        w=9.2, h=0.7, color=ORANGE)
    arrow(cx, L["out"] + 0.4, 5.6, 9.6 - 0.35)
    arrow(px, L["out"] + 0.4, 8.6, 9.6 - 0.35)
    arrow(7.1, 9.6 + 0.35, 7.1, 10.7 - 0.35)
    ax.text(7.1, 4.6, "DAT (optional): each aug vector is trained with MSE to the\nstop-gradient embedding of the opposite tower for the positive pair",
            ha="center", va="center", fontsize=8, style="italic", color="#555")
    ax.text(8.0, -0.8, "Serving: embed user (+cart), score all 45 products, mask items already in the cart, return top-k.   "
            "Dashed = optional (flag).", ha="center", fontsize=9)

    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(draw())
