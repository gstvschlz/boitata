import math

import matplotlib.pyplot as plt
import numpy as np
from style import ACCENT, GREY, HIGHLIGHT, INK, LIGHT, OUT, save, table

d = table(OUT / "02_scores.csv")
order = np.argsort(d["v"])
v, w, y = d["v"][order], d["w"][order], d["score"][order]
cdf = np.cumsum(w) / w.sum()

grid = np.linspace(-3.5, 3.5, 400)
normal_cdf = 0.5 * (1 + np.vectorize(math.erf)(grid / math.sqrt(2)))
normal_pdf = np.exp(-(grid**2) / 2) / math.sqrt(2 * math.pi)

p = 0.75
vp = v[np.searchsorted(cdf, p)]
yp = y[np.searchsorted(cdf, p)]

fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, layout="constrained")
a.step(v, cdf, where="post", color=ACCENT, lw=1.4)
a.set_title("Declustered CDF of V")
a.set_xlabel("V (ppm)")
a.set_ylabel("Cumulative probability")
b.plot(grid, normal_cdf, color=INK, lw=1.4)
b.set_title("Standard normal CDF")
b.set_xlabel("Normal score")
for ax, x in ((a, vp), (b, yp)):
    ax.axhline(p, color=HIGHLIGHT, lw=0.9, ls="--")
    ax.vlines(x, 0, p, color=HIGHLIGHT, lw=0.9, ls="--")
    ax.plot(x, p, "o", color=HIGHLIGHT, ms=5)
a.annotate(f"V = {vp:.0f} ppm", (vp, 0.02), xytext=(4, 0), textcoords="offset points", color=HIGHLIGHT)
b.annotate(f"score = {yp:.2f}", (yp, 0.02), xytext=(4, 0), textcoords="offset points", color=HIGHLIGHT)
fig.suptitle(
    f"Each value maps to the normal score with the same cumulative probability ({p:.2f} shown)",
    x=0.01,
    ha="left",
    fontsize=9,
    color=GREY,
)
save(fig, "02-normal-score", "quantile-mapping")

fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
a.hist(v, np.linspace(0, 1600, 33), weights=w / w.sum(), color=LIGHT, edgecolor=GREY, lw=0.5)
a.set_title("V: positively skewed")
a.set_xlabel("V (ppm)")
a.set_ylabel("Proportion (declustered)")
bins = np.linspace(-3.5, 3.5, 29)
b.hist(
    y,
    bins,
    weights=w / w.sum() / np.diff(bins)[0],
    color=LIGHT,
    edgecolor=GREY,
    lw=0.5,
    label="normal scores",
)
b.plot(grid, normal_pdf, color=ACCENT, lw=1.6, label="N(0, 1)")
b.set_title("Normal scores: standard Gaussian")
b.set_xlabel("Normal score")
b.set_ylabel("Density (declustered)")
b.legend()
save(fig, "02-normal-score", "histograms")
