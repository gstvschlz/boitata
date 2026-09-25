import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

from style import ACCENT, DATA, GREY, HIGHLIGHT, INK, OUT, grid, map_axes, save, table

s = table(OUT / "01_samples.csv")
curve = table(OUT / "01_declustering.csv")
summary = {k: v[0] for k, v in table(OUT / "01_summary.csv").items()}
truth = grid(DATA / "walker_exhaustive.csv")
norm = PowerNorm(0.5, vmin=0, vmax=1500)

fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
image = a.imshow(truth, origin="lower", extent=(0.5, 260.5, 0.5, 300.5), norm=norm)
map_axes(a, "Exhaustive V (78 000 values)")
b.scatter(s["x"], s["y"], c=s["v"], s=14, norm=norm, edgecolors=INK, linewidths=0.3)
b.set_xlim(a.get_xlim())
b.set_ylim(a.get_ylim())
map_axes(b, "470 samples, clustered in high-V areas")
fig.colorbar(image, ax=(a, b), shrink=0.8, label="V (ppm)")
save(fig, "01-data", "maps")

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(curve["cell"], curve["mean"], color=ACCENT, lw=1.6)
ax.axhline(summary["naive"], color=GREY, ls=":", lw=1)
ax.axhline(summary["truth"], color=INK, ls="--", lw=1)
ax.plot(summary["cell"], summary["declustered"], "o", color=HIGHLIGHT, ms=6)
ax.annotate(f"minimum: {summary['declustered']:.0f} ppm at {summary['cell']:.1f} m cells",
            (summary["cell"], summary["declustered"]), xytext=(10, 0), textcoords="offset points", va="center", color=HIGHLIGHT)
ax.text(curve["cell"][-1], summary["naive"], f"naive mean {summary['naive']:.0f}", va="bottom", ha="right", color=GREY)
ax.text(curve["cell"][-1], summary["truth"], f"true mean {summary['truth']:.0f}", va="top", ha="right", color=INK)
ax.set_title("Cell declustering: mean against cell size")
ax.set_xlabel("Cell size (m)")
ax.set_ylabel("Declustered mean of V (ppm)")
save(fig, "01-data", "declustering")

fig, ax = plt.subplots(figsize=(6, 3.4))
bins = np.linspace(0, 1600, 33)
exhaustive = truth.ravel()
ax.hist(exhaustive, bins, weights=np.full(exhaustive.size, 1 / exhaustive.size), color="#e6e6e6", label="exhaustive")
ax.hist(s["v"], bins, weights=np.full(s["v"].size, 1 / s["v"].size), histtype="step", color=GREY, lw=1.4, label="samples, equal weights")
ax.hist(s["v"], bins, weights=s["w"] / s["w"].sum(), histtype="step", color=ACCENT, lw=1.6, label="samples, declustered")
ax.set_title("Declustering moves the sample histogram toward the truth")
ax.set_xlabel("V (ppm)")
ax.set_ylabel("Proportion")
ax.legend()
save(fig, "01-data", "histograms")
