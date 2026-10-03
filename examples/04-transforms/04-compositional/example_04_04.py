"""
# compositional data

porphyry 1 geometallurgical samples: seven minerals in % plus the remainder, a composition summing to 100. raising one
part lowers the others, so raw correlations mix geology with the constant-sum constraint, and estimating parts
independently can break the total.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, INK, save

data = bt.datasets.porphyry_geometallurgy(deposit=1)["synthetic_drillholes"]
minerals = ["arcilla", "calcosina", "bornita", "calcopirita", "tenantita", "molibdenita", "pirita"]
names = ["clay", "chalcocite", "bornite", "chalcopyrite", "tennantite", "molybdenite", "pyrite", "rest"]
parts = np.column_stack([data[m] for m in minerals])
parts = np.column_stack([parts, 100 - parts.sum(axis=1)])

assert (parts > 0).all(), "log-ratios need positive parts"
composition = bt.closure(parts, total=100)


# %% [markdown]
# the isometric log-ratio (ILR) maps each composition to 7 unconstrained coordinates, the balances. a sequential binary
# partition chooses them: the copper sulphides against the other parts, chalcocite and bornite against chalcopyrite and
# tennantite, and so on, each row of signs splitting one group of the row before it in two. `ILR` replaces the part
# columns with the balances; the projection-pursuit multivariate transform (PPMT) then turns those into independent
# standard gaussians, ready for independent simulation. the way back must return every composition.

# %%
samples = bt.PointSet(data.coords, dict(zip(names, composition.T, strict=True)))
#        clay  cc  bn  cp  tn  mo  py rest
signs = [
    [-1, 1, 1, 1, 1, -1, -1, -1],  # copper sulphides | the rest
    [0, 1, 1, -1, -1, 0, 0, 0],  # chalcocite, bornite | chalcopyrite, tennantite
    [0, 1, -1, 0, 0, 0, 0, 0],  # chalcocite | bornite
    [0, 0, 0, 1, -1, 0, 0, 0],  # chalcopyrite | tennantite
    [1, 0, 0, 0, 0, -1, -1, 1],  # clay, rest | molybdenite, pyrite
    [1, 0, 0, 0, 0, 0, 0, -1],  # clay | rest
    [0, 0, 0, 0, 0, 1, -1, 0],  # molybdenite | pyrite
]
balances = [f"ilr_{i + 1}" for i in range(7)]
pipe = bt.Pipeline(
    [
        ("ilr", bt.ILR(parts=names, basis=signs, total=100)),
        ("ppmt", bt.PPMT(iterations=40, seed=7), balances),
    ]
)
gaussian = pipe.fit_transform(samples)
back = pipe.inverse_transform(gaussian)
error = max(np.abs(back[n] - samples[n]).max() for n in names)
print(f"round trip max error {error:.2e} %")
coords = np.column_stack([pipe.named_steps["ilr"].transform(samples)[b] for b in balances])
gauss = np.column_stack([gaussian[b] for b in balances])


# %% [markdown]
# correlations at each stage:

# %%
cmap = "cividis"
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
panels = [
    (np.corrcoef(composition.T), names, "Raw percentages"),
    (np.corrcoef(coords.T), [f"ilr{i + 1}" for i in range(coords.shape[1])], "ILR coordinates"),
    (np.corrcoef(gauss.T), [f"g{i + 1}" for i in range(gauss.shape[1])], "After PPMT"),
]
for ax, (corr, labels, title) in zip(axes, panels):
    image = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_title(title)
    off = np.abs(corr[~np.eye(len(corr), dtype=bool)])
    ax.set_xlabel(f"mean |r| off the diagonal {off.mean():.2f}", color=GRAY)
fig.colorbar(image, ax=axes, shrink=0.8, label="correlation")
save(fig, "correlations")


# %% [markdown]
# two parts before, two gaussian coordinates after:

# %%
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
a.scatter(composition[:, 3], composition[:, 6], s=3, color=ACCENT, alpha=0.3, linewidths=0)
a.set(xlabel="Chalcopyrite (%)", ylabel="Pyrite (%)", title="Two parts of the composition")
b.scatter(gauss[:, 0], gauss[:, 1], s=3, color=ACCENT, alpha=0.3, linewidths=0)
t = np.linspace(0, 2 * np.pi, 200)
for radius in (1, 2, 3):
    b.plot(radius * np.cos(t), radius * np.sin(t), color=GRAY, lw=0.6)
b.set_aspect("equal")
b.set(xlabel="g1", ylabel="g2", title="PPMT output: standard bivariate normal")
b.text(2.2, -3.3, "circles: 1, 2, 3 σ", color=INK, fontsize=8)
save(fig, "scatter")
