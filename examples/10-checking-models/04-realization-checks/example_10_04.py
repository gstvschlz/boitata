"""
# Realization checks

Realizations are only as good as what they reproduce: the declustered histogram of the data, the variogram model they
were drawn from and, with several variables, the correlations between them. `check_realizations` measures all three
for every realization, and `cs.plot.histogram_reproduction`, `variogram_reproduction` and `correlation_reproduction`
draw each as a band across realizations against its target. Realization variograms on a grid pair cells by index
shifts, in parallel over realizations.
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import save

# %% [markdown]
# Walker Lake V, simulated as in [sequential Gaussian simulation](../../08-stochastic-simulation/01-sgs/README.md): declustered normal scores, their variogram along N170° and across it,
# and 50 SGS realizations on a 5 m grid.

# %%
samples = cs.datasets.walker_lake()
xy, v = samples.coords, samples["V"]
weights = cs.cell_declustering(xy, v, sizes=np.arange(2.5, 102.5, 2.5)).weights
y = cs.NormalScore(tails=(0.0, v.max())).fit_transform(v, weights=weights)

azimuth, lag, max_lag = 170.0, 10.0, 120.0
major = cs.experimental_variogram(xy, y, lag, max_lag, azimuth=azimuth).fit("spherical")
minor = cs.experimental_variogram(xy, y, lag, max_lag, azimuth=azimuth + 90).fit("spherical")
total, a_major = major.sill, major.structures[0].range
gaussian = cs.Variogram(
    [("spherical", major.structures[0].sill / total, a_major)],
    nugget=major.nugget / total,
    rotation=(azimuth, 0, 0),
    ratios=(min(minor.structures[0].range / a_major, 1.0), 1.0),
)
grid = cs.BlockModel(origin=(0.5, 0.5), size=(5, 5), count=(52, 60))
sgs = cs.SGS(gaussian, cs.Search(radius=100, max_samples=24)).fit(samples, "V", weights=weights)
summary = sgs.simulate(grid, n=50, seed=42, keep=True)

# %% [markdown]
# One call checks them all. Given the model, the variograms run along its azimuth and across it, in the normal scores
# of the declustered data, the units the model is in. `statistics` holds what `describe` gives for the declustered
# data (realization 0) and for every realization:

# %%
check = cs.check_realizations(
    grid, summary, samples, "V", weights=weights, variogram=gaussian, lag=lag, max_lag=max_lag
)
stats = check.statistics
for name in ["mean", "std", "P50", "P90"]:
    column = stats[name]
    print(f"{name:5} data {column[0]:6.0f}   realizations {column[1:].min():6.0f} to {column[1:].max():6.0f}")

# %% [markdown]
# The realizations' distributions straddle the declustered data's in grades and in normal scores, save the step at
# the lowest scores, where every realization value of 0 ppm shares one score. Their variograms
# follow the model along and across N170° up to its ranges; beyond them the realizations wander around the sill, as
# single 260 × 300 m fields must, and the band shows how far.

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
cs.plot.histogram_reproduction(check, ax=axes[0])
cs.plot.histogram_reproduction(check, scores=True, ax=axes[1])
cs.plot.variogram_reproduction(check, ax=axes[2])
axes[0].set(xlim=(0, 1600), xlabel="V (ppm)", title="Histogram")
axes[1].set(xlim=(-3, 3), title="Normal scores")
axes[2].set(xlabel="Lag distance (m)", title="Variogram of normal scores")
save(fig, "walker_lake")

# %% [markdown]
# ## Several variables
#
# Log chalcocite and log tennantite of porphyry 1, cosimulated through PPMT by turning bands as in [multivariate simulation](../../08-stochastic-simulation/06-multivariate-simulation/README.md). With a
# list of summaries, one per variable, the check adds each realization's correlation matrix.

# %%
data = cs.datasets.porphyry_geometallurgy(deposit=1)["synthetic_drillholes"]
coords = data.coords
names = ["chalcocite", "tennantite"]
logs = cs.PointSet(coords, {"chalcocite": np.log(data["calcosina"]), "tennantite": np.log(data["tenantita"])})
pair = np.column_stack([logs[n] for n in names])
weights = cs.cell_declustering(coords, pair[:, 0], cell_size=50.0).weights

lo, hi = coords.min(axis=0), coords.max(axis=0)
nodes = cs.BlockModel(origin=tuple(lo), size=(25, 25, 25), count=tuple(np.ceil((hi - lo) / 25).astype(int)))
ppmt = cs.PPMT(seed=7)
factors = ppmt.fit_transform(pair, weights=weights)
simulators = [
    cs.TurningBands(cs.experimental_variogram(coords, f, 25.0, 300.0).fit("spherical")) for f in factors.T
]
simulation = cs.MultivariateSimulation(ppmt, simulators).fit(logs, names, weights=weights)
reals = simulation.simulate(nodes, n=20, seed=1, keep=True)

multi = cs.check_realizations(
    nodes, reals, logs, names, weights=weights, lag=25.0, max_lag=200.0, directions=[(0, 0), (90, 0), (0, 90)]
)
r = multi.correlations[:, 0, 1]
print(
    f"{len(nodes.centroids):,} nodes; correlation data {multi.data_correlation[0, 1]:.2f}, "
    f"realizations {r.min():.2f} to {r.max():.2f}"
)

# %% [markdown]
# Both declustered histograms and the correlation are reproduced; the variograms are not. The factors were simulated
# with omnidirectional models, so the realizations are nearly isotropic and smoother than the data, which are more
# continuous down the holes than across them. The check points at the next step: directional factor variograms.

# %%
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), layout="constrained")
for ax, name in zip(axes, names):
    cs.plot.histogram_reproduction(multi, variable=name, ax=ax)
    ax.set(xlabel=f"log {name} (%)", title=name.capitalize())
cs.plot.variogram_reproduction(multi, variable="chalcocite", ax=axes[2])
axes[2].set(xlabel="Lag distance (m)", title="Chalcocite normal scores")
cs.plot.correlation_reproduction(multi, ax=axes[3])
axes[3].set_title("Correlation")
save(fig, "porphyry")
