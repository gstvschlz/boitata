"""
# Saving and loading

Models, transforms and searches save to JSON: `to_json` returns a string and the class's `from_json` builds the
object back from it. A fitted transform keeps what it learned, so the loaded copy transforms new values the same
way. Containers and fitted estimators hold columns and go to Parquet instead
([storing containers in Parquet](../04-parquet/README.md)).
"""

# %% [hidden]
import sys
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parents[1]))

# %%
import tempfile

import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import map_axes, save

# %% [markdown]
# ## A small workflow
#
# Walker Lake `V` goes through five fitted pieces: cell declustering weights, a cap at the declustered P99, a
# normal-score transform, a variogram of the scores and a search. Simple kriging of the scores, back-transformed,
# gives the estimate on a 5 m grid.

# %%
samples = cs.datasets.walker_lake()
declustering = cs.cell_declustering(samples, "V", sizes=np.arange(2.5, 102.5, 2.5))
capping = cs.Capping(quantile=0.99).fit("V", weights=declustering.weights, data=samples)
capped = capping.transform("V", data=samples)
normal_score = cs.NormalScore().fit(capped, weights=declustering.weights)
scores = normal_score.transform(capped)
variogram = cs.Variogram.fit(
    cs.experimental_variogram(samples, scores, 10.0, 120.0), ["spherical", "spherical"]
)
search = cs.Search(radius=60, max_samples=24, min_samples=4)

grid = cs.BlockModel(origin=(2.5, 2.5), size=(5, 5), count=(52, 60))
kriged = cs.SimpleKriging(variogram, search).fit(samples, scores).predict(grid, progress=False)
estimate = normal_score.inverse_transform(kriged)
print(f"cap: {capping.caps_:.0f} ppm | declustered mean: {declustering.mean:.1f} ppm")
print(variogram)

# %% [markdown]
# ## Saving
#
# Each piece goes to its own file. The declustering file stores one weight per sample in the order of `samples`,
# and the normal-score file stores the fitted transform table; both grow with the data, while the cap, the
# variogram and the search take a few hundred bytes.

# %%
folder = Path(tempfile.mkdtemp())
pieces = {
    "declustering": declustering,
    "capping": capping,
    "normal_score": normal_score,
    "variogram": variogram,
    "search": search,
}
for name, piece in pieces.items():
    (folder / f"{name}.json").write_text(piece.to_json())
    print(f"{name + '.json':>18}: {(folder / f'{name}.json').stat().st_size:>6} bytes")
print((folder / "search.json").read_text())

# %% [markdown]
# The files are plain JSON with a `type` and a `format` number, so a reader of another type or a newer format
# raises `InvalidInput` in place of loading the wrong object.
#
# ## Loading and rerunning
#
# `rerun` sees only the folder and the samples. It reads the cap, the normal-score table, the variogram and the
# search, applies them in the same order and returns the estimate. The weights stay out: the fitted cap and table
# already carry them. They come back separately, for declustered statistics in a report.

# %%


def rerun(folder, samples):
    capping = cs.Capping.from_json((folder / "capping.json").read_text())
    normal_score = cs.NormalScore.from_json((folder / "normal_score.json").read_text())
    variogram = cs.Variogram.from_json((folder / "variogram.json").read_text())
    search = cs.Search.from_json((folder / "search.json").read_text())
    scores = normal_score.transform(capping.transform("V", data=samples))
    grid = cs.BlockModel(origin=(2.5, 2.5), size=(5, 5), count=(52, 60))
    kriged = cs.SimpleKriging(variogram, search).fit(samples, scores).predict(grid, progress=False)
    return normal_score.inverse_transform(kriged)


again = rerun(folder, cs.datasets.walker_lake())
assert np.array_equal(again, estimate)
print("identical estimate:", np.array_equal(again, estimate))
weights = cs.Declustering.from_json((folder / "declustering.json").read_text()).weights
print(f"declustered mean from the saved weights: {np.average(capped, weights=weights):.1f} ppm (capped)")

fig, ax = plt.subplots(figsize=(5, 5.2), layout="constrained")
image = ax.imshow(
    again.reshape(60, 52), origin="lower", extent=(0, 260, 0, 300), vmin=0, vmax=np.quantile(again, 0.99)
)
fig.colorbar(image, ax=ax, shrink=0.8, label="V (ppm)")
map_axes(ax, "Estimate rebuilt from the saved pieces")
save(fig, "estimate")

# %% [markdown]
# The rerun matches bit for bit: JSON writes each float with enough digits to read back the same number.
#
# The same pair exists on `Structure`, `Coregionalization`, `HighGrade`, `Categories`, `Trend` and the other
# transforms (`HermiteAnamorphosis`, `BoxCox`, `PPMT`, `PCA`, `MAF`, `StepwiseConditional`, `UniformConditioning`,
# `GaussianImputer`, `KernelDensity`, `GaussianMixture`). These objects also pickle, through the same JSON.
# A fitted estimator such as the `SimpleKriging` above carries its samples, so it saves with `to_parquet`.
