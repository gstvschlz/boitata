"""
# cross-validation

ordinary kriging of walker lake `V` re-estimates each sample from the others, with the same variogram and search:
leave-one-out, then k-fold with fewer and fewer folds.
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
from common import save

samples = bt.datasets.walker_lake()
xy, v = samples.coords, samples["V"]
azimuths = np.arange(0, 180, 22.5)
directional = [bt.experimental_variogram(xy, v, 10.0, 120.0, azimuth=a) for a in azimuths]
model = bt.Variogram.fit_directional(
    directional, [(a, 0) for a in azimuths], ["spherical", "spherical"], weighting="count/gamma"
)
search = bt.Search(radius=80, max_samples=24, min_samples=4, rotation=model.rotation, ratios=(0.5, 1.0))
kriging = bt.OrdinaryKriging(model, search).fit(xy, v)


# %% [markdown]
# leave-one-out removes one sample at a time. k-fold removes a whole fold, sample `i` going to fold `i % k`, so each
# estimate sees data a fraction `1/k` sparser. the slope comes from the regression of actual on estimated values and
# is 1 without conditional bias. the standardized squared error is the mean of error² over kriging variance, 1 when
# the kriging variance is calibrated.

# %%
results = {}
for folds in (None, 10, 5, 2):
    name = "leave-one-out" if folds is None else f"{folds}-fold"
    cv = results[name] = kriging.cross_validate(folds=folds)
    print(
        f"{name:>13}: RMSE {cv.rmse:5.1f} ppm, correlation {cv.correlation:.2f}, slope {cv.slope:.2f}, "
        f"standardized squared error {cv.standardized_squared_error:.2f}"
    )

# %% [markdown]
# errors grow as folds get fewer. leave-one-out judges the model at the sample spacing, which in the clustered areas
# is finer than most blocks see. a slope above 1 and a standardized squared error below 1 hold at each k: high
# estimates understate the samples a little, and the kriging variance is too large. `bt.plot.cross_validation`
# ([result plots](../../10-checking-models/06-result-plots/README.md)) draws a result as actual against estimate, with the regression line and these statistics:

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, name in zip(axes, ("leave-one-out", "2-fold"), strict=True):
    bt.plot.cross_validation(results[name], ax=ax)
    ax.set_title(name.capitalize())
fig.tight_layout()
save(fig, "cross_validation")
