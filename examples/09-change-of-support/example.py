import sys
from pathlib import Path

import ceres as cs
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from common import ACCENT, GREY, HIGHLIGHT, INK, fetch, save

samples = cs.PointSet.from_table(cs.read_csv(fetch("walker-lake/sample.csv")))
truth = cs.read_csv(fetch("walker-lake/exhaustive.csv"))["V"].reshape(300, 260)
xy, v = samples.coords, samples["V"]
azimuth = cs.Variogram.from_json((HERE.parent / "03-variography" / "model.json").read_text()).rotation[0]
weights = cs.cell_declustering(xy, v, sizes=np.arange(2.5, 102.5, 2.5)).weights

anam = cs.HermiteAnamorphosis(degree=40).fit(v, weights=weights)
y = anam.transform(v)
major = cs.experimental_variogram(xy, y, 10, 120, azimuth=azimuth).fit("spherical")
minor = cs.experimental_variogram(xy, y, 10, 120, azimuth=azimuth + 90).fit("spherical")
total = major.sill
gaussian = cs.Variogram(
    [("spherical", major.structures[0].sill / total, major.structures[0].range)],
    nugget=major.nugget / total,
    rotation=(azimuth, 0, 0),
    ratios=(min(minor.structures[0].range / major.structures[0].range, 1.0), 1.0),
)

size = 10
r, block = cs.change_of_support(anam, gaussian, size=(size, size), discretization=(5, 5, 1))
blocks_true = truth.reshape(30, size, 26, size).mean(axis=(1, 3)).ravel()
print(
    f"r = {r:.3f}; point variance {anam.variance_:.0f}, block {block.variance_:.0f}, true block {blocks_true.var():.0f}"
)

cutoffs = np.linspace(0, 1000, 41)
model_point = anam.grade_tonnage(cutoffs)
model_block = block.grade_tonnage(cutoffs)


def empirical(values):
    tonnage = np.array([(values > c).mean() for c in cutoffs])
    grade = np.array([values[values > c].mean() if (values > c).any() else np.nan for c in cutoffs])
    return tonnage, grade


true_point = empirical(truth.ravel())
true_block = empirical(blocks_true)

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
for curves, model, color, label in (
    (true_point, model_point, GREY, "points"),
    (true_block, model_block, ACCENT, f"{size} × {size} m blocks"),
):
    a.plot(cutoffs, curves[0], color=color, lw=3, alpha=0.35)
    a.plot(
        cutoffs,
        model["tonnage"],
        color=color,
        lw=1.4,
        ls="--",
        label=f"{label}: model (dashed), truth (wide)",
    )
    b.plot(cutoffs, curves[1], color=color, lw=3, alpha=0.35)
    b.plot(cutoffs, model["mean_grade"], color=color, lw=1.4, ls="--")
a.set(title="Proportion above cutoff", xlabel="Cutoff V (ppm)", ylabel="Proportion of area")
a.legend(fontsize=8)
b.set(title="Mean grade above cutoff", xlabel="Cutoff V (ppm)", ylabel="Mean V above cutoff (ppm)")
save(fig, HERE, "grade-tonnage")

grid = cs.BlockModel(origin=(0.5, 0.5), size=(5, 5), count=(52, 60))
dk = cs.DisjunctiveKriging(anam, gaussian, cs.Search(radius=100, max_samples=24), order=20).fit(xy, v)
cutoff = 500.0
p = dk.predict_tonnage(grid, cutoff)
nodes = grid.centroids.astype(int)
above = truth[nodes[:, 1] - 1, nodes[:, 0] - 1] > cutoff
edges = np.linspace(0, 1, 11)
bins = np.clip(np.digitize(p, edges) - 1, 0, 9)
predicted = np.array([p[bins == k].mean() for k in range(10)])
observed = np.array([above[bins == k].mean() for k in range(10)])
counts = np.bincount(bins, minlength=10)
print(f"DK: mean predicted P(V > {cutoff:.0f}) {p.mean():.3f}, true proportion {above.mean():.3f}")

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
image = a.imshow(
    np.clip(p, 0, 1).reshape(60, 52), origin="lower", extent=(0.5, 260.5, 0.5, 300.5), vmin=0, vmax=1
)
a.contour(
    above.reshape(60, 52).astype(float),
    levels=[0.5],
    origin="lower",
    extent=(0.5, 260.5, 0.5, 300.5),
    colors=HIGHLIGHT,
    linewidths=0.8,
)
a.set_aspect("equal")
a.set(title=f"Disjunctive kriging: P(V > {cutoff:.0f} ppm)", xlabel="Easting (m)", ylabel="Northing (m)")
fig.colorbar(image, ax=a, shrink=0.8, label="probability; true V > 500 outlined")
keep = counts > 20
b.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
b.scatter(predicted[keep], observed[keep], s=np.sqrt(counts[keep]) * 4, color=ACCENT)
b.set(
    xlim=(0, 1),
    ylim=(0, 1),
    xlabel="Predicted probability",
    ylabel="Observed frequency at 3 120 nodes",
    title="Calibration",
)
b.set_aspect("equal")
b.text(0.03, 0.92, "marker area ∝ nodes per bin", color=INK, fontsize=8)
save(fig, HERE, "disjunctive")
