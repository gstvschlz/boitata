"""
# mesh-crossing interval splits

an assay interval rarely starts or ends at a geological contact, since a logger picks the nearest sample boundary
instead of the true crossing. `Drillholes.mesh_intervals` walks each hole's desurveyed path against a closed
`Mesh` and returns the exact depths where it crosses the surface. you `merge_intervals` them with the assays, so
a composite takes its hard domain from the mesh instead of a logged pick.
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
from common import ACCENT, GRAY, save

# %%
data = bt.datasets.stacked_sulphide_lenses()
lens = data["lens_1"]
collar, survey, assay, lithology = data["collars"], data["surveys"], data["assays"], data["lithology"]
holes = bt.Drillholes.from_tables(data)
crossings = holes.mesh_intervals(lens, step=0.5, tolerance=0.01)
print(f"{crossings.num_rows} runs down {len(holes)} holes, {int(np.sum(crossings['INSIDE']))} inside lens 1")

# %% [markdown]
# `mesh_intervals` samples each hole's path every `step` and bisects to `tolerance` wherever it crosses the mesh, so
# each depth down each hole falls in one run of `INSIDE`. its columns match the defaults of `merge_intervals`, so it
# splits the assays at those depths with no renaming.

# %%
domain = np.where(crossings["INSIDE"], "lens", "other")
boundary = {
    "HOLE_ID": crossings["HOLE_ID"],
    "FROM": crossings["FROM"],
    "TO": crossings["TO"],
    "DOMAIN": domain,
}
mesh_split = bt.merge_intervals(assay, boundary)
mesh_comps = bt.Drillholes(collar, survey, mesh_split).composite(None, ["ZN_PCT"], domain="DOMAIN")

# %% [markdown]
# [contact surfaces](../../11-geological-modeling/02-contact-surfaces/README.md) domains the same holes from the logged lithology instead: `MS` and `SMS` intervals stand for a sulphide
# lens. that pick is only as good as where the logger set the contact, and it cannot tell the three lenses apart.

# %%
lith_split = bt.merge_intervals(assay, lithology)
lith_domain = np.where(np.isin(lith_split["LITH"], ["MS", "SMS"]), "lens", "other")
lith_table = {**{c: lith_split[c] for c in lith_split.column_names}, "DOMAIN": lith_domain}
lith_comps = bt.Drillholes(collar, survey, lith_table).composite(None, ["ZN_PCT"], domain="DOMAIN")


def grade_by_domain(comps):
    stats = bt.describe_by("ZN_PCT", "DOMAIN", data=comps)
    return dict(zip(stats["category"], stats["mean"])), dict(zip(stats["category"], stats["n"]))


mesh_mean, mesh_n = grade_by_domain(mesh_comps)
lith_mean, lith_n = grade_by_domain(lith_comps)
print(f"     mesh contact: lens mean Zn {mesh_mean['lens']:.2f}%, n={mesh_n['lens']:.0f}")
print(f"logged lithology: lens mean Zn {lith_mean['lens']:.2f}%, n={lith_n['lens']:.0f}")

# %% [markdown]
# the lithology domain takes in sulphide picked in the other two lenses. nothing corrects it to the solid, so some
# contacts sit a sample or two off the true one. both effects pull its lens grade away from the mesh's.

# %%
labels = ["lens", "other"]
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(4.5, 3.4), layout="constrained")
for i, (name, mean, color) in enumerate(
    [("mesh contact", mesh_mean, ACCENT), ("logged lithology", lith_mean, GRAY)]
):
    ax.bar(x + (i - 0.5) * width, [mean[k] for k in labels], width, color=color, label=name)
ax.set(xticks=x, xticklabels=labels, ylabel="Mean Zn (%)", title="Domain grade by contact")
ax.legend()
save(fig, "grade")
