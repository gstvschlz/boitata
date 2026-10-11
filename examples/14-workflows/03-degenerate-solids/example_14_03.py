"""
# Find and fix degenerate solids

A solid that arrives from another program or from hand editing can look whole on screen and still be broken:
a hole a few triangles wide, a patch wound inside out, a stray triangle hanging off an edge. Inside tests and
volumes need a closed, consistently wound solid. You damage a clean solid in five typical ways, find each defect
with `Mesh.validate`, repair what `Mesh.repair` can, fix the rest by hand, and check the result against the
original.
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
from common import ACCENT, GRAY, HIGHLIGHT, INK, LIGHT, save

# %% [markdown]
# !!! learn "What you'll learn"
#     - What makes a triangle mesh a valid solid, and the defects that break it.
#     - How to read the `MeshReport` from `Mesh.validate`: a summary of counts and one row per problem.
#     - What `Mesh.repair` fixes, and how to fix a hole or a stray triangle it leaves.
#     - Why a closed mesh can still give a wrong volume.
#
#     Prerequisites: [solids](../../02-data-and-geometry/05-solids/README.md) and [mesh files](../../02-data-and-geometry/07-mesh-files/README.md), which repairs a mesh read as loose triangles.
#
# ## The problem
#
# A solid is a closed surface: every edge is shared by exactly two triangles, and every triangle is wound the
# same way, counter-clockwise seen from outside. Inside tests and volumes rely on both.
#
# <figure class="bt-figure">
# --8<-- "svg/w1-mesh-defects.svg"
# <figcaption><b>Figure 1.</b> The defects <code>Mesh.validate</code> reports. A boundary edge belongs to one triangle
# (a hole); a non-manifold edge to three or more (a fin); an inconsistent edge is run the same way by its two
# triangles (a flipped face); a degenerate face has no area; a duplicate face repeats the corners of another.</figcaption>
# </figure>
#
# ## The data
#
# Lens 2 of the stacked sulphide lenses is a clean solid. `validate` returns a `MeshReport`: its `summary` counts
# each kind of problem.

# %%
data = bt.datasets.stacked_sulphide_lenses()
lens = data["lens_2"]
report = lens.validate()
print(lens, report)
print(f"volume {lens.volume:,.0f} m3")

# %% [markdown]
# Five defects go in at five places, each a common way a solid gets broken:
#
# - a patch of triangles within 80 m of one point is wound the other way, as when a surface is joined back in
#   reversed;
# - the triangles within 12 m of the lowest point are deleted, a hole;
# - three triangles are repeated, as when two exports of the same surface are merged;
# - one triangle gets a repeated corner, a sliver with no area;
# - a triangle is added on an edge near the top, a fin sticking out, as a stray digitized point makes.

# %%
vertices, triangles = lens.coords, lens.triangles
centers = vertices[triangles].mean(axis=1)


def near(point, radius):
    return np.linalg.norm(centers - point, axis=1) < radius


flipped = near(centers.mean(axis=0) + [0, 0, 60], 80)
hole = near(centers[centers[:, 2].argmin()], 12)
damaged = triangles.copy()
damaged[flipped] = damaged[flipped, ::-1]
damaged = damaged[~hole]
top = damaged[centers[~hole][:, 2].argmax()]
fin_tip = vertices[top[:2]].mean(axis=0) + [0, 0, 25]
damaged = np.vstack(
    [
        damaged,
        damaged[[4000, 4001, 4002]],
        [damaged[8000, [0, 0, 1]]],
        [[top[0], top[1], len(vertices)]],
    ]
)
broken = bt.Mesh(np.vstack([vertices, fin_tip]), damaged)
print(f"{flipped.sum()} triangles flipped, {hole.sum()} deleted")
print(broken)

# %% [markdown]
# ## Why it matters
#
# The broken solid is open, so `volume` and `contains` raise an error:

# %%
for name, call in [("volume", lambda: broken.volume), ("contains", lambda: broken.contains(centers[:1]))]:
    try:
        call()
    except ValueError as error:
        print(f"{name}: {error}")

# %% [markdown]
# The flipped patch alone leaves the mesh closed, so neither call raises. Its volume is wrong, and so is the inside
# test near the patch. Twenty thousand random points in the lens's bounding box show it:

# %%
only_flipped = triangles.copy()
only_flipped[flipped] = only_flipped[flipped, ::-1]
closed_but_wrong = bt.Mesh(vertices, only_flipped)
rng = np.random.default_rng(0)
low, high = np.array(lens.bounds)
points = low + rng.random((20_000, 3)) * (high - low)
truth = lens.contains(points)
print(f"closed: {closed_but_wrong.is_closed}")
print(
    f"volume {closed_but_wrong.volume:,.0f} m3 against {lens.volume:,.0f} m3 "
    f"({closed_but_wrong.volume / lens.volume - 1:+.1%})"
)
wrong = closed_but_wrong.contains(points) != truth
print(f"{truth.sum()} points inside, {wrong.sum()} classified wrong")

# %% [markdown]
# !!! pitfall "Pitfall"
#     `is_closed` checks only the edges. A closed mesh with a flipped patch passes it and gives a volume
#     that looks plausible. Read `inconsistent_edges` and `inward_shells` in the report before trusting a volume.
#
# !!! step "Step 1: Validate"
#     `validate` counts each kind of problem in `summary` and lists them in `problems`, a table with one row per
#     problem. Edge problems name a triangle (`face`) holding the edge, the edge's first corner (`edge`, 0 to 2)
#     and its first vertex. `other` is the earlier face a duplicate repeats, or the other face of a flipped edge.

# %%
report = broken.validate()
print({k: v for k, v in report.summary.items() if v})
problems = report.problems.to_polars()
print(problems.group_by("kind", maintain_order=True).len())
print(problems.head(3))

# %% [markdown]
# Five defects give 266 rows. The edges of the repeated triangles become non-manifold, each now held by three or
# more triangles. The fin adds a non-manifold edge where it meets the lens and two boundary edges along its free
# sides. The flipped patch shows only along its rim, where flipped and unflipped triangles meet.
#
# !!! step "Step 2: Locate the problems"
#     Each row's `face` gives a place: the center of that triangle. Plotted on the lens by kind, the rows fall into
#     the five places the damage went in.

# %%
face_centers = broken.coords[broken.triangles].mean(axis=1)
styles = {
    "degenerate_face": ("s", INK),
    "duplicate_face": ("D", INK),
    "boundary_edge": ("o", HIGHLIGHT),
    "non_manifold_edge": ("^", ACCENT),
    "inconsistent_winding": (".", GRAY),
}
fig = plt.figure(figsize=(9, 6.5), layout="constrained")
ax = fig.add_subplot(projection="3d")
ax.plot_trisurf(*vertices.T, triangles=triangles, color=LIGHT, linewidth=0, alpha=0.25)
for kind, (marker, color) in styles.items():
    faces = problems.filter(problems["kind"] == kind)["face"].to_numpy()
    xyz = face_centers[faces]
    ax.scatter(*xyz.T, marker=marker, color=color, s=18, depthshade=False, label=f"{kind} ({len(faces)})")
ax.set_box_aspect(high - low)
ax.set(xlabel="Easting", ylabel="Northing", zlabel="Elevation")
ax.tick_params(labelsize=6)
ax.legend(loc="upper left")
ax.set_title("Problems reported by validate, at the triangle each row names")
save(fig, "problems")

# %% [markdown]
# !!! step "Step 3: Repair what can be repaired automatically"
#     `repair` drops degenerate and repeated triangles and rewinds each connected piece consistently, outward for
#     a closed piece. With `tolerance` it also welds vertices closer than that distance, which fixes a solid read as
#     loose triangles ([mesh files](../../02-data-and-geometry/07-mesh-files/README.md)). It does not delete
#     triangles that belong to the surface or invent new ones, so the fin and the hole stay.

# %%
repaired = broken.repair()
report = repaired.validate()
print(repaired)
print({k: v for k, v in report.summary.items() if v})

# %% [markdown]
# !!! step "Step 4: Fix the rest by hand"
#     The one non-manifold edge left is where the fin meets the lens. Of the three triangles on that edge, the fin
#     is the one that also has boundary edges; deleting it and repairing again drops its unused tip vertex. The
#     boundary edges left then run around the hole. Fanning a triangle from each of them to the mean of their
#     corners closes it. Each new triangle runs its boundary edge in the opposite direction to the triangle across
#     it, so the winding stays consistent.

# %%
problems = report.problems.to_polars()
tri = repaired.triangles
edge = problems.filter(problems["kind"] == "non_manifold_edge").row(0, named=True)
a, b = tri[edge["face"], edge["edge"]], tri[edge["face"], (edge["edge"] + 1) % 3]
on_edge = np.flatnonzero(np.isin(tri, [a, b]).sum(axis=1) == 2)
open_faces = problems.filter(problems["kind"] == "boundary_edge")["face"].to_numpy()
fin = np.intersect1d(on_edge, open_faces)
print(f"triangles on the non-manifold edge {on_edge}, fin {fin}")
without_fin = bt.Mesh(repaired.coords, np.delete(tri, fin, axis=0)).repair()

rim = without_fin.validate().problems.to_polars()
rim = rim.filter(rim["kind"] == "boundary_edge")
tri = without_fin.triangles
start = tri[rim["face"].to_numpy(), rim["edge"].to_numpy()]
end = tri[rim["face"].to_numpy(), (rim["edge"].to_numpy() + 1) % 3]
patch_center = len(without_fin.coords)
cap = np.column_stack([end, start, np.full(len(start), patch_center)])
fixed = bt.Mesh(
    np.vstack([without_fin.coords, without_fin.coords[start].mean(axis=0)]), np.vstack([tri, cap])
)
print(f"hole rim: {len(rim)} edges, closed with {len(cap)} triangles")
print(fixed, fixed.validate())

# %% [markdown]
# !!! check "Check before you move on"
#     The fixed solid has no problems left. Its volume differs from the original by the difference between the
#     flat fan and the curved surface the hole removed, and the inside test agrees with the original's on all but
#     the points in that sliver.

# %%
assert fixed.validate().summary["is_closed"]
print(
    f"volume {fixed.volume:,.0f} m3, original {lens.volume:,.0f} m3 ({fixed.volume / lens.volume - 1:+.3%})"
)
disagree = fixed.contains(points) != truth
print(f"inside test: {disagree.sum()} of {len(points):,} points disagree with the original")

# %% [markdown]
# ## The decision
#
# Validate every solid before using it, and read every count in the summary, `inconsistent_edges` and `inward_shells` included. Let `repair` handle
# duplicates, slivers and winding; delete fins and close holes by hand, then validate again until the report is
# empty. When a hole is larger than a few triangles, a flat fan no longer follows the surface: go back to whoever
# built the solid instead.
#
# !!! seealso "See also"
#     - [Solids](../../02-data-and-geometry/05-solids/README.md): inside tests and block proportions on valid solids.
#     - [Mesh files](../../02-data-and-geometry/07-mesh-files/README.md): reading, writing and welding loose triangles.
#     - [Flag solid proportions in a block model](../../14-workflows/01-solid-proportions/README.md): what a valid solid is for.
