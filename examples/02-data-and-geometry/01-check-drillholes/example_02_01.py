"""
# Checking drill holes

The stacked sulphide lenses as logged: collar, survey (dip positive down, azimuth clockwise from north), assay and
lithology tables with typical data-entry errors planted in them, each listed in the dataset's `raw/README.md`. The
tables are checked, repaired and checked again before anything is desurveyed.
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
from common import ACCENT, GRAY, HIGHLIGHT, save

# %% [markdown]
# `check_drillholes` flags every record of the collar, survey and interval tables: duplicate ids, missing or sentinel
# values, text in numeric columns, from ≥ to, gaps, overlaps, angles out of range, depths past the collar `LENGTH`,
# ids that match a collar only once trimmed and case-folded, holes missing from a table, abrupt survey deviation and
# holes dipping against the rest. Each flagged check is listed with the holes it hit.

# %%
data = bt.datasets.stacked_sulphide_lenses(raw=True)
raw = {
    "collar": data["collars"],
    "survey": data["surveys"],
    "assays": data["assays"],
    "lithology": data["lithology"],
}
for name, table in raw.items():
    print(f"{name:>9}: {table.num_rows:6} rows")


def check(t, show=True):
    flags, summary, details = bt.check_drillholes(
        t["collar"], t["survey"], {"assays": t["assays"], "lithology": t["lithology"]}, max_depth="LENGTH"
    )
    for table, name, rows, holes in zip(
        summary["table"], summary["check"], summary["rows"], summary["holes"]
    ):
        if rows and show:
            ids = np.array(t[table]["HOLE_ID"])
            hit = sorted(set(ids[np.asarray(flags[table][name])]))
            listed = ", ".join(repr(h) for h in hit) if len(hit) <= 8 else f"{hit[0]!r} ... {hit[-1]!r}"
            print(f"{table:>9} {name:<12} {rows:4.0f} rows in {holes:3.0f} holes: {listed}")
    return flags, details


flags, details = check(raw)

# %% [markdown]
# Every flag points at a planted error, or follows from one:
#
# - `duplicate` collars: `DD0067` entered twice with different coordinates, `DD0058` the same row twice.
# - `id_mismatch`: `'dd0062'` in the assays and `'DD0132 '` in the lithology are ids in the wrong case or with a
#   trailing space; the collar `'DD0104 '` has the trailing space, so its survey, assays and lithology miss it.
# - `no_collar`: `DD0111` has surveys, assays and lithology but no collar.
# - `no_survey`: `DD0151` has no survey.
# - `deviation`: two stations of `DD0197`, where the azimuth flips at one station and back at the next.
# - `dip_sign`: `DD0116` is entered with its dips negative, pointing up, while every other hole dips down.
# - `past_depth`: `DD0055` has surveys, assays and lithology below its collar `LENGTH`.
# - `inverted`: an assay of `DD0100` with `FROM` > `TO`.
# - `overlap`: two assays of `DD0200` start before the previous one ends, and three of `DD0162` are entered twice.
# - `gap`: two missing samples in `DD0080`; the gaps in `DD0062`, `DD0100`, `DD0132` and `DD0200` are left by the
#   rows above. The RC holes' gaps are unsampled core and real.
# - `text_values` and `sentinel`: the grades were read as text, because some cells are not numbers; `DD0110` holds
#   `-99` and `-999`.
# - `no_assays`: 112 collars. Only mineralized zones are assayed, so a hole that never logs `MS`, `SMS` or `STR`
#   has no assays by design; one more hides among them.
#
# The details name the value behind each id, dip and text flag, and what it should become:

# %%
groups = {}
for table, check_, hole, column, value, suggestion in zip(
    *(details[c] for c in ("table", "check", "hole", "column", "value", "suggestion"))
):
    groups.setdefault((table, check_, hole, column), set()).add(
        f"{value!r} -> {suggestion!r}" if suggestion else repr(value)
    )
for (table, check_, hole, column), values in groups.items():
    values = sorted(values)
    listed = ", ".join(values) if len(values) <= 2 else f"{values[0]} ... {values[-1]} ({len(values)} values)"
    print(f"{table:>9} {check_:<12} {hole!r:9} {column:<7} {listed}")

# %% [markdown]
# A hole missing its assays looks like one of the unassayed holes, unless its lithology says it is mineralized:

# %%
lith = raw["lithology"]
mineralized = set(np.array(lith["HOLE_ID"])[np.isin(lith["LITH"], ["MS", "SMS", "STR"])])
print("mineralized, no assays:", sorted(mineralized - set(raw["assays"]["HOLE_ID"])))

# %% [markdown]
# Two repairs need a person: the inverted interval's ends are swapped, and `DD0055`'s `LENGTH` is taken from its
# deepest survey station, since the survey, assays and lithology agree the hole is longer. The tables are checked
# again, and `fix_drillholes` resolves the rest with one named rule per check. Two rules are asked for by name:
# `dip_sign="negate"` turns `DD0116` down, and `text_values="half"` reads `<0.01` as half the detection limit
# (by default, as `NS` is, it becomes null). The log says what each rule changed: ids are renamed to their collar,
# the later duplicate collar is dropped (for `DD0067`, the second, wrong entry), overlapping assays keep the one that
# starts first (dropping the twinned re-entries), the two `DD0197` stations turned by the flip are dropped,
# sentinels become null and rows of a hole without collar go.


# %%
def replace(table, **columns):
    return bt.Table({**{c: table[c] for c in table.column_names}, **columns})


tables = dict(raw)
a, s, c = raw["assays"], raw["survey"], raw["collar"]
swap = a["FROM"] > a["TO"]
tables["assays"] = replace(a, FROM=np.where(swap, a["TO"], a["FROM"]), TO=np.where(swap, a["FROM"], a["TO"]))
deepest = {}
for h, d in zip(s["HOLE_ID"], s["DEPTH"]):
    deepest[h] = max(deepest.get(h, 0.0), d)
tables["collar"] = replace(c, LENGTH=np.maximum(c["LENGTH"], [deepest.get(h, 0.0) for h in c["HOLE_ID"]]))

flags, _ = check(tables, show=False)
fixed, log = bt.fix_drillholes(flags, tables, dip_sign="negate", text_values="half")
for table, name, action, rows in zip(log["table"], log["check"], log["action"], log["rows"]):
    if rows:
        print(f"{action:>10} {rows:3.0f} {table} rows flagged {name}")
print()
flags, _ = check(fixed)

# %% [markdown]
# What is left needs the field records, not a rule: `DD0151` has no survey and would be desurveyed as vertical,
# `DD0043` needs its assays found, and the gaps are the two lost `DD0080` samples, the two `DD0200` intervals
# dropped as overlaps and the unsampled RC core. `'DD0104 '` keeps its trailing space, now in every table. `DD0116`
# shows why the dips matter: desurveyed as entered, it climbs out of the ground.


# %%
def dd0116(table):
    return table.filter(table["HOLE_ID"] == "DD0116")


collar = dd0116(fixed["collar"])
fig, ax = plt.subplots(figsize=(4, 5.5), layout="constrained")
for name, survey, color in (("as entered", raw["survey"], HIGHLIGHT), ("fixed", fixed["survey"], ACCENT)):
    path = bt.Drillholes(collar, dd0116(survey)).paths()
    ax.plot(path["x"], path["z"], color=color, lw=1.6, label=name)
ax.axhline(collar["Z"][0], color=GRAY, lw=0.6, ls="--", label="collar elevation")
ax.set_aspect("equal")
ax.set(title="DD0116, east-west section", xlabel="Easting (m)", ylabel="Elevation (m)")
ax.legend()
save(fig, "dip")
