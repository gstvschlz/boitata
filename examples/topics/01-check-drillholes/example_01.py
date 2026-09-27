"""
# 1. Checking drill holes

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
import ceres as cs
import matplotlib.pyplot as plt
import numpy as np
from common import ACCENT, GRAY, HIGHLIGHT, save

# %% [markdown]
# `check_drillholes` flags every record of the collar, survey and interval tables: duplicate ids, missing or sentinel
# values, from ≥ to, gaps, overlaps, angles out of range, depths past the collar `LENGTH`, holes missing from a table
# and abrupt survey deviation. Each flagged check is listed with the holes it hit.

# %%
data = cs.datasets.stacked_sulphide_lenses(raw=True)
raw = {
    "collar": data["collars"],
    "survey": data["surveys"],
    "assays": data["assays"],
    "lithology": data["lithology"],
}
GRADES = ["ZN_PCT", "PB_PCT", "CU_PCT", "AG_GPT", "AU_GPT"]
for name, table in raw.items():
    print(f"{name:>9}: {table.num_rows:6} rows")


def check(t):
    flags, summary = cs.check_drillholes(
        t["collar"], t["survey"], {"assays": t["assays"], "lithology": t["lithology"]}, max_depth="LENGTH"
    )
    for table, name, rows, holes in zip(
        summary["table"], summary["check"], summary["rows"], summary["holes"]
    ):
        if rows:
            ids = np.array(t[table]["HOLE_ID"])
            hit = sorted(set(ids[np.asarray(flags[table][name])]))
            listed = ", ".join(repr(h) for h in hit) if len(hit) <= 8 else f"{hit[0]!r} ... {hit[-1]!r}"
            print(f"{table:>9} {name:<12} {rows:4.0f} rows in {holes:3.0f} holes: {listed}")
    return flags


flags = check(raw)

# %% [markdown]
# Every flag points at a planted error, or follows from one:
#
# - `duplicate` collars: `DD0067` entered twice with different coordinates, `DD0058` the same row twice.
# - `no_collar`: `DD0111` has surveys, assays and lithology but no collar; `'dd0062'` and the lithology of
#   `'DD0132 '` are ids in the wrong case or with a trailing space, and the collar `'DD0104 '` has a trailing space,
#   so none of its survey, assay or lithology rows find it (it also has `no_survey` and `no_lithology`).
# - `no_survey`: `DD0151` has no survey.
# - `deviation`: two stations of `DD0197`, where the azimuth flips at one station and back at the next.
# - `past_depth`: `DD0055` has surveys, assays and lithology below its collar `LENGTH`.
# - `inverted`: an assay of `DD0100` with `FROM` > `TO`.
# - `overlap`: two assays of `DD0200` start before the previous one ends, and three of `DD0162` are entered twice.
# - `gap`: two missing samples in `DD0080`; the gaps in `DD0062`, `DD0100`, `DD0132` and `DD0200` are left by the
#   rows above. The RC holes' gaps are unsampled core and real.
# - `no_assays`: 113 collars. Only mineralized zones are assayed, so a hole that never logs `MS`, `SMS` or `STR`
#   has no assays by design. `'DD0104 '` is the id error again, and one more hides among them.
#
# Three errors no generic check can see. The grades were read as text, because some cells are not numbers:


# %%
def is_number(v):
    try:
        return float(v) == float(v)
    except (TypeError, ValueError):
        return False


for g in GRADES:
    column = np.asarray(raw["assays"][g], dtype=object)
    text = [v for v in column if v is not None and not is_number(v)]
    if text:
        hit = sorted(set(np.array(raw["assays"]["HOLE_ID"])[np.isin(column, text)]))
        print(f"{g}: {', '.join(sorted(set(text)))} in {', '.join(hit)}")

# %% [markdown]
# A hole entered with its dips negative points up, and is consistent station to station, so it passes every check;
# and a hole missing its assays looks like one of the unassayed holes, unless its lithology says it is mineralized:

# %%
survey, lith = raw["survey"], raw["lithology"]
print("upward dips:", sorted(set(np.array(survey["HOLE_ID"])[survey["DIP"] < 0])))
mineralized = set(np.array(lith["HOLE_ID"])[np.isin(lith["LITH"], ["MS", "SMS", "STR"])])
print("mineralized, no assays:", sorted(mineralized - set(raw["assays"]["HOLE_ID"])))

# %% [markdown]
# The repairs that need a person: ids are trimmed and upper-cased; `NS` (not sampled) becomes null and `<0.01` half
# the detection limit; upward dips are turned down; the inverted interval's ends are swapped; and `DD0055`'s
# `LENGTH` is taken from its deepest survey station, since the survey, assays and lithology agree the hole is
# longer. The tables are checked again: now that the grades are numbers, the `-99` and `-999` sentinels of
# `DD0110` show up too.


# %%
def replace(table, **columns):
    return cs.Table({**{c: table[c] for c in table.column_names}, **columns})


def number(v):
    if v is None or v == "NS":
        return np.nan
    return float(v[1:]) / 2 if v.startswith("<") else float(v)


tables = {
    name: replace(t, HOLE_ID=np.char.upper(np.char.strip(np.asarray(t["HOLE_ID"], dtype=str))))
    for name, t in raw.items()
}
a, s, c = tables["assays"], tables["survey"], tables["collar"]
swap = a["FROM"] > a["TO"]
tables["assays"] = replace(
    a,
    FROM=np.where(swap, a["TO"], a["FROM"]),
    TO=np.where(swap, a["FROM"], a["TO"]),
    **{g: np.array([number(v) for v in np.asarray(a[g], dtype=object)]) for g in GRADES},
)
tables["survey"] = replace(s, DIP=np.abs(s["DIP"]))
deepest = {}
for h, d in zip(s["HOLE_ID"], s["DEPTH"]):
    deepest[h] = max(deepest.get(h, 0.0), d)
tables["collar"] = replace(c, LENGTH=np.maximum(c["LENGTH"], [deepest.get(h, 0.0) for h in c["HOLE_ID"]]))

flags = check(tables)

# %% [markdown]
# `fix_drillholes` resolves the rest with one named rule per check, and its log says what each rule changed: the
# later duplicate collar is dropped (for `DD0067`, the second, wrong entry), overlapping
# assays keep the one that starts first (dropping the twinned re-entries), the two `DD0197` stations turned by the
# flip are dropped, sentinels become null and rows of a hole without collar go.

# %%
fixed, log = cs.fix_drillholes(flags, tables)
for table, name, action, rows in zip(log["table"], log["check"], log["action"], log["rows"]):
    if rows:
        print(f"{action:>5} {rows:3.0f} {table} rows flagged {name}")
print()
flags = check(fixed)

# %% [markdown]
# What is left needs the field records, not a rule: `DD0151` has no survey and would be desurveyed as vertical,
# `DD0043` needs its assays found, and the gaps are the two lost `DD0080` samples, the two `DD0200` intervals
# dropped as overlaps and the unsampled RC core. `DD0116` shows why the dips matter: desurveyed as entered, it
# climbs out of the ground.


# %%
def dd0116(table):
    return table.filter(table["HOLE_ID"] == "DD0116")


collar = dd0116(fixed["collar"])
fig, ax = plt.subplots(figsize=(4, 5.5), layout="constrained")
for name, survey, color in (("as entered", raw["survey"], HIGHLIGHT), ("fixed", fixed["survey"], ACCENT)):
    path = cs.Drillholes(collar, dd0116(survey)).paths()
    ax.plot(path["x"], path["z"], color=color, lw=1.6, label=name)
ax.axhline(collar["Z"][0], color=GRAY, lw=0.6, ls="--", label="collar elevation")
ax.set_aspect("equal")
ax.set(title="DD0116, east-west section", xlabel="Easting (m)", ylabel="Elevation (m)")
ax.legend()
save(fig, "dip")
