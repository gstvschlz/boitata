# units

a column can carry a unit, kept in its arrow metadata and through parquet. units combine lengths, masses, grades,
ratios, angles and currencies, so `t/m3`, `USD/m` and `(g/t)^2` are units too. grades are a kind of their own: a
grade never converts to a plain ratio, and ore mass times a grade is metal (`kg metal`), never ore.

<details><summary>Python</summary>

```python
import boitata as bt
import matplotlib.pyplot as plt
import numpy as np
from common import save

assays = bt.datasets.stacked_sulphide_lenses()["assays"]
folder = Path(tempfile.mkdtemp())
bt.write_csv(
    folder / "assays.csv",
    bt.Table({"ZN (%)": assays["ZN_PCT"], "AG [g/t]": assays["AG_GPT"], "DENSITY": assays["DENSITY"]}),
)
```

</details>

## units from the file

`read_csv` splits a header such as `ZN (%)` or `AG [g/t]` into the column name and its unit. a column whose header
says nothing takes its unit from `units=`.

<details><summary>Python</summary>

```python
table = bt.read_csv(folder / "assays.csv", units={"DENSITY": "t/m3"})
print(table.column_names)
print(table.units)
```

</details>

```text
['ZN', 'AG', 'DENSITY']
{'ZN': '%', 'AG': 'g/t', 'DENSITY': 't/m3'}
```

a project that always names its columns the same way declares the units once with `set_units`; readers and
constructors then give those columns their unit whenever the file leaves it out. `with bt.units(...)` does the same
for one block.

<details><summary>Python</summary>

```python
with bt.units(columns={"DENSITY": "t/m3"}):
    print(bt.read_csv(folder / "assays.csv").units)
```

</details>

```text
{'ZN': '%', 'AG': 'g/t', 'DENSITY': 't/m3'}
```

## converting

`convert_units` rescales a column to another unit of the same kind. `oz/t` is troy ounces per short ton, as North
American reports write it.

<details><summary>Python</summary>

```python
silver = table.convert_units("AG", to="oz/t")
print(f"{table['AG'][:3]} g/t is {silver['AG'][:3]} oz/t")
density = table.convert_units("DENSITY", to="kg/m3")["DENSITY"]
print(f"mean density {np.nanmean(table['DENSITY']):.2f} t/m3 = {np.nanmean(density):.0f} kg/m3")
```

</details>

```text
[1.  1.5 0.8] g/t is [0.02916667 0.04375    0.02333333] oz/t
mean density 3.26 t/m3 = 3264 kg/m3
```

a conversion across kinds raises `InvalidInput`: a zinc grade in percent is not a ratio in percent.

<details><summary>Python</summary>

```python
try:
    table.convert_units("ZN", to="ratio%")
except bt.InvalidInput as error:
    print(error)
```

</details>

```text
cannot convert grade (%) to ratio (ratio%)
```

plots label their axes with the unit of the column.

<details><summary>Python</summary>

```python
fig, ax = bt.plot.histogram("AG", data=silver)
ax.set_title("silver grades, converted to oz/t")
save(fig, "histogram")
plt.show()
```

</details>

![histogram](histogram.png)

## units through a workflow

a column read from a container is a `UnitArray`: a numpy array that remembers its unit. composites keep the unit of
the assays, an estimator fitted on a column estimates in its unit, and the kriging variance comes out in its square.
`with_column` stores the unit with the new column.

<details><summary>Python</summary>

```python
data = bt.datasets.stacked_sulphide_lenses()
holes = bt.Drillholes(
    data["collars"], data["surveys"], data["assays"].with_units({"ZN_PCT": "%"}), length_unit="m"
)
composites = holes.composite(2.0, ["ZN_PCT"])
grid = bt.BlockModel.from_extents(
    data["lens_1"], size=(10, 10, 5), buffer=20, rotation=(22.5, 0.0, 55.0), length_unit="m"
)
kriging = bt.OrdinaryKriging(bt.Variogram([("spherical", 1.0, 60.0)]), bt.Search(radius=60, max_samples=12))
estimate, variance = kriging.fit(composites, "ZN_PCT").predict(grid, return_variance=True)
print(f"estimate in {estimate.unit}, variance in {variance.unit}")
grid = grid.with_column("ZN", estimate)
print(grid.units)
```

</details>

```text
estimate in %, variance in (%)^2
{'ZN': '%'}
```

arithmetic drops the unit, since the result may be in another one. give it back with `unit=`.

<details><summary>Python</summary>

```python
grid = grid.with_column("ZN_PPM", estimate * 1e4, unit="ppm")
print(grid.units)
```

</details>

```text
{'ZN': '%', 'ZN_PPM': 'ppm'}
```

## lengths

containers carry the length unit of their coordinates, and so do a `Search` and a `Variogram`. an estimator fitted
with parameters in feet on data in metres converts the coordinates, so the estimate is the one the same lengths in
metres give. containers in different units refuse to meet; `to_length_unit` converts one.

<details><summary>Python</summary>

```python
feet = bt.OrdinaryKriging(
    bt.Variogram([("spherical", 1.0, 60.0 / 0.3048)], length_unit="ft"),
    bt.Search(radius=60.0 / 0.3048, max_samples=12, length_unit="ft"),
)
in_feet = feet.fit(composites, "ZN_PCT").predict(grid)
print(f"largest difference from the estimate in metres: {np.nanmax(np.abs(in_feet - estimate)):.1e} %")
try:
    kriging.predict(grid.to_length_unit("ft"))
except bt.InvalidInput as error:
    print(error)
```

</details>

```text
largest difference from the estimate in metres: 1.6e-11 %
samples are in m and targets in ft; convert one with to_length_unit('m')
```

## tonnage and metal

with the block size in metres and a density in `t/m3`, `grade_tonnage` knows that tonnage is a mass: it reads in
t, kt or Mt and metal in the units of the grade's metal, whichever reads best. metal never converts to ore tonnes.

<details><summary>Python</summary>

```python
grid = grid.with_column("DENSITY", np.full(len(grid), 3.1), unit="t/m3")
curve = bt.grade_tonnage("ZN", [0.0, 2.0, 4.0], data=grid, density="DENSITY")
print(curve.units)
print(curve.to_polars())
```

</details>

```text
{'cutoff': '%', 'tonnage': 'Mt', 'mean_grade': '%', 'metal': 'kt metal'}
shape: (3, 4)
┌────────┬──────────┬────────────┬────────────┐
│ cutoff ┆ tonnage  ┆ mean_grade ┆ metal      │
│ ---    ┆ ---      ┆ ---        ┆ ---        │
│ f64    ┆ f64      ┆ f64        ┆ f64        │
╞════════╪══════════╪════════════╪════════════╡
│ 0.0    ┆ 77.4876  ┆ 1.107841   ┆ 858.439493 │
│ 2.0    ┆ 15.33725 ┆ 4.341728   ┆ 665.90172  │
│ 4.0    ┆ 6.83705  ┆ 6.166689   ┆ 421.619582 │
└────────┴──────────┴────────────┴────────────┘
```

## parameters with units

a parameter takes a number in the unit of the data, or text with its own unit. a cutoff in ppm on grades in percent
and a search radius in feet both convert.

<details><summary>Python</summary>

```python
in_ppm = bt.grade_tonnage("ZN", ["20000 ppm"], data=grid, density="DENSITY")
in_percent = bt.grade_tonnage("ZN", [2.0], data=grid, density="DENSITY")
print(in_ppm["tonnage"], in_percent["tonnage"])
search = bt.Search(radius="200 ft", max_samples=12)
print(search.radius, search.length_unit)
```

</details>

```text
[15.33725] [15.33725]
200.0 ft
```

## other estimators and checks

multiple-indicator kriging, like the other estimators, keeps the unit of the values: its means and quantiles are in
percent, and a cutoff can be given in ppm. a cap can be given in another unit too.

<details><summary>Python</summary>

```python
mik = bt.MultipleIndicatorKriging(
    bt.Variogram([("spherical", 0.2, 60.0)]), bt.Search(radius=60, max_samples=12), [1.0, 3.0, 6.0]
)
summary = mik.fit(composites, "ZN_PCT").predict(grid, cutoffs=["30000 ppm"], quantiles=[0.5])
print(f"mean in {summary.mean.unit}, median in {summary.quantile_values.unit}")
capped = bt.Capping(cap="150000 ppm").fit(composites["ZN_PCT"]).transform(composites["ZN_PCT"])
print(f"largest capped grade: {np.nanmax(capped):.1f} {capped.unit}")
```

</details>

```text
mean in %, median in %
largest capped grade: 15.0 %
```

a validation compares blocks and samples in one unit. composites in ppm against a model in percent are converted
first, so both means read in percent rather than one in ppm.

<details><summary>Python</summary>

```python
in_ppm = composites.convert_units("ZN_PCT", to="ppm")
validation = bt.validate_model(grid, "ZN", in_ppm, "ZN_PCT", density="DENSITY")
print(validation.units["mean"])
print(validation.to_polars().select("source", "n", "mean", "mean_diff"))
```

</details>

```text
%
shape: (2, 4)
┌────────┬───────┬──────────┬───────────┐
│ source ┆ n     ┆ mean     ┆ mean_diff │
│ ---    ┆ ---   ┆ ---      ┆ ---       │
│ str    ┆ u64   ┆ f64      ┆ f64       │
╞════════╪═══════╪══════════╪═══════════╡
│ naive  ┆ 13312 ┆ 0.591816 ┆ 0.0       │
│ model  ┆ 50040 ┆ 1.106608 ┆ 0.869851  │
└────────┴───────┴──────────┴───────────┘
```

Full script: [`example_02_25.py`](example_02_25.py)
