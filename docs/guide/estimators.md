# fit, transform, predict

Transforms, estimators and simulators share one lifecycle. You build the object with its options, `fit` it to samples, and then ask it for results. The same calls work for a normal-score transform, for kriging and for simulation.

!!! learn "What you'll learn"
    - The three states of a transform or an estimator.
    - Where results go: arrays, new columns, diagnostic tables.
    - How seeds make a simulation repeatable.

    Prerequisites: [How Boitatá is organized](organization.md) and [Containers and Arrow](containers.md).

## Three states

<figure class="bt-figure">
--8<-- "svg/g-lifecycle.svg"
<figcaption><b>Figure 1.</b> Both kinds start configured, with options and no data. <code>fit</code> stores what the object needs from the samples. An estimator then predicts at targets; a transform maps values to scores and, with <code>inverse_transform</code>, scores back to values.</figcaption>
</figure>

1. **Configured.** The constructor takes options only: a variogram, a search, a number of classes. Nothing is computed yet.
2. **Fitted.** `fit` takes the samples and returns the same object, now holding what it learned. Because it returns the object, you can chain: `bt.OrdinaryKriging(variogram, search).fit(samples, "V")`.
3. **Results.** A transform answers `transform` and `inverse_transform`; an estimator answers `predict`; a simulator answers `simulate`. You can ask as many times as you like, at as many targets as you like, without fitting again.

## Transforms

A transform learns a mapping from the data and applies it to any values. `NormalScore` learns the table that turns the grades into scores with a standard normal distribution:

```python
import boitata as bt
import numpy as np

samples = bt.datasets.walker_lake()

normal_score = bt.NormalScore().fit(samples["V"])
scores = normal_score.transform(samples["V"])
back = normal_score.inverse_transform(scores)
print(f"std {scores.std():.2f}, back to V: {np.allclose(back, samples['V'])}")
```

```text
std 1.00, back to V: True
```

`fit_transform` does both steps at once, with one difference: it gives tied values, such as Walker Lake's 22 zeros, distinct scores by breaking ties at random from `seed`, while `transform` sends equal values to one score.

`Capping`, `BoxCox`, `HermiteAnamorphosis`, `PPMT`, `PCA`, `MAF` and the other transforms follow the same pattern. The [Transforms API](../api/transforms/index.md) lists them.

## Estimators

An estimator takes a variogram and a search, fits to samples and predicts at targets. The targets can be an `(n, 3)` array, a `PointSet` or a `BlockModel`:

```python
variogram = bt.experimental_variogram(samples, "V", 10.0, 120.0).fit("spherical")
search = bt.Search(radius=60, max_samples=24, min_samples=4)
grid = bt.BlockModel(origin=(2.5, 2.5), size=(5, 5), count=(52, 60))

kriging = bt.OrdinaryKriging(variogram, search).fit(samples, "V")
estimate, variance = kriging.predict(grid, return_variance=True)
grid = grid.with_column("V", estimate).with_column("V_variance", variance)
print(grid)
```

```text
BlockModel(regular, 3120 of 3120 cells, count [52, 60, 1], size [5.0, 5.0, 1.0], rotation [0.0, 0.0, 0.0])
  V: Float64
  V_variance: Float64
```

### Where results go

`predict` returns plain NumPy arrays, one value per target in the order of the targets. For a block model that order is the row order, so you can put the array on the model with `with_column`. A target the search could not fill comes back as `NaN` and is stored as null.

Ask for `diagnostics=True` and `predict` returns a `Table` with one row per target instead:

```python
report = kriging.predict(grid, diagnostics=True)
print(report.column_names)
grid = grid.with_columns({"slope": report["slope"], "n_samples": report["n_samples"]})
```

```text
['value', 'variance', 'efficiency', 'slope', 'n_samples', 'pass', 'n_holes', 'n_other_domain', 'mean_distance', 'negative_weight_sum', 'lagrange', 'support_variance', 'estimate_variance', 'max_samples_reached', 'target_met']
```

The slope of regression and the kriging efficiency say how far each estimate can be trusted; [kriging diagnostics](../examples/10-checking-models/03-kriging-diagnostics/example_10_03.md) reads them.

A fitted estimator also checks itself. `cross_validate` re-estimates every sample from the others with the same variogram and search:

```python
check = kriging.cross_validate()
print(f"mean error {check.mean_error:.1f}, RMSE {check.rmse:.1f}, slope {check.slope:.2f}")
```

```text
mean error 6.8, RMSE 182.8, slope 1.11
```

`OrdinaryKriging`, `SimpleKriging`, `UniversalKriging`, `BlockKriging`, `IndicatorKriging`, `InverseDistance` and the rest share `fit`, `predict`, `cross_validate` and `with_search`. The [Estimation API](../api/estimation/index.md) lists them.

## Simulators and seeds

A simulator draws many equally likely maps, called realizations, in place of one smooth estimate. `SGS` fits like an estimator, on normal scores, and `simulate` returns a `SimulationSummary`: the mean, variance and probabilities over the realizations, and the realizations themselves when you pass `keep=True`.

```python
sgs = bt.SGS(bt.Variogram([("spherical", 1.0, 40.0)]), search).fit(samples, scores)
first = sgs.simulate(grid, n=10, seed=42, keep=True)
again = sgs.simulate(grid, n=10, seed=42, keep=True)
print(first.realizations.shape, np.array_equal(first.realizations, again.realizations))
```

```text
(10, 3120) True
```

Realization `k` draws from a seed built from your `seed` and `k`, and the work is split across threads so that the thread count never changes the numbers. The same seed gives the same realizations with one thread or many; change the seed to get a new set.

!!! key "Key idea"
    Options go to the constructor, data goes to `fit`, targets go to `predict`, `transform` or `simulate`. Results come back as arrays in target order, ready for `with_column`.

!!! pitfall "Pitfall"
    The method names match scikit-learn's, but these objects are not scikit-learn estimators. They take one column of values plus coordinates, not a 2D feature matrix, and have no `get_params`, so they do not drop into a scikit-learn `Pipeline` unchanged. Chain them by hand, as above.

!!! seealso "See also"
    - [Normal-score transform](../examples/04-transforms/01-normal-score/example_04_01.md)
    - [Ordinary kriging](../examples/06-kriging/01-ordinary-kriging/example_06_01.md)
    - [Cross-validation](../examples/10-checking-models/02-cross-validation/example_10_02.md)
    - [Sequential Gaussian simulation](../examples/08-stochastic-simulation/01-sgs/example_08_01.md)
    - [Saving models](saving.md): a fitted transform saves as JSON, a fitted estimator as Parquet.
