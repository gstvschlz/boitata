# 4. Ordinary kriging

Ordinary kriging of `V` on a 5 m grid with the model from [chapter 3](../03-variography/README.md), up to 24 samples within 100 m along the major axis.

![maps](maps.png)

The kriging standard deviation depends only on the data layout and the model: low near samples, high in gaps.

![validation](validation.png)

Kriging is smooth: estimates vary less than the truth (variance 35 600 against 62 300 ppm²), so the regression of estimates on true values has a slope below 1.
Cross-validation re-estimates each sample without it: mean error 5.6 ppm, RMSE 190 ppm, correlation 0.78.
The mean of error² / kriging variance is 0.65; below 1 means the model's variance is somewhat pessimistic here.

```python
model = cs.Variogram.from_json(open("model.json").read())
grid = cs.BlockModel(origin=(0.5, 0.5), size=(5, 5), count=(52, 60))
ok = cs.OrdinaryKriging(model, cs.Search(radius=100, max_samples=24, min_samples=4))
estimate, variance = ok.fit(samples.coords, samples["V"]).predict(grid, return_variance=True)
grid = grid.with_column("estimate", estimate)
cv = ok.cross_validate()
```

[`example.py`](example.py)
