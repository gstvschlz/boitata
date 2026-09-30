import boitata as bt
import numpy as np
import pytest

rng = np.random.default_rng(11)


@pytest.fixture
def model():
    bm = bt.BlockModel(origin=(0, 0, 0), size=(5, 5, 5), count=(20, 15, 4))
    return bm.with_column("grade", rng.lognormal(0, 1, len(bm)))


def test_chunks_cover_the_model_and_stay_bounded(model, tmp_path):
    path = tmp_path / "model.parquet"
    bt.write_parquet(path, model)
    file = bt.BlockModelFile(path)
    assert len(file) == 1200 and file.count == [20, 15, 4] and file.column_names == ["grade"]
    chunks = list(file.chunks(rows=250))
    assert [len(c) for c in chunks] == [250, 250, 250, 250, 200]
    np.testing.assert_array_equal(np.concatenate([c.centroids for c in chunks]), model.centroids)
    np.testing.assert_array_equal(np.concatenate([c["grade"] for c in chunks]), model["grade"])


def test_map_blocks_kriging_matches_in_memory(model, tmp_path):
    xy = rng.uniform(0, 100, (80, 3)) * [1, 0.75, 0.2]
    v = rng.normal(size=80)
    ok = bt.OrdinaryKriging(bt.Variogram([("spherical", 1.0, 40.0)]), bt.Search(radius=60)).fit(xy, v)
    masked = model.mask(np.arange(len(model)) % 3 != 0)
    source, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    bt.write_parquet(source, masked)
    sizes = []

    def estimate(chunk):
        sizes.append(len(chunk))
        d = ok.predict(chunk, diagnostics=True)
        return {"estimate": d["value"], "class": np.where(d["slope"] > 0.9, "high", "low")}

    bt.map_blocks(source, out, estimate, rows=117)
    assert max(sizes) <= 117 and sum(sizes) == len(masked)
    back = bt.read_parquet(out)
    np.testing.assert_array_equal(back.index, masked.index)
    np.testing.assert_allclose(back["estimate"], ok.predict(masked))
    np.testing.assert_array_equal(back["grade"], masked["grade"])
    assert set(back["class"]) <= {"high", "low"}


def test_map_blocks_rejects_wrong_lengths(model, tmp_path):
    path = tmp_path / "model.parquet"
    bt.write_parquet(path, model)
    with pytest.raises(bt.InvalidInput, match="rows"):
        bt.map_blocks(path, tmp_path / "out.parquet", lambda chunk: {"x": np.zeros(3)}, rows=100)


def test_turning_bands_streamed_equals_in_memory(model, tmp_path):
    xyz = rng.uniform(0, 100, (60, 3)) * [1, 0.75, 0.2]
    values = rng.lognormal(0, 0.5, 60)
    tb = bt.TurningBands(bt.Variogram([("spherical", 1.0, 30.0)]), bands=80).fit(xyz, values)
    whole = tb.simulate(model, n=8, seed=3, cutoffs=[1.5], quantiles=[0.1, 0.9])
    source, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    bt.write_parquet(source, model)
    result = tb.simulate_to_parquet(source, out, n=8, seed=3, cutoffs=[1.5], quantiles=[0.1, 0.9], rows=333)
    back = bt.read_parquet(out)
    np.testing.assert_array_equal(back["mean"], whole.mean)
    np.testing.assert_array_equal(back["p_above_1.5"], whole.probability_above[:, 0])
    np.testing.assert_array_equal(back["q0.9"], whole.quantile_values[:, 1])
    np.testing.assert_array_equal(back["grade"], model["grade"])
    np.testing.assert_allclose(result["realization_mean"], whole.realization_mean)
    np.testing.assert_allclose(result["realization_above"], whole.realization_above)
    with pytest.raises(bt.FileError):
        tb.simulate_to_parquet(tmp_path / "missing.parquet", out)


def test_turning_bands_streamed_with_domains_equals_in_memory(model, tmp_path):
    xyz = rng.uniform(0, 100, (60, 3)) * [1, 0.75, 0.2]
    values = rng.lognormal(0, 0.5, 60)
    zone = np.where(xyz[:, 0] < 50, "west", "east")
    search = bt.Search(40.0, max_samples=12, soft=10.0)
    tb = bt.TurningBands(bt.Variogram([("spherical", 1.0, 30.0)]), bands=80, search=search)
    tb.fit(xyz, values, domains=zone)
    labels = np.where(model.centroids[:, 0] < 50, "west", "east")
    whole = tb.simulate(model, n=4, seed=3, domains=labels)
    source, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    bt.write_parquet(source, model)
    tb.simulate_to_parquet(source, out, n=4, seed=3, rows=333, domains=labels)
    np.testing.assert_array_equal(bt.read_parquet(out)["mean"], whole.mean)
    bt.write_parquet(source, model.with_columns({"zone": labels}))
    tb.simulate_to_parquet(source, out, n=4, seed=3, rows=333, domain_column="zone")
    np.testing.assert_array_equal(bt.read_parquet(out)["mean"], whole.mean)
    with pytest.raises(bt.InvalidInput, match="one of domains or domain_column"):
        tb.simulate_to_parquet(source, out, domains=labels, domain_column="zone")
    with pytest.raises(bt.MissingColumn):
        tb.simulate_to_parquet(source, out, domain_column="rock")
    with pytest.raises(TypeError):
        tb.simulate_to_parquet(source, out, 4)
    with pytest.raises(bt.InvalidInput, match="simulate_to_parquet needs domains"):
        tb.simulate_to_parquet(source, out)
    with pytest.raises(ValueError):
        tb.simulate_to_parquet(source, out, domains=labels[1:])


def test_turning_bands_streamed_blocks_average_their_nodes(model, tmp_path):
    xyz = rng.uniform(0, 100, (60, 3)) * [1, 0.75, 0.2]
    values = rng.lognormal(0, 0.5, 60)
    tb = bt.TurningBands(bt.Variogram([("spherical", 1.0, 30.0)]), bands=80, classes=3)
    tb.fit(xyz, values, trend=xyz[:, 0] / 100)
    model = model.with_column("drift", model.centroids[:, 0] / 100)
    nodes = model.discretize((2, 2, 1))
    drift = model["drift"][nodes["block"].astype(int)]
    whole = tb.simulate(nodes, n=4, seed=3, cutoffs=[1.5], blocks=model, trend=drift)
    source, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    bt.write_parquet(source, model)
    tb.simulate_to_parquet(
        source, out, n=4, seed=3, cutoffs=[1.5], rows=333, trend="drift", discretization=(2, 2, 1)
    )
    back = bt.read_parquet(out)
    np.testing.assert_array_equal(back["mean"], whole.mean)
    np.testing.assert_array_equal(back["p_above_1.5"], whole.probability_above[:, 0])
    with pytest.raises(bt.InvalidInput, match="positive"):
        tb.simulate_to_parquet(source, out, trend="drift", discretization=(2, 0, 1))
    with pytest.raises(bt.InvalidInput, match="no trend column"):
        tb.simulate_to_parquet(source, out, trend="missing")


def test_kernel_trend_beyond_its_data_must_be_filled_before_streaming(model, tmp_path):
    xyz = rng.uniform(0, 30, (60, 3)) * [1, 1, 0.2]
    values = rng.lognormal(0, 0.5, 60)
    trend, _ = bt.detrend(xyz, values, bandwidth=5.0)
    at = trend.predict(model)
    assert np.isnan(at).any() and not np.isnan(at).all()
    tb = bt.TurningBands(bt.Variogram([("spherical", 1.0, 30.0)]), bands=50).fit(
        xyz, values, trend=trend.predict(xyz)
    )
    source, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    bt.write_parquet(source, model.with_column("trend", at))
    with pytest.raises(bt.InvalidInput, match="has nulls; fill them"):
        tb.simulate_to_parquet(source, out, n=2, trend="trend")
    filled = np.where(np.isnan(at), values.mean(), at)
    bt.write_parquet(source, model.with_column("trend", filled))
    assert np.isfinite(tb.simulate_to_parquet(source, out, n=2, trend="trend")["realization_mean"]).all()


def test_turning_bands_search_defaults_to_the_nearest_32(model):
    xyz = rng.uniform(0, 100, (50, 3)) * [1, 0.75, 0.2]
    values = rng.lognormal(0, 0.5, 50)
    variogram = bt.Variogram([("spherical", 1.0, 30.0)])
    default = bt.TurningBands(variogram, bands=50).fit(xyz, values).simulate(model, n=3, seed=2)
    explicit = bt.TurningBands(
        variogram, bands=50, search=bt.Search(radius=1e9, max_samples=32, min_samples=1)
    )
    np.testing.assert_array_equal(explicit.fit(xyz, values).simulate(model, n=3, seed=2).mean, default.mean)


def test_streamed_file_holds_kept_realizations(model, tmp_path):
    xyz = rng.uniform(0, 100, (60, 3)) * [1, 0.75, 0.2]
    tb = bt.TurningBands(bt.Variogram([("spherical", 1.0, 30.0)]), bands=80).fit(
        xyz, rng.lognormal(0, 0.5, 60)
    )
    source, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    bt.write_parquet(source, model)
    tb.simulate_to_parquet(source, out, n=5, seed=3, keep=[0, 3], rows=333)
    back = bt.read_parquet(out)
    whole = tb.simulate(model, n=5, seed=3, keep=[0, 3])
    np.testing.assert_array_equal(back["realization_3"], whole.realizations[1])
    assert "realization_1" not in back.attributes.column_names
