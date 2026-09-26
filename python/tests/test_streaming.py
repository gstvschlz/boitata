import ceres as cs
import numpy as np
import pytest

rng = np.random.default_rng(11)


@pytest.fixture
def model():
    bm = cs.BlockModel(origin=(0, 0, 0), size=(5, 5, 5), count=(20, 15, 4))
    return bm.with_column("grade", rng.lognormal(0, 1, len(bm)))


def test_chunks_cover_the_model_and_stay_bounded(model, tmp_path):
    path = tmp_path / "model.parquet"
    cs.write_parquet(path, model)
    file = cs.BlockModelFile(path)
    assert len(file) == 1200 and file.count == [20, 15, 4] and file.column_names == ["grade"]
    chunks = list(file.chunks(rows=250))
    assert [len(c) for c in chunks] == [250, 250, 250, 250, 200]
    np.testing.assert_array_equal(np.concatenate([c.centroids for c in chunks]), model.centroids)
    np.testing.assert_array_equal(np.concatenate([c["grade"] for c in chunks]), model["grade"])


def test_map_blocks_kriging_matches_in_memory(model, tmp_path):
    xy = rng.uniform(0, 100, (80, 3)) * [1, 0.75, 0.2]
    v = rng.normal(size=80)
    ok = cs.OrdinaryKriging(cs.Variogram([("spherical", 1.0, 40.0)]), cs.Search(radius=60)).fit(xy, v)
    masked = model.mask(np.arange(len(model)) % 3 != 0)
    source, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    cs.write_parquet(source, masked)
    sizes = []

    def estimate(chunk):
        sizes.append(len(chunk))
        d = ok.predict(chunk, diagnostics=True)
        return {"estimate": d["value"], "class": np.where(d["slope"] > 0.9, "high", "low")}

    cs.map_blocks(source, out, estimate, rows=117)
    assert max(sizes) <= 117 and sum(sizes) == len(masked)
    back = cs.read_parquet(out)
    np.testing.assert_array_equal(back.index, masked.index)
    np.testing.assert_allclose(back["estimate"], ok.predict(masked))
    np.testing.assert_array_equal(back["grade"], masked["grade"])
    assert set(back["class"]) <= {"high", "low"}


def test_map_blocks_rejects_wrong_lengths(model, tmp_path):
    path = tmp_path / "model.parquet"
    cs.write_parquet(path, model)
    with pytest.raises(cs.InvalidInput, match="rows"):
        cs.map_blocks(path, tmp_path / "out.parquet", lambda chunk: {"x": np.zeros(3)}, rows=100)


def test_turning_bands_streamed_equals_in_memory(model, tmp_path):
    xyz = rng.uniform(0, 100, (60, 3)) * [1, 0.75, 0.2]
    values = rng.lognormal(0, 0.5, 60)
    tb = cs.TurningBands(cs.Variogram([("spherical", 1.0, 30.0)]), bands=80).fit(xyz, values)
    whole = tb.simulate(model, n=8, seed=3, cutoffs=[1.5], quantiles=[0.1, 0.9])
    source, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    cs.write_parquet(source, model)
    result = tb.simulate_to_parquet(source, out, n=8, seed=3, cutoffs=[1.5], quantiles=[0.1, 0.9], rows=333)
    back = cs.read_parquet(out)
    np.testing.assert_array_equal(back["mean"], whole.mean)
    np.testing.assert_array_equal(back["p_above_1.5"], whole.probability_above[0])
    np.testing.assert_array_equal(back["q0.9"], whole.quantile_values[1])
    np.testing.assert_array_equal(back["grade"], model["grade"])
    np.testing.assert_allclose(result["realization_mean"], whole.realization_mean)
    np.testing.assert_allclose(result["realization_above"], whole.realization_above)
    with pytest.raises(cs.FileError):
        tb.simulate_to_parquet(tmp_path / "missing.parquet", out)
