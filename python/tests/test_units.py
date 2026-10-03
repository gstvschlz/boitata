import boitata as bt
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

points = bt.PointSet(np.zeros((3, 3)), {"au": [1.0, np.nan, 34.285714], "d": [2.7, 2.8, 2.9]}).with_units(
    {"au": "g/t", "d": "t/m3"}
)


def test_units_convert_and_round_trip(tmp_path):
    assert points.units == {"au": "g/t", "d": "t/m3"}
    oz = points.convert_units("au", to="oz/t")
    np.testing.assert_allclose(oz["au"], [1 / 34.285714, np.nan, 1.0], rtol=1e-6)
    assert oz.units["au"] == "oz/t"
    back = oz.convert_units("au", to="ppb").convert_units("au", to="g/t")
    np.testing.assert_allclose(back["au"], points["au"])
    assert points.convert_units("d", to="g/cm3")["d"].tolist() == points["d"].tolist()
    path = tmp_path / "p.parquet"
    bt.write_parquet(path, points)
    assert bt.read_parquet(path).units == points.units
    assert points.attributes.units == points.units


def test_units_are_dropped_with_replaced_values_and_refused_across_kinds():
    assert points.with_column("au", [1.0, 2.0, 3.0]).units == {"d": "t/m3"}
    assert points.with_units({"d": None}).units == {"au": "g/t"}
    with pytest.raises(bt.InvalidInput, match="cannot convert grade"):
        points.convert_units("au", to="m")
    with pytest.raises(bt.InvalidInput, match="unknown unit"):
        points.with_units({"au": "dwt"}).convert_units("au", to="ppm")
    with pytest.raises(bt.InvalidInput, match="has no unit"):
        points.with_units({"au": None}).convert_units("au", to="ppm")


def test_plots_label_axes_with_units():
    _, ax = bt.plot.histogram("au", data=points)
    assert ax.get_xlabel() == "au (g/t)"
    _, ax = bt.plot.scatter("au", "d", data=points)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("au (g/t)", "d (t/m3)")
    _, ax = bt.plot.histogram([1.0, 2.0])
    assert ax.get_xlabel() == ""
