import boitata as bt
import numpy as np
import pytest
from boitata import _display

rng = np.random.default_rng(0)
coords = rng.uniform(0, 100, (60, 3))
values = rng.lognormal(0, 1, 60)
categories = (values > 0.8).astype(int) + (values > 1.5)
model = bt.Variogram([("spherical", 1.0, 40.0)], nugget=0.1, rotation=(30, 0, 0), ratios=(0.5, 0.5))
search = bt.Search(60.0, max_samples=12)
grid = bt.BlockModel((0, 0, 0), (10, 10, 10), (5, 5, 2))
points = bt.PointSet(coords, {"v": values, "w": values**2, "u": values + coords[:, 0]})


@pytest.fixture(scope="module")
def shown(tmp_path_factory):
    anam = bt.HermiteAnamorphosis(degree=10).fit(values)
    lmc = bt.Coregionalization(
        [[0.1, 0.0], [0.0, 0.2]], structures=[("spherical", 40.0, [[1.0, 0.5], [0.5, 1.0]])]
    )
    mik = bt.MultipleIndicatorKriging(model, search, [0.8, 1.5]).fit(coords, values)
    cik = bt.CategoricalIndicatorKriging(model, search).fit(coords, categories)
    sis = bt.SIS([model] * 3, search).fit(coords, categories)
    sgs = bt.SGS(model, search).fit(points, "v")
    pair = np.c_[values, values**2]
    path = tmp_path_factory.mktemp("display") / "model.parquet"
    bt.write_parquet(path, grid.with_column("a", np.arange(50.0)))
    pipeline = bt.Pipeline(
        [
            ("names", bt.RenameColumns(case="snake")),
            ("cap", bt.Capping(cap=6.0), "v"),
            ("ns", bt.NormalScore(), "v"),
        ]
    )
    return {
        "DualKriging": bt.DualKriging(model).fit(coords, values),
        "Cokriging": bt.Cokriging(lmc, search).fit(coords, values, np.arange(60) % 2),
        "DisjunctiveKriging": bt.DisjunctiveKriging(anam, model, search),
        "MultipleIndicatorKriging": mik,
        "MultigaussianKriging": bt.MultigaussianKriging(model, [search, search]),
        "CategoricalIndicatorKriging": cik,
        "SIS": sis,
        "Plurigaussian": bt.Plurigaussian(model, proportions=[0.4, 0.4, 0.2]),
        "NormalScore": bt.NormalScore(tails=(0.0, 50.0)).fit(values),
        "Capping": bt.Capping(quantile=0.95).fit(values),
        "BoxCox": bt.BoxCox().fit(values),
        "HermiteAnamorphosis": anam,
        "KernelDensity": bt.KernelDensity(bandwidth=0.3, log=True).fit(values),
        "GaussianMixture": bt.GaussianMixture(components=2).fit(pair),
        "PCA": bt.PCA(standardize=True).fit(pair),
        "MAF": bt.MAF(10.0),
        "PPMT": bt.PPMT(seed=1, iterations=5),
        "StepwiseConditional": bt.StepwiseConditional(),
        "GaussianImputer": bt.GaussianImputer(),
        "Declustering": bt.cell_declustering(coords, values),
        "Categories": bt.Categories(["ore", "waste"], mapping={"OX": "ore"}, other="waste"),
        "LocalGrid": bt.LocalGrid(scale=0.9996, rotation=12.5, shift=(1000.0, 2000.0), crs="EPSG:32723"),
        "Coregionalization": lmc,
        "Transiogram": bt.Transiogram([0.5, 0.5], 10.0),
        "UniformConditioning": bt.UniformConditioning(anam, 0.8),
        "Trend": bt.detrend(coords, values, degree=2)[0],
        "Domains": bt.domains(points, ["v", "w", "u"], 2, realizations=10, seed=1),
        "MeshReport": bt.Mesh(np.eye(3), [[0, 1, 2]]).validate(),
        "BlockModelFile": bt.BlockModelFile(path),
        "LocalAnisotropy": bt.LocalAnisotropy(
            coords[:5], np.tile([35.0, 0.0, 0.0], (5, 1)), np.tile([0.3, 1.0], (5, 1))
        ),
        "IndicatorSummary": mik.predict(grid),
        "CategoricalIndicatorSummary": cik.predict(grid),
        "CategoricalSummary": sis.simulate(grid, n=3),
        "VariogramMap": bt.variogram_map(coords, values, 10.0, 50.0),
        "VariogramVolume": bt.variogram_volume(coords, values, 10.0, 50.0),
        "Clip": bt.Clip(lower=0.0),
        "CLR": bt.CLR(parts=["a", "b"]),
        "Pipeline": pipeline,
        "RealizationCheck": bt.check_realizations(grid, sgs.simulate(grid, n=5, keep=True), points, "v"),
        "Topography": bt.topography(coords, cell=20.0),
    }


def test_every_object_has_a_line_a_card_and_html(shown):
    for title, obj in shown.items():
        line, card, html = repr(obj), str(obj), obj._repr_html_()
        assert line.startswith(f"{title}(") and "\n" not in line and len(line) < 200
        assert card.splitlines()[0].split("  ")[0] == title
        assert f"<b>{title}</b>" in html and f"{obj}" == line
        for cell in card.replace("\n", "  ").split("  "):
            assert _display.escape(cell.strip()) in html


def test_estimators_and_transforms_say_whether_they_are_fitted(shown):
    for title in (
        "DualKriging",
        "Cokriging",
        "MultipleIndicatorKriging",
        "CategoricalIndicatorKriging",
        "SIS",
    ):
        assert "fitted on 60 samples" in repr(shown[title]) and "fitted on 60 samples" in str(shown[title])
    for title in (
        "DisjunctiveKriging",
        "MultigaussianKriging",
        "Plurigaussian",
        "MAF",
        "PPMT",
        "GaussianImputer",
    ):
        assert "not fitted" in repr(shown[title]) and "not fitted" in str(shown[title])
    for title in ("NormalScore", "Capping", "BoxCox", "HermiteAnamorphosis", "KernelDensity", "PCA"):
        assert repr(shown[title]).startswith(f"{title}(fitted")


def test_a_transform_line_gives_only_the_options_that_were_set():
    assert repr(bt.NormalScore()) == "NormalScore(not fitted)"
    assert repr(bt.KernelDensity()) == "KernelDensity(not fitted)"
    assert repr(bt.GaussianImputer()) == "GaussianImputer(not fitted)"
    assert repr(bt.PPMT(seed=1, iterations=5)) == "PPMT(not fitted, iterations=5, seed=1)"
    assert repr(bt.Capping(cap=6.0)) == "Capping(not fitted, cap=6.0)"
    assert repr(bt.RenameColumns(case="snake")) == "RenameColumns(case='snake')"


def test_a_fitted_transform_card_shows_what_it_learned(shown):
    capping = shown["Capping"]
    assert str(capping) == (
        "Capping  fitted\n"
        "  quantile       0.95\n"
        f"  caps           {_display._number(capping.caps_)}\n"
        f"  metal removed  {_display._number(capping.metal_removed_)}"
    )


def test_estimator_card_lists_variogram_search_and_options(shown):
    assert str(shown["MultipleIndicatorKriging"]) == (
        "MultipleIndicatorKriging  fitted on 60 samples\n"
        "  variogram      nugget 0.1 + spherical(1, 40)\n"
        "  anisotropy     azimuth 30°, dip 0°, rake 0°; semi/major 0.5, minor/major 0.5\n"
        "  search         radius 60, 1 to 12 samples\n"
        "  thresholds     0.8, 1.5\n"
        "  interpolation  global"
    )
    assert "2 search passes" in repr(shown["MultigaussianKriging"])
    assert "3 variograms" in repr(shown["SIS"]) and "variogram 3" in str(shown["SIS"])


@pytest.mark.parametrize(
    "obj",
    [
        bt.Categories(["a", "b"]),
        bt.Categories(["ore", "waste"], colors=["#ff0000", "#888888"], mapping={"OX": "ore"}, other="waste"),
        bt.LocalGrid(),
        bt.LocalGrid(scale=2.0, rotation=12.5, shift=(1.0, 2.0), z_shift=3.0, crs="EPSG:32723"),
        bt.Transiogram([0.5, 0.5], 10.0),
    ],
)
def test_the_line_of_a_value_object_builds_it_back(obj):
    again = eval(repr(obj), vars(bt))
    assert repr(again) == repr(obj) and again.to_json() == obj.to_json()


def test_categories_card_lists_codes_colors_and_merged_labels():
    scheme = bt.Categories(
        ["ore", "waste"], colors=["#ff0000", "#888888"], mapping={"OX": "ore", "SU": "ore"}
    )
    assert str(scheme) == (
        "Categories  2 categories\n"
        "  code  name   color    from\n"
        "     0  ore    #ff0000  OX, SU\n"
        "     1  waste  #888888"
    )


def test_pipeline_card_lists_its_steps(shown):
    pipeline = shown["Pipeline"]
    assert repr(pipeline) == "Pipeline(not fitted, names, cap, ns)"
    assert str(pipeline) == (
        "Pipeline  3 steps, not fitted\n"
        "  step   transform                     columns\n"
        "  names  RenameColumns(case='snake')\n"
        "  cap    Capping(not fitted, cap=6.0)  v\n"
        "  ns     NormalScore(not fitted)       v"
    )
    assert repr(pipeline.fit(points)).startswith("Pipeline(fitted")


def test_showing_a_fitted_object_does_not_copy_its_arrays(shown, monkeypatch):
    for title in ("DualKriging", "MultipleIndicatorKriging", "SIS", "NormalScore", "Capping"):
        obj = shown[title]
        for costly in ("to_json", "_state"):
            if hasattr(type(obj), costly):
                monkeypatch.setattr(type(obj), costly, None)
        assert repr(obj) and str(obj)
