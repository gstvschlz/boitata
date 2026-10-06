import hashlib
import io
import re

import boitata as bt
import numpy as np
import pytest

CSV = b"X,Y,V\n1,2,3\n"
SAMPLE = "mining/2d/walker-lake/sample.csv"


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("BOITATA_DATA", str(tmp_path))
    monkeypatch.setitem(bt.datasets.FILES, SAMPLE, hashlib.sha256(CSV).hexdigest())
    return tmp_path


def test_fetch_checks_the_hash_and_caches(offline, monkeypatch):
    monkeypatch.setattr(bt.datasets.urllib.request, "urlopen", lambda url: io.BytesIO(CSV))
    path = bt.datasets.fetch(SAMPLE)
    assert path.read_bytes() == CSV and path.is_relative_to(offline)

    def no_network(url):
        raise AssertionError("downloaded twice")

    monkeypatch.setattr(bt.datasets.urllib.request, "urlopen", no_network)
    assert len(bt.datasets.walker_lake()) == 1


def test_corrupt_download_is_an_error_and_leaves_nothing(offline, monkeypatch):
    monkeypatch.setattr(bt.datasets.urllib.request, "urlopen", lambda url: io.BytesIO(b"tampered"))
    with pytest.raises(bt.FileError, match="SHA-256"):
        bt.datasets.fetch(SAMPLE)
    assert not any(p.is_file() for p in offline.rglob("*"))


def test_files_come_from_the_pinned_commit(offline, monkeypatch):
    urls = []
    monkeypatch.setattr(
        bt.datasets.urllib.request, "urlopen", lambda url: urls.append(url) or io.BytesIO(CSV)
    )
    bt.datasets.fetch(SAMPLE)
    assert urls == [f"{bt.datasets.REPO}/{bt.datasets.COMMIT}/{SAMPLE}"]


def test_registry_paths_and_hashes():
    files = bt.datasets.FILES
    assert all(re.fullmatch(r"(mining|oil-gas)/(2d|3d)/[a-z0-9-]+/[\w/-]+\.(csv|stl|sgy)", k) for k in files)
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in files.values())
    folders = {k.split("/")[2] for k in files}
    assert {f.replace("-", "_") for f in folders} <= set(bt.datasets.__all__)
    assert sum(k.startswith("mining/3d/stacked-sulphide-lenses/raw/") for k in files) == 4


def test_unknown_file():
    with pytest.raises(bt.InvalidInput):
        bt.datasets.fetch("nope.csv")


def test_grid_infers_geometry_and_orders_x_fastest():
    x, y = np.meshgrid([15.0, 5.0, 25.0], [100.0, 120.0], indexing="ij")
    table = bt.Table({"X": x.ravel(), "Y": y.ravel(), "Z": x.ravel() + y.ravel()})
    grid = bt.datasets._grid(table)
    assert grid.origin[:2] == [0.0, 90.0] and grid.size[:2] == [10.0, 20.0] and grid.count == [3, 2, 1]
    assert grid.index is None
    np.testing.assert_array_equal(grid.coords[:, 0], [5, 15, 25, 5, 15, 25])
    np.testing.assert_array_equal(grid["Z"], grid.coords[:, 0] + grid.coords[:, 1])


def test_grid_masks_missing_cells_in_3d():
    table = bt.Table({"X": [0.0, 2.0, 0.0], "Y": [0.0, 0.0, 1.0], "Z": [5.0, 5.0, 8.0], "V": [1.0, 2.0, 3.0]})
    grid = bt.datasets._grid(table, ("X", "Y", "Z"))
    assert grid.count == [2, 2, 2] and grid.size == [2.0, 1.0, 3.0]
    assert grid.index.tolist() == [0, 1, 6]
    np.testing.assert_array_equal(grid["V"], [1.0, 2.0, 3.0])


def test_porphyry_deposit_is_checked():
    with pytest.raises(bt.InvalidInput):
        bt.datasets.porphyry_geometallurgy(4)


def _shape(data):
    return {k: (type(v).__name__, None if isinstance(v, bt.Mesh) else len(v)) for k, v in data.items()}


@pytest.mark.slow
def test_download_walker_lake():
    assert len(bt.datasets.walker_lake()) == 470
    assert bt.datasets.walker_lake_exhaustive().num_rows == 78_000
    assert [len(p) for p in bt.datasets.jura().values()] == [259, 100, 5957]


@pytest.mark.slow
@pytest.mark.parametrize(
    "load, expected",
    [
        (
            bt.datasets.coal_seam_thickness,
            {"boreholes": ("PointSet", 295), "boundary": ("Polylines", 1), "grid": ("BlockModel", 10_800)},
        ),
        (
            bt.datasets.soil_geochemistry_survey,
            {"samples": ("PointSet", 1227), "boundary": ("Polylines", 1), "covariates": ("BlockModel", 9600)},
        ),
        (
            bt.datasets.stacked_sulphide_lenses,
            {
                "collars": ("Table", 289),
                "surveys": ("Table", 4348),
                "assays": ("Table", 16_995),
                "lithology": ("Table", 1726),
                "topography": ("BlockModel", 18_471),
                "lens_1": ("Mesh", None),
                "lens_2": ("Mesh", None),
                "lens_3": ("Mesh", None),
            },
        ),
        (
            lambda: bt.datasets.stacked_sulphide_lenses(raw=True),
            {
                "collars": ("Table", 290),
                "surveys": ("Table", 4326),
                "assays": ("Table", 16_907),
                "lithology": ("Table", 1726),
            },
        ),
        (
            bt.datasets.iron_formation_plateau,
            {
                "collars": ("Table", 187),
                "surveys": ("Table", 1088),
                "assays": ("Table", 17_475),
                "lithology": ("Table", 7573),
                "blastholes": ("PointSet", 1953),
                "block_model": ("BlockModel", 129_278),
                "topography": ("BlockModel", 14_641),
                "high_grade": ("Mesh", None),
                "iron_formation": ("Mesh", None),
            },
        ),
        (
            bt.datasets.vein_gold_grade_control,
            {
                "collars": ("Table", 2610),
                "surveys": ("Table", 6736),
                "assays": ("Table", 39_799),
                "lithology": ("Table", 10_954),
                "topography": ("BlockModel", 5041),
                **{f"vein_V{i}": ("Mesh", None) for i in range(1, 5)},
            },
        ),
        (
            bt.datasets.phosphate_weathering_profile,
            {
                "collars": ("Table", 251),
                "surveys": ("Table", 858),
                "assays": ("Table", 5039),
                "horizons": ("Table", 1255),
                "topography": ("BlockModel", 46_031),
            },
        ),
        (
            bt.datasets.nickel_laterite_profile,
            {
                "collars": ("Table", 448),
                "surveys": ("Table", 896),
                "assays": ("Table", 10_810),
                "horizons": ("Table", 1773),
            },
        ),
        (
            bt.datasets.tailings_reprocessing,
            {
                "collars": ("Table", 107),
                "surveys": ("Table", 214),
                "assays": ("Table", 4129),
                "lithology": ("Table", 3724),
                "surface": ("BlockModel", 4536),
                "original_ground": ("BlockModel", 4536),
            },
        ),
        (
            bt.datasets.porphyry_geometallurgy,
            {
                "block_model": ("BlockModel", 147_231),
                "correlations": ("Table", 7),
                "grindability_distribution": ("Table", 1366),
                "mineralogical_distributions": ("Table", 741),
                "pseudo_drillholes": ("PointSet", 3101),
                "synthetic_drillholes": ("PointSet", 6817),
            },
        ),
        (
            lambda: bt.datasets.porphyry_geometallurgy(2),
            {"drillholes": ("PointSet", 4597), "topography": ("BlockModel", 13_774)},
        ),
        (
            lambda: bt.datasets.porphyry_geometallurgy(3),
            {"drillholes": ("PointSet", 3467), "topography": ("BlockModel", 13_774)},
        ),
    ],
)
def test_download_dataset(load, expected):
    assert _shape(load()) == expected


@pytest.mark.slow
def test_dataset_details():
    iron = bt.datasets.iron_formation_plateau()
    assert iron["block_model"].count == [89, 81, 33] and iron["block_model"].index is not None
    np.testing.assert_array_equal(iron["blastholes"].coords[:, 2], iron["blastholes"]["Z_TOP"] - 6.0)
    assert np.isnan(bt.datasets.porphyry_geometallurgy()["grindability_distribution"]["bwi"]).sum() > 0
    assert "-999" in bt.datasets.stacked_sulphide_lenses(raw=True)["assays"]["ZN_PCT"]
    high_grade = iron["high_grade"]
    assert high_grade.repair().volume == pytest.approx(high_grade.volume, rel=1e-9)


@pytest.mark.slow
def test_download_strebelle():
    ti = bt.datasets.strebelle()
    assert ti.count == [250, 250, 1] and ti.origin[:2] == [0.0, 0.0] and ti.size[:2] == [1.0, 1.0]
    assert ti.index is None and set(np.unique(ti["facies"])) == {0.0, 1.0}
    assert ti["facies"].mean() == pytest.approx(0.28, abs=0.01)


@pytest.mark.slow
def test_download_f3_seismic():
    cube = bt.datasets.f3_seismic()
    assert cube.count == [45, 45, 51] and cube.crs == "EPSG:23031"
    assert cube.size == pytest.approx([25.0, 25.0, 4.0], abs=1e-3)
    assert cube.rotation[0] == pytest.approx(88.4, abs=0.05)
    assert np.isfinite(cube["amplitude"]).all()
