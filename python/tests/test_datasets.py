import os
import subprocess
import sys

import boitata as bt
import numpy as np
import pytest
from boitata._boitata import _gaussian_field

d = bt.datasets
HOLES = {"HOLE_ID", "X", "Y", "Z", "LENGTH", "TYPE"}
SURVEYS = {"HOLE_ID", "DEPTH", "DIP", "AZIMUTH"}
LOADERS = {
    "clustered_metal_survey": d.clustered_metal_survey,
    "clustered_metal_survey_exhaustive": d.clustered_metal_survey_exhaustive,
    "topsoil_metals": d.topsoil_metals,
    "coal_seam_thickness": d.coal_seam_thickness,
    "soil_geochemistry_survey": d.soil_geochemistry_survey,
    "stacked_sulphide_lenses": d.stacked_sulphide_lenses,
    "stacked_sulphide_lenses_raw": lambda seed=0: d.stacked_sulphide_lenses(raw=True, seed=seed),
    "iron_formation_plateau": d.iron_formation_plateau,
    "vein_gold_grade_control": d.vein_gold_grade_control,
    "phosphate_weathering_profile": d.phosphate_weathering_profile,
    "nickel_laterite_profile": d.nickel_laterite_profile,
    "tailings_reprocessing": d.tailings_reprocessing,
    "porphyry_geometallurgy": d.porphyry_geometallurgy,
    "porphyry_geometallurgy_2": lambda seed=0: d.porphyry_geometallurgy(2, seed=seed),
    "channel_training_image": d.channel_training_image,
    "seismic_cube": d.seismic_cube,
}


def _kinds(data):
    return {k: type(v).__name__ for k, v in data.items()}


def _fingerprint(data):
    items = data.items() if isinstance(data, dict) else [("", data)]
    out = {}
    for key, value in items:
        if isinstance(value, bt.Mesh):
            out[key] = value.coords.tolist()
        elif isinstance(value, bt.Polylines):
            out[key] = len(value)
        else:
            columns = value.column_names
            out[key] = {
                c: np.nan_to_num(value[c]).tolist() if value[c].dtype.kind == "f" else value[c].tolist()
                for c in columns
            }
    return out


def test_field_is_one_truth_for_any_points():
    variogram = bt.Variogram(
        [("spherical", 0.8, 40.0)], nugget=0.2, rotation=(30.0, 0.0, 0.0), ratios=(0.5, 1.0)
    )
    points = np.random.default_rng(0).uniform(0.0, 100.0, (2000, 2))
    lo, hi = [0.0, 0.0, 0.0], [100.0, 100.0, 0.0]
    field = _gaussian_field(points, variogram, lo, hi, seed=5)
    np.testing.assert_array_equal(_gaussian_field(points[::-1], variogram, lo, hi, seed=5), field[::-1])
    np.testing.assert_array_equal(_gaussian_field(points[:7], variogram, lo, hi, seed=5), field[:7])
    assert abs(field.mean()) < 0.3 and 0.6 < field.var() < 1.4
    assert not np.array_equal(_gaussian_field(points, variogram, lo, hi, seed=6), field)


@pytest.mark.parametrize("name", LOADERS)
def test_a_seed_fixes_the_data(name):
    load = LOADERS[name]
    assert _fingerprint(load()) == _fingerprint(load(seed=0))
    assert _fingerprint(load(seed=1)) != _fingerprint(load(seed=0))


def test_data_ignore_the_thread_count():
    script = (
        "import hashlib, boitata as bt\n"
        "d = bt.datasets\n"
        "a = d.iron_formation_plateau()['assays']['FE_PCT']\n"
        "b = d.clustered_metal_survey_exhaustive()['V']\n"
        "print(hashlib.sha256(a.tobytes() + b.tobytes()).hexdigest())\n"
    )
    run = lambda threads: (
        subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "RAYON_NUM_THREADS": threads},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    assert run("1") == run("4")


def test_survey_samples_read_the_exhaustive_grid_and_are_biased_high():
    samples = d.clustered_metal_survey()
    truth = d.clustered_metal_survey_exhaustive()
    assert len(samples) == 470 and truth.num_rows == 78_000
    grid = truth["V"].reshape(300, 260)
    x, y = samples.coords[:, 0].astype(int) - 1, samples.coords[:, 1].astype(int) - 1
    np.testing.assert_array_equal(samples["V"], grid[y, x])
    assert np.isnan(samples["U"][:195]).all() and np.isfinite(samples["U"][195:]).all()
    assert samples["V"].mean() > 1.3 * truth["V"].mean()
    assert samples["V"][:195].mean() == pytest.approx(truth["V"].mean(), rel=0.2)


def test_survey_is_anisotropic_with_the_long_axis_near_north():
    samples = d.clustered_metal_survey_exhaustive()
    grid = samples["V"].reshape(300, 260)
    along = np.mean((grid[30:, :] - grid[:-30, :]) ** 2)
    across = np.mean((grid[:, 30:] - grid[:, :-30]) ** 2)
    assert along < 0.9 * across


def test_topsoil_metals_correlate_and_follow_the_rock():
    data = d.topsoil_metals()
    assert _kinds(data) == {"prediction": "PointSet", "validation": "PointSet", "grid": "PointSet"}
    assert len(data["prediction"]) == 259 and len(data["validation"]) == 100
    p = data["prediction"]
    assert np.corrcoef(np.log(p["Ni"]), np.log(p["Co"]))[0, 1] > 0.5
    assert np.corrcoef(np.log(p["Cu"]), np.log(p["Pb"]))[0, 1] > 0.5
    assert p["Co"][p["Rock"] == "Argovian"].mean() < 0.8 * p["Co"][p["Rock"] == "Kimmeridgian"].mean()


def test_coal_infill_targets_the_thick_seam_inside_the_lease():
    data = d.coal_seam_thickness()
    b = data["boreholes"]
    t = b["THICKNESS_M"]
    assert t[145:].mean() > 1.3 * t[:145].mean()
    assert np.all((b["CATEGORY"] == "MECHANIZED") == (t > 1.4))
    regional = np.polyfit(b.coords[:145, 0], t[:145], 1)[0]
    assert regional < 0
    assert data["grid"].count == [120, 90, 1] and 0.5 < data["grid"]["INSIDE"].mean() < 0.8


def test_soil_censored_values_hold_their_limit():
    s = d.soil_geochemistry_survey()["samples"]
    assert np.all(s["AU_PPB"][s["AU_BDL"] == 1] == 2.0) and np.all(s["AS_PPM"][s["AS_BDL"] == 1] == 5.0)
    assert np.all(s["AU_PPB"] >= 2.0) and np.all(s["AS_PPM"] >= 5.0)
    assert 0.2 < s["AU_BDL"].mean() < 0.55 and 0.2 < np.isnan(s["ZN_PPM"]).mean() < 0.4


@pytest.mark.parametrize(
    "load, names, extra",
    [
        (d.stacked_sulphide_lenses, ("lens_1", "lens_2", "lens_3"), "lithology"),
        (d.iron_formation_plateau, ("high_grade", "iron_formation"), "lithology"),
        (d.vein_gold_grade_control, ("vein_V1", "vein_V2", "vein_V3", "vein_V4"), "lithology"),
    ],
)
def test_solids_are_closed_and_holes_are_complete(load, names, extra):
    data = load()
    for name in names:
        mesh = data[name]
        assert mesh.volume > 0
        assert mesh.repair().volume == pytest.approx(mesh.volume, rel=1e-9)
    assert set(data["collars"].column_names) == HOLES and set(data["surveys"].column_names) == SURVEYS
    ids = set(data["collars"]["HOLE_ID"].tolist())
    assert len(ids) == data["collars"].num_rows
    for table in ("surveys", "assays", extra):
        assert set(data[table]["HOLE_ID"].tolist()) <= ids
    assert np.all(data["assays"]["TO"] > data["assays"]["FROM"])


def test_lens_grades_follow_the_logs():
    data = d.stacked_sulphide_lenses()
    a = data["assays"]
    rc = np.array([h.startswith("RC") for h in a["HOLE_ID"].tolist()])
    au = a["AU_GPT"]
    filled = np.isfinite(np.asarray(au, dtype=float))
    assert not filled[rc].any() and filled[~rc].all()
    density = a["DENSITY"]
    known = np.isfinite(density)
    assert 0.05 < known.mean() < 0.2
    assert np.corrcoef(density[known], a["ZN_PCT"][known])[0, 1] > 0.7


def test_raw_tables_hold_the_planted_errors():
    clean = d.stacked_sulphide_lenses()
    raw = d.stacked_sulphide_lenses(raw=True)
    assert set(raw) == {"collars", "surveys", "assays", "lithology"}
    ids = raw["collars"]["HOLE_ID"].tolist()
    assert len(ids) == clean["collars"].num_rows + 1 and len(set(ids)) < len(ids)
    assert any(h != h.strip() for h in ids)
    zn = raw["assays"]["ZN_PCT"].tolist()
    assert "-999" in zn and "NS" in zn
    assert "<0.01" in raw["assays"]["CU_PCT"].tolist()
    assert any(h != h.upper() for h in raw["assays"]["HOLE_ID"].tolist())
    assert any(h != h.strip() for h in raw["lithology"]["HOLE_ID"].tolist())
    assert (raw["assays"]["FROM"] > raw["assays"]["TO"]).sum() == 1
    assert (np.asarray(raw["surveys"]["DIP"], dtype=float) < 0).any()


def test_iron_formation_closes_and_blastholes_average_their_bench():
    data = d.iron_formation_plateau()
    a = data["assays"]
    total = (
        a["FE_PCT"] * 1.4297
        + a["SIO2_PCT"]
        + a["AL2O3_PCT"]
        + a["P_PCT"] * 2.2914
        + a["MN_PCT"] * 1.2912
        + a["LOI_PCT"]
    )
    assert total.max() <= 100.5
    b = data["blastholes"]
    np.testing.assert_array_equal(b.coords[:, 2], b["Z_TOP"] - 6.0)
    assert b["FE_PCT"].std() < a["FE_PCT"].std()
    model = data["block_model"]
    assert model.count == [89, 81, 33] and model.index is not None
    assert set(model["LITH"].tolist()) >= {"HF", "IC", "MAF"}


def test_vein_gold_is_extremely_skewed_in_the_vein():
    data = d.vein_gold_grade_control()
    au = data["assays"]["AU_GPT"]
    assert au.std() > 3 * au.mean() and au.max() > 100 * np.median(au)
    collars = data["collars"]
    channels = collars["TYPE"] == "CH"
    assert channels.sum() > 1500 and (~channels).sum() == 118
    assert np.median(collars["LENGTH"][channels]) < 10.0


def test_phosphate_chemistry_limits():
    a = d.phosphate_weathering_profile()["assays"]
    assert np.all(a["P2O5_AP_PCT"] <= a["CAO_PCT"] / 1.318 + 0.011)
    oxides = (
        a["P2O5_PCT"]
        + a["CAO_PCT"]
        + a["SIO2_PCT"]
        + a["FE2O3_PCT"]
        + a["AL2O3_PCT"]
        + np.nan_to_num(a["MGO_PCT"])
    )
    assert oxides.max() <= 95.5


def test_nickel_is_richer_at_the_saprolite_base_and_cobalt_in_the_limonite():
    data = d.nickel_laterite_profile()
    h, a = data["horizons"], data["assays"]
    assert set(h["HORIZON"].tolist()) == {"FERR", "LIM", "SAP", "BRK"}
    limits = {}
    for hole, start, end, name in zip(h["HOLE_ID"].tolist(), h["FROM"], h["TO"], h["HORIZON"].tolist()):
        limits[hole, name] = (start, end)
    upper, lower, lim, sap = [], [], [], []
    for hole, start, end, ni, co in zip(a["HOLE_ID"].tolist(), a["FROM"], a["TO"], a["NI_PCT"], a["CO_PCT"]):
        mid = (start + end) / 2
        if (hole, "LIM") in limits and limits[hole, "LIM"][0] <= mid < limits[hole, "LIM"][1]:
            lim.append(co)
        if limits[hole, "SAP"][0] <= mid < limits[hole, "SAP"][1]:
            sap.append(co)
            top, base = limits[hole, "SAP"]
            (lower if mid > (top + base) / 2 else upper).append(ni)
    assert np.mean(lower) > 1.2 * np.mean(upper) and np.mean(lim) > 2 * np.mean(sap)


def test_tailings_slimes_dominate_toward_the_pond_and_cyanide_is_censored():
    data = d.tailings_reprocessing()
    assert data["surface"].count == [81, 56, 1] and data["original_ground"].count == [81, 56, 1]
    a = data["assays"]
    assert np.all(a["CN_WAD_PPM"][a["CN_WAD_BDL"] == 1] == 0.5)
    assert 0.3 < a["CN_WAD_BDL"].mean() < 0.6
    y = dict(zip(data["collars"]["HOLE_ID"].tolist(), data["collars"]["Y"]))
    l = data["lithology"]
    south = np.array([y[h] < 2225.0 for h in l["HOLE_ID"].tolist()])
    slime = l["LITH"] == "SLIME"
    assert slime[south].mean() > slime[~south].mean()


def test_porphyry_deposits():
    with pytest.raises(bt.InvalidInput):
        d.porphyry_geometallurgy(4)
    first = d.porphyry_geometallurgy()
    assert set(first) == {
        "block_model",
        "correlations",
        "grindability_distribution",
        "mineralogical_distributions",
        "pseudo_drillholes",
        "synthetic_drillholes",
    }
    assert first["block_model"].count == [49, 71, 44] and np.all(first["block_model"]["ton"] > 0)
    assert np.isnan(first["grindability_distribution"]["bwi"]).sum() > 0
    for k in (2, 3):
        later = d.porphyry_geometallurgy(k)
        assert set(later) == {"drillholes", "topography"}
        assert later["topography"].count == [97, 142, 1]
        assert {"tenantite", "molybdenite", "recovery"} <= set(later["drillholes"].column_names)


def test_channel_image_has_connected_channels_along_y():
    image = d.channel_training_image()
    assert image.count == [250, 250, 1] and image.origin[:2] == [0.0, 0.0] and image.index is None
    assert set(np.unique(image["facies"])) == {0.0, 1.0}
    assert image["facies"].mean() == pytest.approx(0.28, abs=0.03)
    grid = image["facies"].reshape(250, 250)
    assert np.mean(grid[20:, :] != grid[:-20, :]) < np.mean(grid[:, 20:] != grid[:, :-20])


def test_seismic_cube_survives_segy(tmp_path):
    cube = d.seismic_cube()
    assert cube.count == [45, 45, 51] and cube.crs == "EPSG:23031"
    assert cube.size == [25.0, 25.0, 4.0] and cube.rotation[0] == pytest.approx(88.4, abs=0.05)
    assert np.isfinite(cube["amplitude"]).all() and cube["amplitude"].std() > 1000
    path = tmp_path / "cube.sgy"
    bt.write_segy(path, cube, "amplitude")
    back = bt.read_segy(path)
    assert back.count == cube.count
    np.testing.assert_allclose(back["amplitude"], cube["amplitude"], rtol=1e-5, atol=1e-2)
