import sqlite3
import struct

import boitata as bt
import numpy as np
import pytest


def points():
    return bt.PointSet(
        [[500000.5, 7000000.25, 350.0], [500010.0, 7000020.0, -12.5], [500020.0, 7000040.0, 0.0]],
        {
            "au": [0.3, float("nan"), -999.0],
            "rock": ["óxido", None, "fresh"],
            "ok": np.array([True, False, True]),
        },
        crs="EPSG:31982",
    )


def test_points_round_trip(tmp_path):
    path = tmp_path / "samples.gpkg"
    bt.write_geopackage(path, points())
    back = bt.read_geopackage(path)
    np.testing.assert_array_equal(back.coords, points().coords)
    np.testing.assert_array_equal(back["au"], [0.3, np.nan, np.nan])
    assert list(back["rock"]) == ["óxido", None, "fresh"]
    assert back["ok"].tolist() == [1.0, 0.0, 1.0] and back.crs == "EPSG:31982"
    assert bt.read_geopackage(path, nodata=[])["au"][2] == -999.0


def test_polylines_round_trip_and_layers(tmp_path):
    outer = [[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]]
    hole = [[4, 4, 5], [4, 6, 5], [6, 6, 5], [6, 4, 5]]
    pits = bt.Polylines(
        [outer, hole, [[20, 0, 1], [22, 0, 1], [21, 2, 1]]],
        closed=True,
        features=[0, 0, 2],
        attributes={"name": ["pit", "empty", "dump"]},
        crs='PROJCS["SIRGAS 2000 / UTM zone 22S"]',
    )
    lines = bt.Polylines([[[0, 0], [5, 5], [9, 1]]], attributes={"id": [7.0]})
    path = tmp_path / "mine.gpkg"
    bt.write_geopackage(path, pits, layer="pits")
    bt.write_geopackage(path, lines, layer="lines")
    back = bt.read_geopackage(path, layer="pits")
    assert back.feature.tolist() == [0, 0, 2] and back.closed.all() and len(back) == 3
    assert back.crs == pits.crs and list(back["name"]) == ["pit", "empty", "dump"]
    np.testing.assert_allclose(back.area(), pits.area())
    np.testing.assert_array_equal(np.sort(back.coords, 0), np.sort(pits.coords, 0))
    other = bt.read_geopackage(path, layer="lines")
    np.testing.assert_array_equal(other.coords, lines.coords)
    assert not other.closed.any() and other.crs is None
    bt.write_geopackage(path, points(), layer="lines")
    assert isinstance(bt.read_geopackage(path, layer="lines"), bt.PointSet)
    with pytest.raises(bt.InvalidInput, match="lines, pits|pits, lines"):
        bt.read_geopackage(path)
    with pytest.raises(bt.InvalidInput):
        bt.write_geopackage(path, bt.Polylines([outer, [[0, 0], [1, 1]]], closed=[True, False]))
    with pytest.raises(bt.InvalidInput):
        bt.write_geopackage(path, points().with_column("fid", [1, 2, 3]))
    with pytest.raises(bt.FileError):
        bt.read_geopackage(tmp_path / "missing.gpkg")


def test_written_file_is_a_valid_geopackage(tmp_path):
    path = tmp_path / "samples.gpkg"
    bt.write_geopackage(path, points())
    con = sqlite3.connect(path)
    assert con.execute("PRAGMA application_id").fetchone()[0] == 0x47504B47
    assert con.execute("PRAGMA user_version").fetchone()[0] == 10400
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    srs = {row[0]: row[1] for row in con.execute("SELECT srs_id, organization FROM gpkg_spatial_ref_sys")}
    assert srs == {-1: "NONE", 0: "NONE", 4326: "EPSG", 31982: "EPSG"}
    assert con.execute("SELECT table_name, data_type, srs_id, min_x FROM gpkg_contents").fetchall() == [
        ("samples", "features", 31982, 500000.5)
    ]
    assert con.execute("SELECT * FROM gpkg_geometry_columns").fetchall() == [
        ("samples", "geom", "POINT", 31982, 1, 0)
    ]
    info = {row[1]: (row[2], row[5]) for row in con.execute("PRAGMA table_info(samples)")}
    assert info == {
        "fid": ("INTEGER", 1),
        "geom": ("POINT", 0),
        "au": ("DOUBLE", 0),
        "rock": ("TEXT", 0),
        "ok": ("DOUBLE", 0),
    }
    blob = con.execute("SELECT geom FROM samples WHERE fid = 1").fetchone()[0]
    assert blob[:4] == b"GP\x00\x01" and struct.unpack("<i", blob[4:8])[0] == 31982
    assert struct.unpack("<BI3d", blob[8:]) == (1, 1001, 500000.5, 7000000.25, 350.0)
    con.close()


def test_reads_2d_multipoints_and_skips_null_geometry(tmp_path):
    path = tmp_path / "gps.gpkg"
    bt.write_geopackage(path, points().filter(np.array([True, False, False])))
    con = sqlite3.connect(path)
    wkb = (
        struct.pack("<BII", 1, 4, 2)
        + struct.pack("<BI2d", 1, 1, 1.0, 2.0)
        + struct.pack("<BI2d", 1, 1, 3.0, 4.0)
    )
    header = b"GP\x00\x01" + struct.pack("<i", 31982)
    con.execute("INSERT INTO gps (geom, au, rock) VALUES (?, 'NA', 'x')", (header + wkb,))
    con.execute("INSERT INTO gps (geom, au, rock) VALUES (NULL, 1, 'y')")
    con.commit()
    con.close()
    back = bt.read_geopackage(path)
    np.testing.assert_array_equal(back.coords[1:], [[1, 2, 0], [3, 4, 0]])
    assert list(back["rock"]) == ["óxido", "x", "x"] and np.isnan(back["au"][1:]).all()
