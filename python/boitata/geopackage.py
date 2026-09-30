"""GeoPackage vector layers, stored with the standard library's `sqlite3`."""

import re
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from itertools import pairwise
from os import PathLike
from pathlib import Path

import numpy as np

from boitata._boitata import PointSet, Polylines, _decode_geometries, _encode_geometries
from boitata.errors import FileError, InvalidInput

__all__ = ["read_geopackage", "write_geopackage"]

_APPLICATION_ID = 0x47504B47
_NODATA = (-99.0, -999.0, 1e21, "NA", "N/A", "N.A.", "ND", "N/D", "NULL", "NONE", "NAN", "#N/A", "-", "--")
_NUMERIC = {"BOOLEAN", "TINYINT", "SMALLINT", "MEDIUMINT", "INT", "INTEGER", "FLOAT", "DOUBLE", "REAL"}
_WGS84 = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
    'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
)
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS gpkg_spatial_ref_sys (
  srs_name TEXT NOT NULL, srs_id INTEGER PRIMARY KEY, organization TEXT NOT NULL,
  organization_coordsys_id INTEGER NOT NULL, definition TEXT NOT NULL, description TEXT);
CREATE TABLE IF NOT EXISTS gpkg_contents (
  table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL, identifier TEXT UNIQUE,
  description TEXT DEFAULT '', last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER,
  CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id));
CREATE TABLE IF NOT EXISTS gpkg_geometry_columns (
  table_name TEXT NOT NULL, column_name TEXT NOT NULL, geometry_type_name TEXT NOT NULL,
  srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL,
  CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
  CONSTRAINT uk_gc_table_name UNIQUE (table_name),
  CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),
  CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id));
INSERT OR IGNORE INTO gpkg_spatial_ref_sys VALUES
  ('Undefined cartesian SRS', -1, 'NONE', -1, 'undefined', 'undefined cartesian coordinate reference system'),
  ('Undefined geographic SRS', 0, 'NONE', 0, 'undefined', 'undefined geographic coordinate reference system'),
  ('WGS 84 geodetic', 4326, 'EPSG', 4326, '{_WGS84}', 'longitude/latitude coordinates in decimal degrees');
"""


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _crs(con, srs_id):
    row = con.execute(
        "SELECT organization, organization_coordsys_id, definition FROM gpkg_spatial_ref_sys WHERE srs_id = ?",
        (srs_id,),
    ).fetchone()
    if row is None:
        return None
    organization, code, definition = row
    if str(organization).upper() == "EPSG":
        return f"EPSG:{code}"
    return None if definition.strip().lower() == "undefined" else definition


def _srs_id(con, crs):
    if crs is None:
        return -1
    code = int(crs[5:]) if crs[:5].upper() == "EPSG:" and crs[5:].isdigit() else None
    if code is None:
        found = con.execute("SELECT srs_id FROM gpkg_spatial_ref_sys WHERE definition = ?", (crs,))
    else:
        found = con.execute(
            "SELECT srs_id FROM gpkg_spatial_ref_sys WHERE upper(organization) = 'EPSG' "
            "AND organization_coordsys_id = ?",
            (code,),
        )
    if found := found.fetchone():
        return found[0]
    srs_id = code
    if code is None or con.execute("SELECT 1 FROM gpkg_spatial_ref_sys WHERE srs_id = ?", (code,)).fetchone():
        srs_id = max(100000, con.execute("SELECT max(srs_id) FROM gpkg_spatial_ref_sys").fetchone()[0] + 1)
    if code is None:
        name = re.search(r'"([^"]*)"', crs)
        row = (name[1] if name else "custom", srs_id, "NONE", srs_id, crs)
    else:
        row = (crs, srs_id, "EPSG", code, "undefined")
    con.execute("INSERT INTO gpkg_spatial_ref_sys VALUES (?, ?, ?, ?, ?, NULL)", row)
    return srs_id


def read_geopackage(
    path: str | PathLike[str], *, layer: str | None = None, nodata: Sequence[float | str] | None = None
) -> PointSet | Polylines:
    """Reads a GeoPackage feature layer as a PointSet or Polylines.

    Parameters
    ----------
    path : str or Path
        The `.gpkg` file.
    layer : str, optional
        Feature table to read; required when the file has more than one.
    nodata : sequence of float or str, optional
        Values read as null, as in `read_csv`: numbers match numerically, strings match text case-insensitively.
        Default -99, -999, 1e21 and text such as ``NA``, ``N/A`` or ``NULL``. SQL NULLs are always null.

    Returns
    -------
    PointSet or Polylines
        Point and multipoint layers give one point per member, a multipoint repeating its row; rows without a
        geometry or with an empty one are dropped. Line and polygon layers give Polylines, lines as open parts and
        polygon rings as closed parts without the repeated closing vertex; a row without a geometry is a feature
        without parts. 2D geometries get z = 0, M values are dropped. Integer, real and boolean columns are float64,
        blobs are skipped, the rest are text. The CRS is ``"EPSG:<code>"`` when the layer's spatial reference is
        an EPSG one, otherwise its WKT definition.

    Raises
    ------
    FileError
        When the file does not exist.
    InvalidInput
        When the file is not a GeoPackage, `layer` is missing or ambiguous, or the layer mixes points with lines
        or holds geometry collections.
    """
    path = Path(path)
    if not path.is_file():
        raise FileError(f"no such file: {path}")
    numbers = {float(v) for v in (_NODATA if nodata is None else nodata) if not isinstance(v, str)}
    texts = {v.upper() for v in (_NODATA if nodata is None else nodata) if isinstance(v, str)}

    def missing(v):
        if isinstance(v, str):
            try:
                return v.upper() in texts or float(v) in numbers
            except ValueError:
                return False
        return v is None or (isinstance(v, int | float) and float(v) in numbers)

    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as con:
        try:
            layers = con.execute(
                "SELECT table_name, column_name, geometry_type_name, srs_id FROM gpkg_geometry_columns"
            ).fetchall()
        except sqlite3.DatabaseError as e:
            raise InvalidInput(f"{path} is not a GeoPackage: {e}") from None
        names = [row[0] for row in layers]
        if layer is None and len(layers) != 1:
            raise InvalidInput(f"pass layer=, one of: {', '.join(names)}" if layers else "no feature layers")
        if layer is not None and layer not in names:
            raise InvalidInput(f"no layer {layer!r}; layers: {', '.join(names)}")
        table, geometry, type_name, srs_id = layers[0] if layer is None else layers[names.index(layer)]
        info = con.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
        key = next((c[1] for c in info if c[5]), "rowid")
        columns = [
            (c[1], c[2].split("(")[0].strip().upper() in _NUMERIC)
            for c in info
            if c[1] not in (key, geometry) and c[2].upper() != "BLOB"
        ]
        select = ", ".join(_quote(c) for c in [geometry, *(name for name, _ in columns)])
        rows = con.execute(f"SELECT {select} FROM {_quote(table)} ORDER BY {_quote(key)}").fetchall()
        crs = _crs(con, srs_id)

    attributes = {}
    for i, (name, numeric) in enumerate(columns, start=1):
        values = [row[i] for row in rows]
        if numeric:
            attributes[name] = np.array(
                [np.nan if missing(v) or not isinstance(v, int | float) else float(v) for v in values]
            )
        else:
            attributes[name] = np.array([None if missing(v) else str(v) for v in values], dtype=object)
    kinds, vertices, offsets, parts_row, closed = _decode_geometries([row[0] for row in rows])
    kinds = set(kinds) - {0}
    if not kinds:
        kinds = {1} if type_name.upper() in ("POINT", "MULTIPOINT") else {2}
    if kinds <= {1, 4}:
        take = np.asarray(parts_row, dtype=np.int64)
        attributes = {name: values[take] for name, values in attributes.items()}
        return PointSet(vertices, attributes or None, crs=crs)
    if not kinds <= {2, 3, 5, 6}:
        raise InvalidInput(f"layer {table!r} mixes points with lines or holds geometry collections")
    return Polylines(
        [vertices[a:b] for a, b in pairwise(offsets)],
        closed=closed,
        features=parts_row,
        attributes=attributes or None,
        crs=crs,
    )


def write_geopackage(
    path: str | PathLike[str], data: PointSet | Polylines, *, layer: str | None = None
) -> None:
    """Writes a PointSet or Polylines as a GeoPackage feature layer.

    Parameters
    ----------
    path : str or Path
        The `.gpkg` file; created when missing, otherwise its other layers are kept.
    data : PointSet or Polylines
        Points are written as POINT Z, one row each. Polylines are written one row per feature: as MULTILINESTRING Z
        when all parts are open, as MULTIPOLYGON Z when all are closed, rings grouped into polygons by nesting,
        exteriors counter-clockwise and holes clockwise; split mixed ones by ``closed``. A feature without parts is
        a NULL geometry. Numeric columns are written as DOUBLE, text as TEXT, nulls as NULL.
    layer : str, optional
        Feature table name, the file stem by default. A layer of that name is replaced.

    Notes
    -----
    An ``"EPSG:<code>"`` CRS is stored as that EPSG spatial reference, any other as a custom one whose definition
    is the CRS text, which GIS software expects as WKT. Without a CRS the layer uses the undefined Cartesian
    reference (-1).

    Raises
    ------
    InvalidInput
        When `data` is neither container, parts mix open and closed, an attribute is named ``fid`` or ``geom``,
        `layer` starts with ``gpkg_``, or the file exists and is not a GeoPackage.
    """
    path = Path(path)
    layer = path.stem if layer is None else layer
    if not isinstance(data, PointSet | Polylines):
        raise InvalidInput("data must be a PointSet or Polylines")
    if layer.lower().startswith("gpkg_"):
        raise InvalidInput("layer names starting with gpkg_ are reserved")
    names = data.attributes.column_names
    if reserved := {"fid", "geom"} & {n.lower() for n in names}:
        raise InvalidInput(f"attribute names {sorted(reserved)} are reserved for the key and the geometry")
    columns, fields = [], ""
    for name in names:
        values = data[name]
        if values.dtype == object:
            columns.append(list(values))
            fields += f", {_quote(name)} TEXT"
        else:
            columns.append([None if np.isnan(v) else float(v) for v in values])
            fields += f", {_quote(name)} DOUBLE"
    vertices = data.coords if isinstance(data, PointSet) else data.vertices
    bounds = (
        [float(b) for b in (*vertices[:, :2].min(0), *vertices[:, :2].max(0))]
        if len(vertices)
        else [None] * 4
    )

    new = not path.exists()
    try:
        with closing(sqlite3.connect(path, isolation_level=None)) as con:
            try:
                application_id = con.execute("PRAGMA application_id").fetchone()[0]
            except sqlite3.DatabaseError as e:
                raise InvalidInput(f"{path} is not a GeoPackage: {e}") from None
            if not new and application_id != _APPLICATION_ID:
                raise InvalidInput(f"{path} is not a GeoPackage")
            con.execute("BEGIN")
            try:
                con.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
                con.execute("PRAGMA user_version = 10400")
                for statement in _SCHEMA.split(";"):
                    con.execute(statement)
                srs_id = _srs_id(con, data.crs)
                type_name, blobs = _encode_geometries(data, srs_id)
                con.execute(f"DROP TABLE IF EXISTS {_quote(layer)}")
                con.execute("DELETE FROM gpkg_geometry_columns WHERE table_name = ?", (layer,))
                con.execute("DELETE FROM gpkg_contents WHERE table_name = ?", (layer,))
                con.execute(
                    f"CREATE TABLE {_quote(layer)} (fid INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, "
                    f"geom {type_name}{fields})"
                )
                con.execute(
                    "INSERT INTO gpkg_contents (table_name, data_type, identifier, min_x, min_y, max_x, max_y, "
                    "srs_id) VALUES (?, 'features', ?, ?, ?, ?, ?, ?)",
                    (layer, layer, *bounds, srs_id),
                )
                con.execute(
                    "INSERT INTO gpkg_geometry_columns VALUES (?, 'geom', ?, ?, 1, 0)",
                    (layer, type_name, srs_id),
                )
                con.executemany(
                    f"INSERT INTO {_quote(layer)} (geom{''.join(', ' + _quote(n) for n in names)}) "
                    f"VALUES ({', '.join('?' * (len(names) + 1))})",
                    zip(blobs, *columns, strict=True),
                )
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
    except BaseException:
        if new:
            path.unlink(missing_ok=True)
        raise
