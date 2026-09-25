"""Example datasets, downloaded once from a pinned commit and checked by SHA-256.

Files are cached under ``$CERES_DATA``, else the user cache directory
(``%LOCALAPPDATA%/ceres`` on Windows, ``$XDG_CACHE_HOME/ceres`` or
``~/.cache/ceres`` elsewhere).
"""

import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

from ceres._ceres import Drillholes, PointSet, Table, read_csv
from ceres.errors import FileError, InvalidInput

__all__ = [
    "drillhole_tables",
    "drillholes",
    "fetch",
    "geomet",
    "jura",
    "walker_lake",
    "walker_lake_exhaustive",
]

COMMIT = "560bd39c9ec79dcd6565dc763009aa83cac68fb8"
URL = f"https://raw.githubusercontent.com/gstvschlz/datasets/{COMMIT}"
FILES = {
    "drillholes/assay.csv": "73c0ef83b4b95b0e9e27f8fd70bb3b16db6dd83e2f03af2594348de288c4b50c",
    "drillholes/collar.csv": "32659b881ed13f7576ec2baf81a159dde4fbb943d83890df79768041581dd572",
    "drillholes/geology.csv": "91c15f49d15be88a145944dcb3f459b86fdd2608e6b6d9f4a1ba1d8c7b0475ef",
    "drillholes/survey.csv": "84257952228ba01200c6d440eaa6bca104a9afb09ccdb199e38e5a4d730914db",
    "geomet/porphyry_01/block_model.csv": "c8acd0f83c003f48176cf8ed72b6d18d5844623f4f6894099adce3bfdb1dddd9",
    "geomet/porphyry_01/correlations.csv": "0cf251ae8da93b939d5d68515e7ed038fb2f0ea309c02972e075709dfc2db824",
    "geomet/porphyry_01/grindability_distribution.csv": "27dca2a990d8332fc649b24d1deabff5efeb7743ed50c4321b92513636509ff5",
    "geomet/porphyry_01/mineralogical_distributions.csv": "7dee2184b75b5484a8f890fe3b7f86de27787f90457554d9fa91f480c2e80177",
    "geomet/porphyry_01/pseudo_drillholes.csv": "26218bc06f8abb1a5db4fd8062bb41fc6b1241d30bd2e9447b33b1abe5686ad3",
    "geomet/porphyry_01/synthetic_drillholes.csv": "f9aa26c9b44aca13de13f322f11d5836b3c1445f336faf99113f81e95e71de6e",
    "geomet/porphyry_02/drillholes.csv": "3de5c043a1c327350c026622c8a073226087417ab939815923a7dee09c21590e",
    "geomet/porphyry_02/topography.csv": "03327e5c4a96b38b14186aa81e809da08a0b923dc19586dcf29b6319f893423a",
    "geomet/porphyry_03/drillholes.csv": "01ee1469de27746e782d376e89a948573e9c6751caeef360e5f81445e809d03f",
    "geomet/porphyry_03/topography.csv": "2f0f9bc706d8f0378fa2e57d6d605a755e7fab6aed7ae892088aca666dfc181f",
    "jura/grid.csv": "f8170fabb6156a73ae0f43298118e61a413f2e460e40bfe0078d1bff38897a28",
    "jura/prediction.csv": "e3e82599d844d1f44dec83d5c0772bb6c9222d0b4050318a44795915f66c6ff6",
    "jura/validation.csv": "a88f6b2b0b8273f4aa6159114a659194de38a58e386f9d6710bfbed6273c58fb",
    "walker-lake/exhaustive.csv": "0f817af11b19c18236330412d7f6959a5418296ac54974fd87a6dfeb58019e46",
    "walker-lake/sample.csv": "78618f1684c78ed06759f738672c768e06fb0cf1d7e24f5636041e44afdbd146",
}


def _cache() -> Path:
    if "CERES_DATA" in os.environ:
        return Path(os.environ["CERES_DATA"])
    if sys.platform == "win32" and "LOCALAPPDATA" in os.environ:
        base = Path(os.environ["LOCALAPPDATA"])
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "ceres" / COMMIT[:12]


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch(name: str) -> Path:
    """Local path of dataset file `name` (e.g. ``"walker-lake/sample.csv"``), downloaded on first use.

    Raises
    ------
    InvalidInput
        If `name` is not a known dataset file.
    FileError
        If the download fails or its SHA-256 does not match.
    """
    if name not in FILES:
        raise InvalidInput(f"unknown dataset file {name!r}; known: {', '.join(sorted(FILES))}")
    path = _cache() / name
    if path.exists() and _sha256(path) == FILES[name]:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        try:
            with urllib.request.urlopen(f"{URL}/{name}") as response:
                tmp.write(response.read())
        except OSError as e:
            tmp.close()
            os.unlink(tmp.name)
            raise FileError(f"could not download {name}: {e}") from e
    if _sha256(Path(tmp.name)) != FILES[name]:
        os.unlink(tmp.name)
        raise FileError(f"{name} does not match its SHA-256; the download is corrupt")
    os.replace(tmp.name, path)
    return path


def walker_lake() -> PointSet:
    """The 470 Walker Lake samples: V, U (ppm) and type T, on a 260 x 300 m area."""
    return PointSet.from_table(read_csv(fetch("walker-lake/sample.csv")))


def walker_lake_exhaustive() -> Table:
    """The exhaustive Walker Lake grid: 78 000 values at 1 m spacing, x fastest."""
    return read_csv(fetch("walker-lake/exhaustive.csv"))


def jura() -> dict[str, PointSet]:
    """Jura soil samples (``"prediction"``, ``"validation"``) and the ``"grid"`` of land use and rock type."""
    return {
        k: PointSet.from_table(read_csv(fetch(f"jura/{k}.csv"))) for k in ("prediction", "validation", "grid")
    }


def drillhole_tables() -> dict[str, Table]:
    """The ``collar``, ``survey``, ``assay`` and ``geology`` tables of the drillhole dataset."""
    return {k: read_csv(fetch(f"drillholes/{k}.csv")) for k in ("collar", "survey", "assay", "geology")}


def drillholes() -> Drillholes:
    """Desurveyed drillholes with their assays."""
    t = drillhole_tables()
    return Drillholes(t["collar"], t["survey"], t["assay"])


def geomet(name: str = "porphyry_01/synthetic_drillholes") -> Table:
    """A table of the geometallurgical datasets, e.g. ``"porphyry_02/drillholes"``."""
    return read_csv(fetch(f"geomet/{name}.csv"))
