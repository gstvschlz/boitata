"""Example datasets, downloaded once from a pinned commit and checked by SHA-256.

Files are cached under ``$BOITATA_DATA``, else the user cache directory
(``%LOCALAPPDATA%/boitata`` on Windows, ``$XDG_CACHE_HOME/boitata`` or
``~/.cache/boitata`` elsewhere). Coordinates are local metres (Jura: km) with no CRS, except F3 (EPSG:23031).
Synthetic datasets are CC BY 4.0; classic ones keep their original terms.
"""

import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import TypedDict

import numpy as np

from boitata._boitata import BlockModel, Mesh, PointSet, Polylines, Table, read_csv, read_mesh, read_segy
from boitata.errors import FileError, InvalidInput

__all__ = [
    "coal_seam_thickness",
    "f3_seismic",
    "fetch",
    "iron_formation_plateau",
    "jura",
    "nickel_laterite_profile",
    "phosphate_weathering_profile",
    "porphyry_geometallurgy",
    "soil_geochemistry_survey",
    "stacked_sulphide_lenses",
    "strebelle",
    "tailings_reprocessing",
    "vein_gold_grade_control",
    "walker_lake",
    "walker_lake_exhaustive",
]

REPO = "https://raw.githubusercontent.com/gstvschlz/datasets"
COMMIT = "6cba7994ef7376381039c757ad9a996fd7927d8e"
URL = f"{REPO}/{COMMIT}"
FILES = {
    "mining/2d/coal-seam-thickness/boreholes.csv": "722383adf51416b50c897b08e61d8a7d7cafd2bdb244426999c7057fdd07420d",
    "mining/2d/coal-seam-thickness/boundary.csv": "5c29cb26c0b925fb37e811ad98bd7ae31d459adc8905114830865b02790b05a8",
    "mining/2d/coal-seam-thickness/grid.csv": "8c96de664d7301ab5f086085a1793db94a31a3faa47764b03cace05f6e7aaf08",
    "mining/2d/jura/grid.csv": "f8170fabb6156a73ae0f43298118e61a413f2e460e40bfe0078d1bff38897a28",
    "mining/2d/jura/prediction.csv": "e3e82599d844d1f44dec83d5c0772bb6c9222d0b4050318a44795915f66c6ff6",
    "mining/2d/jura/validation.csv": "a88f6b2b0b8273f4aa6159114a659194de38a58e386f9d6710bfbed6273c58fb",
    "mining/2d/soil-geochemistry-survey/boundary.csv": "844bad122697bd857f16be17672e652352f6c69cdbdb6ba3af20b1a765843885",
    "mining/2d/soil-geochemistry-survey/covariates.csv": "bad0b6dcece0d1ae5b5a7fa5bf935dbfcf68fb0b5d1c36ea0c00af982271116b",
    "mining/2d/soil-geochemistry-survey/samples.csv": "7bbc0344d0140234c042477c05b8ebd46fa710d384ebb657d08576c50d64dab8",
    "mining/2d/walker-lake/exhaustive.csv": "0f817af11b19c18236330412d7f6959a5418296ac54974fd87a6dfeb58019e46",
    "mining/2d/walker-lake/sample.csv": "78618f1684c78ed06759f738672c768e06fb0cf1d7e24f5636041e44afdbd146",
    "mining/3d/iron-formation-plateau/assays.csv": "6138ab317ef95aee8b18388dcb81c1670877a86353a3b886958abe5d573d21fa",
    "mining/3d/iron-formation-plateau/blastholes.csv": "e40a3d02e0cd28f2b57317fd1d05036f24df90a0e72531eb6fae8b697f7e2d05",
    "mining/3d/iron-formation-plateau/block_model.csv": "ae624dfdfae03ce9fe7dd2e88c7d4afe46bf992c4dfb496e83e60002a55958d9",
    "mining/3d/iron-formation-plateau/collars.csv": "68d24008cc9118f2d88d548a4150e45649717156b4b8e21a358130202f43ed05",
    "mining/3d/iron-formation-plateau/high_grade.stl": "cdd4e8ad76a975b976fcbc20262ceaa35a4a72880136136e1e5cc9c183a73124",
    "mining/3d/iron-formation-plateau/iron_formation.stl": "aba4baf436e322840f24e8efa11e77f7dda948cfc3986722a2745ee9ddf75fa5",
    "mining/3d/iron-formation-plateau/lithology.csv": "edecb92d526474e039e2121b97d133d35ae9186e3c7a4b2750125f27972bcfc0",
    "mining/3d/iron-formation-plateau/surveys.csv": "0de68bf2b8b8ab30f99fea9f8c760128cb8b9a1c2e6cdbc8b6967a26b0efdc84",
    "mining/3d/iron-formation-plateau/topography.csv": "681608f3c8786bfa80fb912983c70bddcd2164eab759aad250e0c6a151b4f8c5",
    "mining/3d/nickel-laterite-profile/assays.csv": "ee09bf18d8012d23e3362253b74ece0ddb418d111c11bb65afaa05fed2b530c2",
    "mining/3d/nickel-laterite-profile/collars.csv": "360b79695d3fd786e9758f14be64fb334b601ad8d678ca3101adb748a473f61c",
    "mining/3d/nickel-laterite-profile/horizons.csv": "020d65523601aaf5e9f01e8fa8080bd2b00ad55c937ca5d16cb6541a3bbc51f1",
    "mining/3d/nickel-laterite-profile/surveys.csv": "d069af89bacbd5b9905d8bf4ae02253d5cdf5ba010eb4b92eb341e7848b0f792",
    "mining/3d/phosphate-weathering-profile/assays.csv": "ff168e3a632b923d38d54562e627011a75d36856d08f5e600fd3841ca49ede7a",
    "mining/3d/phosphate-weathering-profile/collars.csv": "f23f51374e57894a76a0f9f09d6527b1c9680f12f8a85b1f0502124056746d2c",
    "mining/3d/phosphate-weathering-profile/horizons.csv": "7d2878317fd2051fdb00ce9ee3e8c13cfa2c705dd3d4f9b21aff53347be07dad",
    "mining/3d/phosphate-weathering-profile/surveys.csv": "0804b8bf6a937b29f999622adae964bfa2f36c2dcbdd6d37bec7f64bbb57a891",
    "mining/3d/phosphate-weathering-profile/topography.csv": "9c12357595fedbf108f08dd7635513b1e4082027e588565ca3cb0ca30afb7567",
    "mining/3d/porphyry-geometallurgy/porphyry_01/block_model.csv": "c8acd0f83c003f48176cf8ed72b6d18d5844623f4f6894099adce3bfdb1dddd9",
    "mining/3d/porphyry-geometallurgy/porphyry_01/correlations.csv": "0cf251ae8da93b939d5d68515e7ed038fb2f0ea309c02972e075709dfc2db824",
    "mining/3d/porphyry-geometallurgy/porphyry_01/grindability_distribution.csv": "27dca2a990d8332fc649b24d1deabff5efeb7743ed50c4321b92513636509ff5",
    "mining/3d/porphyry-geometallurgy/porphyry_01/mineralogical_distributions.csv": "7dee2184b75b5484a8f890fe3b7f86de27787f90457554d9fa91f480c2e80177",
    "mining/3d/porphyry-geometallurgy/porphyry_01/pseudo_drillholes.csv": "26218bc06f8abb1a5db4fd8062bb41fc6b1241d30bd2e9447b33b1abe5686ad3",
    "mining/3d/porphyry-geometallurgy/porphyry_01/synthetic_drillholes.csv": "f9aa26c9b44aca13de13f322f11d5836b3c1445f336faf99113f81e95e71de6e",
    "mining/3d/porphyry-geometallurgy/porphyry_02/drillholes.csv": "3de5c043a1c327350c026622c8a073226087417ab939815923a7dee09c21590e",
    "mining/3d/porphyry-geometallurgy/porphyry_02/topography.csv": "03327e5c4a96b38b14186aa81e809da08a0b923dc19586dcf29b6319f893423a",
    "mining/3d/porphyry-geometallurgy/porphyry_03/drillholes.csv": "01ee1469de27746e782d376e89a948573e9c6751caeef360e5f81445e809d03f",
    "mining/3d/porphyry-geometallurgy/porphyry_03/topography.csv": "2f0f9bc706d8f0378fa2e57d6d605a755e7fab6aed7ae892088aca666dfc181f",
    "mining/3d/stacked-sulphide-lenses/assays.csv": "1b91f6ae05b9932e6f1e220e99be654fc12a4bfd58a65fec45ee11d379a65a1a",
    "mining/3d/stacked-sulphide-lenses/collars.csv": "a0c84d5614565f90300a0ff115e01357d23a9b218daecb7e0113b814ecb1846c",
    "mining/3d/stacked-sulphide-lenses/lens_1.stl": "9faf5bbf7b2b0b00d12ad1cf435c1113a1f7f2f5b9f626ef2dda8290d5f6c035",
    "mining/3d/stacked-sulphide-lenses/lens_2.stl": "697cc65d886a4aedeae1973b494a34a438b971e9baf5c34870910ed33383a517",
    "mining/3d/stacked-sulphide-lenses/lens_3.stl": "45651f98347eb4ec347e82fce1106d1bdad9b506a37e577acc31f31ebaf7e1ca",
    "mining/3d/stacked-sulphide-lenses/lithology.csv": "d91f97cc0bcfce30d91eb159294ff47fbfca8fb729aa3ee3079b84c392cdb87e",
    "mining/3d/stacked-sulphide-lenses/raw/assays.csv": "ab959842f61e072e229a0b8c74085b2643afda2af1ff9a9b60c1dedcd2ed184c",
    "mining/3d/stacked-sulphide-lenses/raw/collars.csv": "9a7110e62e0ddb2eacf4ebbfc3748b901ab53ed93b47429914322ff5c173bdda",
    "mining/3d/stacked-sulphide-lenses/raw/lithology.csv": "6d40be7f9488246e53e74c5a3da20c4526bc53c90e15b5dd1aa3f400b06ac0ac",
    "mining/3d/stacked-sulphide-lenses/raw/surveys.csv": "8360f29a7a2e8ada3433124093935fbc5ad9bab647ce13d64d7e8f743d628691",
    "mining/3d/stacked-sulphide-lenses/surveys.csv": "fedfd53f157efca2fd124977bedf6e517b20f59c9025e70cb95ba485c1a2a88c",
    "mining/3d/stacked-sulphide-lenses/topography.csv": "c10a1a7eb49dfe2dc007a6b9be8872c649c80cb54202769c364b9dd613b28882",
    "mining/3d/tailings-reprocessing/assays.csv": "01b5fbebfbbbb09244ce41c87b79618d241270d65ef52c89214304460264ef88",
    "mining/3d/tailings-reprocessing/collars.csv": "446b9b6042059c31e3613e5f48f136f070b50577888aeecf480040eee2197c24",
    "mining/3d/tailings-reprocessing/lithology.csv": "b9946c48a772fc15d1762c3071c07d41622dbf011ad2b3749174df352cffb744",
    "mining/3d/tailings-reprocessing/original_ground.csv": "9f4e748672430f956362d2177288826cc1895d1f8409ea63241b6e0ea9990e57",
    "mining/3d/tailings-reprocessing/surface.csv": "40ae1b8a60e5eba9a93eb7a27edc9421a1708d1ac5bedd32dd51dc24b7e50c4d",
    "mining/3d/tailings-reprocessing/surveys.csv": "2e56929afd9a2058e6fc83ce53b8d5f08720003100348874df6f776e678cfaa3",
    "mining/3d/vein-gold-grade-control/assays.csv": "03cfb07528dfce42d26416a8782b3c009fd3eb3c7b18c0ff331312a966eff2b1",
    "mining/3d/vein-gold-grade-control/collars.csv": "751b93a01f721563a8a49bea3084271907199202d15d2f2504d944aa5b09b4c3",
    "mining/3d/vein-gold-grade-control/lithology.csv": "c44c9a0d588a5e12420829334a12ebc1f3b43347a052d9b40ced73b343188235",
    "mining/3d/vein-gold-grade-control/surveys.csv": "7226e9174acdfdbb38374db9cac4dca4e884e7a8158fefb887c5bd81db7a04bb",
    "mining/3d/vein-gold-grade-control/topography.csv": "657b25c64ba63dd81e4482feb2a36f245215687e1a71c9b5751a34f3f0520363",
    "mining/3d/vein-gold-grade-control/vein_V1.stl": "6233902355de32fe79ce723e8757d91f331df7531ee00f0f5112d6da4cf9c349",
    "mining/3d/vein-gold-grade-control/vein_V2.stl": "0d2c5a8d497a67c14d456782929346ce5358125508b239932b868630ed3b646c",
    "mining/3d/vein-gold-grade-control/vein_V3.stl": "9d34c4ae2b4002e90998251fd82dfe164dbc0ffe2a6d9edd4448332ccdccec31",
    "mining/3d/vein-gold-grade-control/vein_V4.stl": "889ef7c06e49d8734352a443bd420d797e583a0af3a2d97b15ff0e9f4b093f05",
    "oil-gas/2d/strebelle/training_image.csv": "520159482ac4def977300ef8a1af9d4a09e037ac78e53a7b3c8bbc8d5eea1d4b",
    "oil-gas/3d/f3-seismic/seismic.sgy": "ddd2038885a7abd436413fa31dcc600c3972c93f8c0a404282f2057d5f42c703",
}
HOLES = ("collars", "surveys", "assays")


class CoalSeamThickness(TypedDict):
    boreholes: PointSet
    boundary: Polylines
    grid: BlockModel


class SoilGeochemistrySurvey(TypedDict):
    samples: PointSet
    boundary: Polylines
    covariates: BlockModel


class StackedSulphideLenses(TypedDict):
    collars: Table
    surveys: Table
    assays: Table
    lithology: Table
    lens_1: Mesh
    lens_2: Mesh
    lens_3: Mesh
    topography: BlockModel


class StackedSulphideLensesRaw(TypedDict):
    collars: Table
    surveys: Table
    assays: Table
    lithology: Table


class IronFormationPlateau(TypedDict):
    collars: Table
    surveys: Table
    assays: Table
    lithology: Table
    blastholes: PointSet
    block_model: BlockModel
    high_grade: Mesh
    iron_formation: Mesh
    topography: BlockModel


class VeinGoldGradeControl(TypedDict):
    collars: Table
    surveys: Table
    assays: Table
    lithology: Table
    vein_V1: Mesh
    vein_V2: Mesh
    vein_V3: Mesh
    vein_V4: Mesh
    topography: BlockModel


class PhosphateWeatheringProfile(TypedDict):
    collars: Table
    surveys: Table
    assays: Table
    horizons: Table
    topography: BlockModel


class NickelLateriteProfile(TypedDict):
    collars: Table
    surveys: Table
    assays: Table
    horizons: Table


class TailingsReprocessing(TypedDict):
    collars: Table
    surveys: Table
    assays: Table
    lithology: Table
    surface: BlockModel
    original_ground: BlockModel


class PorphyryGeometallurgy(TypedDict, total=False):
    block_model: BlockModel
    correlations: Table
    grindability_distribution: Table
    mineralogical_distributions: Table
    pseudo_drillholes: PointSet
    synthetic_drillholes: PointSet
    drillholes: PointSet
    topography: BlockModel


def _cache() -> Path:
    if "BOITATA_DATA" in os.environ:
        return Path(os.environ["BOITATA_DATA"])
    if sys.platform == "win32" and "LOCALAPPDATA" in os.environ:
        base = Path(os.environ["LOCALAPPDATA"])
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "boitata" / COMMIT[:12]


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch(name: str) -> Path:
    """Local path of dataset file `name` (e.g. ``"mining/2d/walker-lake/sample.csv"``), downloaded on first use.

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


def _tables(folder: str, stems: tuple[str, ...], **kwargs) -> dict[str, Table]:
    return {s: read_csv(fetch(f"{folder}/{s}.csv"), **kwargs) for s in stems}


def _meshes(folder: str, stems: tuple[str, ...]) -> dict[str, Mesh]:
    return {s: read_mesh(fetch(f"{folder}/{s}.stl")) for s in stems}


def _polygon(table: Table) -> Polylines:
    return Polylines([np.column_stack([table["X"], table["Y"]])], closed=True)


def _grid(table: Table, axes: tuple[str, ...] = ("X", "Y")) -> BlockModel:
    """Regular grid of the cell centres in `axes`, masked if some cells are missing."""
    xyz = np.column_stack([table[a] for a in axes])
    size = np.array([np.diff(np.unique(c)).min() for c in xyz.T])
    ijk = np.rint((xyz - xyz.min(0)) / size).astype(np.int64)
    count = ijk.max(0) + 1
    index = np.ravel_multi_index(ijk.T[::-1], count[::-1]).astype(np.uint64)
    order = np.argsort(index, kind="stable")
    attributes = {c: table[c][order] for c in table.column_names if c not in axes}
    return BlockModel(
        xyz.min(0) - size / 2,
        size,
        count.tolist(),
        attributes=attributes,
        index=None if len(index) == count.prod() else index[order],
    )


def walker_lake() -> PointSet:
    """The 470 Walker Lake samples: V, U (ppm) and type T, on a 260 x 300 m area.

    Classic public dataset, under its original terms.
    """
    return PointSet.from_table(read_csv(fetch("mining/2d/walker-lake/sample.csv")))


def walker_lake_exhaustive() -> Table:
    """The exhaustive Walker Lake grid: 78 000 values at 1 m spacing, x fastest.

    Classic public dataset, under its original terms.
    """
    return read_csv(fetch("mining/2d/walker-lake/exhaustive.csv"))


def jura() -> dict[str, PointSet]:
    """Jura topsoil metals: 259 ``"prediction"`` and 100 ``"validation"`` samples, and the ``"grid"`` of
    land use and rock type (5957 nodes), in km.

    Classic public dataset, under its original terms.
    """
    return {
        k: PointSet.from_table(t)
        for k, t in _tables("mining/2d/jura", ("prediction", "validation", "grid")).items()
    }


def coal_seam_thickness() -> CoalSeamThickness:
    """A coal seam in a lease, with infill drilled where the seam is thick. Synthetic, CC BY 4.0.

    Returns
    -------
    dict
        ``boreholes``: PointSet of 295 holes (THICKNESS_M, ASH_PCT, CV_MJKG, CATEGORY).
        ``boundary``: Polylines, the lease polygon (48 vertices).
        ``grid``: BlockModel of 120 x 90 cells of 100 m, INSIDE the lease or not.
    """
    t = _tables("mining/2d/coal-seam-thickness", ("boreholes", "boundary", "grid"))
    return {
        "boreholes": PointSet.from_table(t["boreholes"]),
        "boundary": _polygon(t["boundary"]),
        "grid": _grid(t["grid"]),
    }


def soil_geochemistry_survey() -> SoilGeochemistrySurvey:
    """Soil samples along survey lines, with detection limits and covariates. Synthetic, CC BY 4.0.

    Returns
    -------
    dict
        ``samples``: PointSet of 1227 samples (CU_PPM, ZN_PPM, AU_PPB, AS_PPM; ``AU_BDL`` and ``AS_BDL``
        flag values below detection, which hold the limit).
        ``boundary``: Polylines, the survey area (36 vertices).
        ``covariates``: BlockModel of 120 x 80 cells of 50 m (ELEVATION_M, MAG_NT, LITHOLOGY, INSIDE).
    """
    t = _tables("mining/2d/soil-geochemistry-survey", ("samples", "boundary", "covariates"))
    return {
        "samples": PointSet.from_table(t["samples"]),
        "boundary": _polygon(t["boundary"]),
        "covariates": _grid(t["covariates"]),
    }


def stacked_sulphide_lenses(raw: bool = False) -> StackedSulphideLenses | StackedSulphideLensesRaw:
    """Three dipping polymetallic lenses cut by 244 diamond and 45 RC holes. Synthetic, CC BY 4.0.

    Parameters
    ----------
    raw : bool
        Return the drillhole tables as logged, with planted errors to check and fix (duplicate and
        missing holes, overlapping, inverted and past-depth intervals, survey flips, text in numeric
        columns, twinned samples), read as-is: -99 and -999 stay numbers.

    Returns
    -------
    dict
        ``collars`` (289 rows), ``surveys`` (4348), ``assays`` (16 995: ZN_PCT, PB_PCT, CU_PCT, AG_GPT,
        AU_GPT, DENSITY) and ``lithology`` (1726: LITH) tables; the raw ones have 290, 4326, 16 907 and
        1726 rows. Unless `raw`, also ``lens_1`` to ``lens_3`` Mesh solids and ``topography``, a BlockModel
        of 141 x 131 cells of 20 m with Z.
    """
    folder = "mining/3d/stacked-sulphide-lenses"
    if raw:
        return _tables(f"{folder}/raw", (*HOLES, "lithology"), nodata=[])
    t = _tables(folder, (*HOLES, "lithology", "topography"))
    return {**t, **_meshes(folder, ("lens_1", "lens_2", "lens_3")), "topography": _grid(t["topography"])}


def iron_formation_plateau() -> IronFormationPlateau:
    """An iron formation with exploration holes, grade-control blastholes and a block model. Synthetic, CC BY 4.0.

    Returns
    -------
    dict
        ``collars`` (187 rows), ``surveys`` (1088), ``assays`` (17 475: FE_PCT, SIO2_PCT, AL2O3_PCT, P_PCT,
        MN_PCT, LOI_PCT, DENSITY) and ``lithology`` (7573: LITH) tables.
        ``blastholes``: PointSet of 1953 holes at mid-bench, with Z_TOP, Z_BOTTOM, FE_PCT, SIO2_PCT.
        ``block_model``: masked BlockModel of 129 278 cells of 25 x 25 x 12 m with LITH.
        ``high_grade``, ``iron_formation``: Mesh solids.
        ``topography``: BlockModel of 121 x 121 cells of 25 m with Z.
    """
    folder = "mining/3d/iron-formation-plateau"
    t = _tables(folder, (*HOLES, "lithology", "blastholes", "block_model", "topography"))
    b = t["blastholes"]
    blastholes = PointSet(
        np.column_stack([b["X"], b["Y"], (b["Z_TOP"] + b["Z_BOTTOM"]) / 2]),
        {c: b[c] for c in b.column_names if c not in ("X", "Y")},
    )
    return {
        **t,
        "blastholes": blastholes,
        "block_model": _grid(t["block_model"], ("X", "Y", "Z")),
        **_meshes(folder, ("high_grade", "iron_formation")),
        "topography": _grid(t["topography"]),
    }


def vein_gold_grade_control() -> VeinGoldGradeControl:
    """Four steep gold veins with 118 diamond holes and 2492 channels. Synthetic, CC BY 4.0.

    Returns
    -------
    dict
        ``collars`` (2610 rows), ``surveys`` (6736), ``assays`` (39 799: AU_GPT, AG_GPT, AS_PPM) and
        ``lithology`` (10 954: LITH, VEIN) tables; ``vein_V1`` to ``vein_V4`` Mesh solids;
        ``topography``, a BlockModel of 71 x 71 cells of 20 m with Z.
    """
    folder = "mining/3d/vein-gold-grade-control"
    t = _tables(folder, (*HOLES, "lithology", "topography"))
    return {
        **t,
        **_meshes(folder, ("vein_V1", "vein_V2", "vein_V3", "vein_V4")),
        "topography": _grid(t["topography"]),
    }


def phosphate_weathering_profile() -> PhosphateWeatheringProfile:
    """Phosphate in weathering horizons under rolling topography. Synthetic, CC BY 4.0.

    Returns
    -------
    dict
        ``collars`` (251 rows), ``surveys`` (858), ``assays`` (5039: P2O5_PCT, P2O5_AP_PCT, CAO_PCT,
        SIO2_PCT, FE2O3_PCT, AL2O3_PCT, MGO_PCT) and ``horizons`` (1255: HORIZON) tables;
        ``topography``, a BlockModel of 241 x 191 cells of 10 m with Z.
    """
    t = _tables("mining/3d/phosphate-weathering-profile", (*HOLES, "horizons", "topography"))
    return {**t, "topography": _grid(t["topography"])}


def nickel_laterite_profile() -> NickelLateriteProfile:
    """A nickel laterite drilled by 448 vertical holes, 50 m mesh with 25 m infill. Synthetic, CC BY 4.0.

    Returns
    -------
    dict
        ``collars`` (448 rows), ``surveys`` (896), 1 m ``assays`` (10 810: NI_PCT, CO_PCT, FE_PCT, MGO_PCT,
        SIO2_PCT, AL2O3_PCT) and ``horizons`` (1773: HORIZON) tables.
    """
    return _tables("mining/3d/nickel-laterite-profile", (*HOLES, "horizons"))


def tailings_reprocessing() -> TailingsReprocessing:
    """A tailings storage facility drilled by 107 sonic holes. Synthetic, CC BY 4.0.

    Returns
    -------
    dict
        ``collars`` (107 rows), ``surveys`` (214), ``assays`` (4129: AU_GPT, CU_PPM, AS_PPM, CN_WAD_PPM,
        MOISTURE_PCT; ``CN_WAD_BDL`` flags values below detection) and ``lithology`` (3724: LITH) tables;
        ``surface`` and ``original_ground``, BlockModels of 81 x 56 cells of 10 m with Z.
    """
    t = _tables("mining/3d/tailings-reprocessing", (*HOLES, "lithology", "surface", "original_ground"))
    return {**t, "surface": _grid(t["surface"]), "original_ground": _grid(t["original_ground"])}


def porphyry_geometallurgy(deposit: int = 1) -> PorphyryGeometallurgy:
    """Mineralogy, grindability and recovery of three porphyry deposits. CC BY-NC-SA 4.0.

    Parameters
    ----------
    deposit : {1, 2, 3}

    Returns
    -------
    dict
        Deposit 1: ``block_model``, a BlockModel of 49 x 71 x 44 cells of 20 m with air (ton = 0)
        masked; ``correlations`` (7 rows), ``grindability_distribution`` (1366: bwi, -9 is null) and
        ``mineralogical_distributions`` (741) tables; ``pseudo_drillholes`` (3101) and
        ``synthetic_drillholes`` (6817) PointSets.
        Deposits 2 and 3: ``drillholes`` PointSet (4597 and 3467 composites) and ``topography``, a
        BlockModel of 97 x 142 cells of 10 m with midz.
    """
    if deposit not in (1, 2, 3):
        raise InvalidInput(f"deposit must be 1, 2 or 3, got {deposit!r}")
    folder = f"mining/3d/porphyry-geometallurgy/porphyry_0{deposit}"
    mid = {"x": "midx", "y": "midy", "z": "midz"}
    if deposit > 1:
        t = _tables(folder, ("drillholes", "topography"))
        return {
            "drillholes": PointSet.from_table(t["drillholes"], **mid),
            "topography": _grid(t["topography"], ("midx", "midy")),
        }
    t = _tables(
        folder,
        (
            "block_model",
            "correlations",
            "mineralogical_distributions",
            "pseudo_drillholes",
            "synthetic_drillholes",
        ),
    )
    blocks = _grid(t["block_model"], ("x", "y", "z"))
    return {
        "block_model": blocks.mask(blocks["ton"] > 0),
        "correlations": t["correlations"],
        "grindability_distribution": _tables(folder, ("grindability_distribution",), nodata=[-9])[
            "grindability_distribution"
        ],
        "mineralogical_distributions": t["mineralogical_distributions"],
        "pseudo_drillholes": PointSet.from_table(t["pseudo_drillholes"], **mid),
        "synthetic_drillholes": PointSet.from_table(t["synthetic_drillholes"], **mid),
    }


def strebelle() -> BlockModel:
    """The 250 x 250 channel training image: sinuous sand channels along Y in shale.

    Strebelle, S. (2002). Conditional simulation of complex geological structures using multiple-point
    statistics. *Mathematical Geology* 34(1), 1-21. Classic public dataset, under its original terms.

    Returns
    -------
    BlockModel
        250 x 250 x 1 cells of 1 m from the origin (0, 0), with ``facies`` 0 (shale) or 1 (sand); sand is
        28 % of the cells. The spacing is nominal: the image has no physical scale.
    """
    t = read_csv(fetch("oil-gas/2d/strebelle/training_image.csv"))
    return _grid(Table({"X": t["X"], "Y": t["Y"], "facies": np.asarray(t["FACIES"], dtype=np.float64)}))


def f3_seismic() -> BlockModel:
    """A crop of the F3 post-stack seismic cube, Dutch sector of the North Sea.

    Inlines 320-364, crosslines 580-624 and two-way time 1600-1800 ms at 4 ms: 45 x 45 traces of 51
    samples in 25 m bins, dip-steered median-filtered amplitude. Coordinates are in ED50 / UTM zone 31N
    (EPSG:23031), which the SEG-Y file does not state. Released by the Dutch government through TNO and
    published by dGB Earth Sciences under CC BY-SA 3.0; the crop keeps that license.

    Returns
    -------
    BlockModel
        45 x 45 x 51 cells from `read_segy`, rotated to the inline direction, with ``amplitude`` and
        ``crs`` set.
    """
    m = read_segy(fetch("oil-gas/3d/f3-seismic/seismic.sgy"))
    return BlockModel(
        m.origin,
        m.size,
        m.count,
        rotation=tuple(m.rotation),
        attributes=m.attributes,
        index=m.index,
        crs="EPSG:23031",
    )
