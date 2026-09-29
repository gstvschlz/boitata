from collections.abc import Callable, Iterator, Mapping, Sequence
from os import PathLike
from typing import Any, Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt

from ceres.estimation import CategoricalCrossValidation, IndicatorCrossValidation

__version__: str

ArrayLike = npt.ArrayLike
TableLike: TypeAlias = (
    Any  # Arrow PyCapsule object (pyarrow, polars, Table, PointSet) or Mapping[str, ArrayLike]
)
Path: TypeAlias = str | PathLike[str]
Labels: TypeAlias = Sequence[int] | Sequence[str] | ArrayLike
Holes: TypeAlias = Labels
Column: TypeAlias = str
Data: TypeAlias = PointSet | BlockModel | Table | Mapping[str, ArrayLike]
Text: TypeAlias = npt.NDArray[np.object_]  # str, None for null
Label: TypeAlias = str | int | float | bool
PlurigaussianRule: TypeAlias = int | tuple[int, Sequence[PlurigaussianRule]]

class Table:
    def __init__(self, data: TableLike | Mapping[str, ArrayLike]) -> None: ...
    @property
    def num_rows(self) -> int: ...
    @property
    def column_names(self) -> list[str]: ...
    def column(self, name: str) -> npt.NDArray[np.float64] | npt.NDArray[np.bool_] | Text: ...
    def __getitem__(self, name: str) -> npt.NDArray[np.float64] | npt.NDArray[np.bool_] | Text: ...
    def filter(self, mask: npt.NDArray[np.bool_]) -> Table: ...
    def __len__(self) -> int: ...
    def to_polars(self) -> Any: ...
    def to_pyarrow(self) -> Any: ...
    def to_pandas(self) -> Any: ...
    def __arrow_c_schema__(self) -> object: ...
    def __arrow_c_array__(self, requested_schema: object | None = None) -> tuple[object, object]: ...
    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...

class PointSet:
    def __init__(
        self,
        coords: ArrayLike,
        attributes: TableLike | Mapping[str, ArrayLike] | None = None,
        *,
        crs: str | None = None,
    ) -> None: ...
    @staticmethod
    def from_table(
        table: TableLike, *, x: str = "X", y: str = "Y", z: str | None = None, crs: str | None = None
    ) -> PointSet: ...
    @property
    def coords(self) -> npt.NDArray[np.float64]: ...
    @property
    def attributes(self) -> Table: ...
    @property
    def crs(self) -> str | None: ...
    def to_table(self) -> Table: ...
    def __getitem__(self, name: str) -> npt.NDArray[np.float64] | Text: ...
    def __len__(self) -> int: ...
    def with_column(self, name: str, values: ArrayLike | Sequence[str | None]) -> PointSet: ...
    def with_columns(self, data: Mapping[str, ArrayLike] | TableLike) -> PointSet: ...
    def filter(self, mask: npt.NDArray[np.bool_]) -> PointSet: ...
    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...

class Polylines:
    def __init__(
        self,
        parts: Sequence[ArrayLike],
        *,
        closed: bool | Sequence[bool] = False,
        features: Sequence[int] | None = None,
        attributes: TableLike | Mapping[str, ArrayLike] | None = None,
        crs: str | None = None,
    ) -> None: ...
    @property
    def vertices(self) -> npt.NDArray[np.float64]: ...
    @property
    def parts(self) -> list[npt.NDArray[np.float64]]: ...
    @property
    def closed(self) -> npt.NDArray[np.bool_]: ...
    @property
    def feature(self) -> npt.NDArray[np.int64]: ...
    @property
    def attributes(self) -> Table: ...
    @property
    def crs(self) -> str | None: ...
    def with_column(self, name: str, values: ArrayLike | Sequence[str]) -> Polylines: ...
    def to_points(self) -> PointSet: ...
    def contains(
        self, points: PointSet | BlockModel | ArrayLike, *, feature: int | None = None
    ) -> npt.NDArray[np.bool_]: ...
    def locate(self, points: PointSet | BlockModel | ArrayLike) -> npt.NDArray[np.int64]: ...
    def distance(
        self,
        points: PointSet | BlockModel | ArrayLike,
        *,
        feature: int | None = None,
        signed: bool = False,
    ) -> npt.NDArray[np.float64]: ...
    def length(self) -> npt.NDArray[np.float64]: ...
    def area(self) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def from_table(
        table: TableLike,
        *,
        feature: str = "ID",
        x: str = "X",
        y: str = "Y",
        z: str | None = None,
        part: str | None = None,
        closed: bool = False,
        crs: str | None = None,
    ) -> Polylines: ...
    def to_table(self) -> Table: ...
    def __getitem__(self, name: str) -> npt.NDArray[np.float64] | Text: ...
    def __len__(self) -> int: ...
    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...

class BlockModel:
    def __init__(
        self,
        origin: Sequence[float],
        size: Sequence[float],
        count: Sequence[int],
        *,
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        attributes: TableLike | Mapping[str, ArrayLike] | None = None,
        index: npt.NDArray[np.uint64] | None = None,
        crs: str | None = None,
    ) -> None: ...
    @property
    def origin(self) -> list[float]: ...
    @property
    def size(self) -> list[float]: ...
    @property
    def count(self) -> list[int]: ...
    @property
    def rotation(self) -> list[float]: ...
    @property
    def crs(self) -> str | None: ...
    @property
    def index(self) -> npt.NDArray[np.uint64] | None: ...
    @property
    def centroids(self) -> npt.NDArray[np.float64]: ...
    @property
    def corners(self) -> npt.NDArray[np.float64]: ...
    def row_at(self, points: ArrayLike) -> npt.NDArray[np.int64]: ...
    @property
    def attributes(self) -> Table: ...
    def mask(self, keep: npt.NDArray[np.bool_]) -> BlockModel: ...
    def to_regular(self) -> BlockModel: ...
    def regularize(self, target: BlockModel, *, min_fraction: float = 0.0) -> BlockModel: ...
    def subblock(
        self,
        meshes: Sequence[
            tuple[Mesh, Literal["inside", "below", "above"], str] | tuple[Polylines, Literal["inside"], str]
        ],
        subgrid: int | Sequence[int],
        *,
        column: str = "domain",
        fill: str | None = None,
    ) -> BlockModel: ...
    @staticmethod
    def from_meshes(
        origin: Sequence[float],
        size: Sequence[float],
        count: Sequence[int],
        meshes: Sequence[
            tuple[Mesh, Literal["inside", "below", "above"], str] | tuple[Polylines, Literal["inside"], str]
        ],
        subgrid: int | Sequence[int],
        *,
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        column: str = "domain",
        fill: str | None = None,
        crs: str | None = None,
    ) -> BlockModel: ...
    @staticmethod
    def from_extents(
        *objects: PointSet | Drillholes | Mesh | Polylines | BlockModel | ArrayLike,
        size: Sequence[float | None],
        buffer: float | Sequence[float] = 0.0,
        rotation: tuple[float, float, float] | None = None,
        snap: bool | float | Sequence[float] = False,
        crs: str | None = None,
    ) -> BlockModel: ...
    def discretize(self, n: int | Sequence[int]) -> BlockModel: ...
    def with_column(self, name: str, values: ArrayLike | Sequence[str | None]) -> BlockModel: ...
    def with_columns(self, data: Mapping[str, ArrayLike] | TableLike) -> BlockModel: ...
    @property
    def extents(self) -> npt.NDArray[np.float64] | None: ...
    @property
    def volumes(self) -> npt.NDArray[np.float64]: ...
    @staticmethod
    def subblocked(
        origin: Sequence[float],
        size: Sequence[float],
        count: Sequence[int],
        parent: npt.NDArray[np.uint64],
        extents: ArrayLike,
        *,
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        subgrid: tuple[int, int, int] | None = None,
        attributes: TableLike | Mapping[str, ArrayLike] | None = None,
        crs: str | None = None,
    ) -> BlockModel: ...
    def to_table(self) -> Table: ...
    def __getitem__(self, name: str) -> npt.NDArray[np.float64] | Text: ...
    def __len__(self) -> int: ...
    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...

def read_csv(
    path: Path, *, nodata: Sequence[float | str] | None = None, delimiter: str = ",", progress: bool = True
) -> Table: ...
def write_csv(path: Path, table: TableLike, *, progress: bool = True) -> None: ...
def read_gslib(
    path: Path, *, nodata: Sequence[float | str] | None = None, progress: bool = True
) -> Table: ...
def write_gslib(path: Path, table: TableLike, *, nodata: float = -999.0, progress: bool = True) -> None: ...

class NormalScore:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> NormalScore: ...
    def __init__(
        self,
        *,
        tails: tuple[float, float] | None = None,
        reference: KernelDensity | GaussianMixture | None = None,
    ) -> None: ...
    def fit(
        self,
        values: ArrayLike,
        *,
        weights: ArrayLike | None = None,
        censored: ArrayLike | None = None,
        seed: int = 0,
    ) -> NormalScore: ...
    def fit_transform(
        self,
        values: ArrayLike,
        *,
        weights: ArrayLike | None = None,
        censored: ArrayLike | None = None,
        seed: int = 0,
    ) -> npt.NDArray[np.float64]: ...
    def transform(self, values: ArrayLike) -> npt.NDArray[np.float64]: ...
    def inverse_transform(self, scores: ArrayLike) -> npt.NDArray[np.float64]: ...
    @property
    def table_(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

class Capping:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Capping: ...
    def __init__(
        self,
        *,
        cap: float | Mapping[int | str, float] | None = None,
        quantile: float | None = None,
        metal_removed: float | None = None,
        cv: float | None = None,
    ) -> None: ...
    def fit(
        self,
        values: ArrayLike | Column,
        *,
        domains: Labels | None = None,
        domain_column: Column | None = None,
        weights: ArrayLike | Column | None = None,
        data: Data | None = None,
    ) -> Capping: ...
    def fit_transform(
        self,
        values: ArrayLike | Column,
        *,
        domains: Labels | None = None,
        domain_column: Column | None = None,
        weights: ArrayLike | Column | None = None,
        data: Data | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def transform(
        self,
        values: ArrayLike | Column,
        *,
        domains: Labels | None = None,
        domain_column: Column | None = None,
        data: Data | None = None,
    ) -> npt.NDArray[np.float64]: ...
    @property
    def caps_(self) -> float | dict[int | str, float]: ...
    @property
    def metal_removed_(self) -> float | dict[int | str, float]: ...

class HermiteAnamorphosis:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> HermiteAnamorphosis: ...
    def __init__(self, *, degree: int = 30) -> None: ...
    def fit(self, values: ArrayLike, *, weights: ArrayLike | None = None) -> HermiteAnamorphosis: ...
    def fit_transform(
        self, values: ArrayLike, *, weights: ArrayLike | None = None
    ) -> npt.NDArray[np.float64]: ...
    def transform(self, values: ArrayLike) -> npt.NDArray[np.float64]: ...
    def inverse_transform(self, scores: ArrayLike) -> npt.NDArray[np.float64]: ...
    @property
    def coefficients_(self) -> npt.NDArray[np.float64]: ...
    @property
    def mean_(self) -> float: ...
    @property
    def variance_(self) -> float: ...
    def block(self, r: float) -> HermiteAnamorphosis: ...
    def grade_tonnage(self, cutoffs: ArrayLike) -> Table: ...

class BoxCox:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> BoxCox: ...
    def __init__(self, *, lambda_: float | None = None) -> None: ...
    def fit(self, values: ArrayLike) -> BoxCox: ...
    def fit_transform(self, values: ArrayLike) -> npt.NDArray[np.float64]: ...
    @property
    def lambda_(self) -> float: ...
    def transform(self, values: ArrayLike) -> npt.NDArray[np.float64]: ...
    def inverse_transform(self, values: ArrayLike) -> npt.NDArray[np.float64]: ...

class PPMT:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> PPMT: ...
    def __init__(
        self, *, iterations: int = 30, candidates: int = 60, seed: int = 0, marginal: bool = True
    ) -> None: ...
    def fit(self, data: ArrayLike, *, weights: ArrayLike | None = None) -> PPMT: ...
    def fit_transform(
        self, data: ArrayLike, *, weights: ArrayLike | None = None
    ) -> npt.NDArray[np.float64]: ...
    def transform(self, data: ArrayLike) -> npt.NDArray[np.float64]: ...
    def inverse_transform(self, data: ArrayLike) -> npt.NDArray[np.float64]: ...

class GaussianImputer:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> GaussianImputer: ...
    def __init__(
        self,
        *,
        components: int | None = 1,
        seed: int = 0,
        spatial: Variogram | None = None,
        neighbors: int = 16,
    ) -> None: ...
    def fit(
        self,
        data: ArrayLike,
        *,
        coords: ArrayLike | PointSet | None = None,
        weights: ArrayLike | None = None,
    ) -> GaussianImputer: ...
    def fit_transform(
        self,
        data: ArrayLike,
        *,
        coords: ArrayLike | PointSet | None = None,
        weights: ArrayLike | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def transform(
        self, data: ArrayLike, *, coords: ArrayLike | PointSet | None = None
    ) -> npt.NDArray[np.float64]: ...
    @property
    def correlation_(self) -> npt.NDArray[np.float64]: ...

class KernelDensity:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> KernelDensity: ...
    def __init__(
        self,
        *,
        bandwidth: Literal["silverman", "scott"] | float | None = None,
        lower: float | None = None,
        upper: float | None = None,
        log: bool = False,
    ) -> None: ...
    def fit(
        self,
        values: ArrayLike | Column,
        *,
        weights: ArrayLike | Column | None = None,
        data: Data | None = None,
    ) -> KernelDensity: ...
    def pdf(self, x: ArrayLike) -> npt.NDArray[np.float64]: ...
    def cdf(self, x: ArrayLike) -> npt.NDArray[np.float64]: ...
    def quantile(self, p: ArrayLike) -> npt.NDArray[np.float64]: ...
    def sample(self, n: int, *, seed: int = 0) -> npt.NDArray[np.float64]: ...
    @property
    def bandwidth_(self) -> float: ...

class GaussianMixture:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> GaussianMixture: ...
    def __init__(self, *, components: int | None = None, max_components: int = 6, seed: int = 0) -> None: ...
    def fit(self, data: ArrayLike, *, weights: ArrayLike | None = None) -> GaussianMixture: ...
    def pdf(self, data: ArrayLike) -> npt.NDArray[np.float64]: ...
    def predict(self, data: ArrayLike) -> npt.NDArray[np.int64]: ...
    def sample(self, n: int, *, seed: int = 0) -> npt.NDArray[np.float64]: ...
    @property
    def proportions_(self) -> npt.NDArray[np.float64]: ...
    @property
    def means_(self) -> npt.NDArray[np.float64]: ...
    @property
    def covariances_(self) -> npt.NDArray[np.float64]: ...
    @property
    def bic_(self) -> dict[int, float]: ...

class PCA:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> PCA: ...
    def __init__(self, *, standardize: bool = False) -> None: ...
    def fit(self, data: ArrayLike, *, weights: ArrayLike | None = None) -> PCA: ...
    def fit_transform(
        self, data: ArrayLike, *, weights: ArrayLike | None = None
    ) -> npt.NDArray[np.float64]: ...
    def transform(self, data: ArrayLike) -> npt.NDArray[np.float64]: ...
    def inverse_transform(self, data: ArrayLike) -> npt.NDArray[np.float64]: ...
    @property
    def components_(self) -> npt.NDArray[np.float64]: ...
    @property
    def explained_variance_(self) -> npt.NDArray[np.float64]: ...
    @property
    def explained_variance_ratio_(self) -> npt.NDArray[np.float64]: ...

class MAF:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> MAF: ...
    def __init__(self, lag: float, *, tolerance: float | None = None) -> None: ...
    def fit(self, data: ArrayLike, coords: ArrayLike) -> MAF: ...
    def fit_transform(self, data: ArrayLike, coords: ArrayLike) -> npt.NDArray[np.float64]: ...
    def transform(self, data: ArrayLike) -> npt.NDArray[np.float64]: ...
    def inverse_transform(self, data: ArrayLike) -> npt.NDArray[np.float64]: ...
    @property
    def gammas_(self) -> npt.NDArray[np.float64]: ...

class StepwiseConditional:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> StepwiseConditional: ...
    def __init__(self, *, classes: int = 10) -> None: ...
    def fit(self, data: ArrayLike, *, weights: ArrayLike | None = None) -> StepwiseConditional: ...
    def fit_transform(
        self, data: ArrayLike, *, weights: ArrayLike | None = None
    ) -> npt.NDArray[np.float64]: ...
    def transform(self, data: ArrayLike) -> npt.NDArray[np.float64]: ...
    def inverse_transform(self, data: ArrayLike) -> npt.NDArray[np.float64]: ...

class UniformConditioning:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> UniformConditioning: ...
    def __init__(
        self, anamorphosis: HermiteAnamorphosis, r_smu: float, *, r_panel: float | None = None
    ) -> None: ...
    def panel_recovery(
        self, panel_grade: float, cutoffs: ArrayLike, *, estimate_variance: float | None = None
    ) -> Table: ...
    def localized_grades(
        self, panel_grade: float, n_smu: int, *, estimate_variance: float | None = None
    ) -> npt.NDArray[np.float64]: ...
    def grade_tonnage(
        self,
        panels: BlockModel,
        grade: Column,
        cutoffs: ArrayLike,
        *,
        estimate_variance: Column | None = None,
        density: float | ArrayLike | Column = 1.0,
    ) -> Table: ...
    def localize(
        self,
        smus: BlockModel,
        ranking: Column,
        panels: BlockModel,
        grade: Column,
        *,
        estimate_variance: Column | None = None,
        name: str = "localized",
    ) -> BlockModel: ...

class Trend:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Trend: ...
    @property
    def degree(self) -> int | None: ...
    @property
    def coefficients(self) -> npt.NDArray[np.float64] | None: ...
    @property
    def bandwidth(self) -> float | None: ...
    @property
    def bandwidths(self) -> npt.NDArray[np.float64] | None: ...
    @property
    def scores(self) -> npt.NDArray[np.float64] | None: ...
    @property
    def categories(self) -> list[str] | None: ...
    def predict(self, coords: ArrayLike | PointSet | BlockModel) -> npt.NDArray[np.float64] | Table: ...

class Declustering:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Declustering: ...
    @property
    def mean(self) -> float: ...
    @property
    def cell_size(self) -> float: ...
    @property
    def weights(self) -> npt.NDArray[np.float64]: ...
    @property
    def sizes(self) -> npt.NDArray[np.float64]: ...
    @property
    def means(self) -> npt.NDArray[np.float64]: ...

def detrend(
    coords: ArrayLike | PointSet | BlockModel,
    values: ArrayLike | Column,
    *,
    degree: int = 1,
    bandwidth: float | Sequence[float] | None = None,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ratios: tuple[float, float] = (1.0, 1.0),
    weights: ArrayLike | Column | None = None,
    categorical: bool = False,
    scheme: Categories | None = None,
) -> tuple[Trend, npt.NDArray[np.float64] | Table]: ...
def cell_declustering(
    coords: ArrayLike | PointSet | BlockModel,
    values: ArrayLike | Column,
    *,
    cell_size: float | None = None,
    sizes: ArrayLike | None = None,
    offsets: int = 25,
    minimize: bool = True,
) -> Declustering: ...
def polygon_declustering(
    coords: ArrayLike | PointSet | BlockModel, values: ArrayLike | Column, *, nodes: int = 10_000
) -> Declustering: ...
@overload
def despike(
    coords: ArrayLike | PointSet | BlockModel,
    values: Sequence[Column],
    *,
    radii: Sequence[float] | None = None,
    seed: int = 0,
) -> Table: ...
@overload
def despike(
    coords: ArrayLike | PointSet | BlockModel,
    values: ArrayLike | Column,
    *,
    radii: Sequence[float] | None = None,
    seed: int = 0,
) -> npt.NDArray[np.float64]: ...
def spatial_bootstrap(
    coords: ArrayLike | PointSet | BlockModel,
    values: ArrayLike | Column,
    variogram: Variogram,
    *,
    weights: ArrayLike | Column | None = None,
    n: int = 100,
    seed: int = 0,
    quantiles: Sequence[float] = (),
    cutoffs: Sequence[float] = (),
) -> Table: ...
def affine_correction(
    values: ArrayLike, f: float, *, weights: ArrayLike | None = None
) -> npt.NDArray[np.float64]: ...
def indirect_lognormal_correction(
    values: ArrayLike, f: float, *, weights: ArrayLike | None = None
) -> npt.NDArray[np.float64]: ...
def upscale(
    coords: ArrayLike | PointSet | BlockModel,
    values: ArrayLike | Column,
    size: Sequence[float],
    *,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def downscale(
    center: Sequence[float], size: Sequence[float], value: float, refine: tuple[int, int, int]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def normal_cdf(x: ArrayLike) -> npt.NDArray[np.float64]: ...
def normal_ppf(p: ArrayLike) -> npt.NDArray[np.float64]: ...

_Limit: TypeAlias = float | tuple[float, float]

class Structure:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Structure: ...
    def __init__(
        self,
        model: str,
        sill: float,
        range: float,
        *,
        order: float | None = None,
        exponent: float | None = None,
    ) -> None: ...
    @property
    def model(self) -> str: ...
    @property
    def sill(self) -> float: ...
    @property
    def range(self) -> float: ...

class Variogram:
    def __init__(
        self,
        structures: Sequence[Structure | tuple[str, float, float]],
        *,
        nugget: float = 0.0,
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        ratios: tuple[float, float] = (1.0, 1.0),
    ) -> None: ...
    @staticmethod
    def fit(
        experimental: ExperimentalVariogram,
        model: str | Sequence[str] = "spherical",
        *,
        weighting: str = "count",
        nugget: _Limit | None = None,
        sills: Sequence[_Limit | None] | None = None,
        ranges: Sequence[_Limit | None] | None = None,
    ) -> Variogram: ...
    @staticmethod
    def fit_directional(
        experimentals: Sequence[ExperimentalVariogram],
        directions: Sequence[tuple[float, float]],
        model: str | Sequence[str] = "spherical",
        *,
        weighting: str = "count",
        nugget: _Limit | None = None,
        sills: Sequence[_Limit | None] | None = None,
        ranges: Sequence[_Limit | None] | None = None,
        rotation: Sequence[_Limit | None] | None = None,
        ratios: Sequence[_Limit | None] | None = None,
    ) -> Variogram: ...
    def with_anisotropy(
        self, rotation: tuple[float, float, float], ratios: tuple[float, float]
    ) -> Variogram: ...
    @property
    def nugget(self) -> float: ...
    @property
    def structures(self) -> list[Structure]: ...
    @property
    def rotation(self) -> tuple[float, float, float]: ...
    @property
    def ratios(self) -> tuple[float, float]: ...
    @property
    def sill(self) -> float: ...
    def gamma(self, h: ArrayLike) -> npt.NDArray[np.float64]: ...
    def covariance(self, h: ArrayLike) -> npt.NDArray[np.float64]: ...
    def gamma_between(self, a: ArrayLike, b: ArrayLike) -> npt.NDArray[np.float64]: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Variogram: ...

class ExperimentalVariogram:
    @property
    def lags(self) -> npt.NDArray[np.float64]: ...
    @property
    def gammas(self) -> npt.NDArray[np.float64]: ...
    @property
    def counts(self) -> npt.NDArray[np.float64]: ...
    @property
    def covariances(self) -> npt.NDArray[np.float64] | None: ...
    def fit(
        self,
        model: str | Sequence[str] = "spherical",
        *,
        weighting: str = "count",
        nugget: _Limit | None = None,
        sills: Sequence[_Limit | None] | None = None,
        ranges: Sequence[_Limit | None] | None = None,
    ) -> Variogram: ...
    def nugget(self, *, lags: int = 3) -> float: ...

class VariogramSet:
    @property
    def nvar(self) -> int: ...
    @property
    def names(self) -> list[str | None]: ...
    @property
    def directions(self) -> list[tuple[float, float]] | None: ...
    def __getitem__(
        self, key: tuple[int | str, int | str]
    ) -> ExperimentalVariogram | list[ExperimentalVariogram]: ...

class VariogramMap:
    lags: npt.NDArray[np.float64]
    angles: npt.NDArray[np.float64]
    gammas: npt.NDArray[np.float64]
    counts: npt.NDArray[np.float64]
    ranges: npt.NDArray[np.float64]

class VariogramVolume:
    lags: npt.NDArray[np.float64]
    gammas: npt.NDArray[np.float64]
    counts: npt.NDArray[np.float64]
    directions: npt.NDArray[np.float64]
    direction_ranges: npt.NDArray[np.float64]
    rotation: tuple[float, float, float]
    axes: list[tuple[float, float]]
    ranges: tuple[float, float, float]
    ratios: tuple[float, float]

class Coregionalization:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Coregionalization: ...
    def __init__(
        self,
        nugget: Sequence[Sequence[float]],
        *,
        structures: Sequence[tuple[str, float, Sequence[Sequence[float]]]],
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        ratios: tuple[float, float] = (1.0, 1.0),
    ) -> None: ...
    @staticmethod
    def fit(
        experimentals: VariogramSet
        | Sequence[Sequence[ExperimentalVariogram | Sequence[ExperimentalVariogram] | None]],
        model: str | Sequence[str] = "spherical",
        *,
        weighting: str = "count",
        nugget: bool = True,
        ranges: Sequence[_Limit | None] | None = None,
        directions: Sequence[tuple[float, float]] | None = None,
        rotation: Sequence[_Limit | None] | None = None,
        ratios: Sequence[_Limit | None] | None = None,
        intrinsic: bool = False,
    ) -> Coregionalization: ...
    @staticmethod
    def intrinsic(variogram: Variogram, covariance: ArrayLike) -> Coregionalization: ...
    @property
    def nvar(self) -> int: ...
    @property
    def nugget(self) -> npt.NDArray[np.float64]: ...
    @property
    def structures(self) -> list[tuple[str, float, npt.NDArray[np.float64]]]: ...
    @property
    def rotation(self) -> tuple[float, float, float]: ...
    @property
    def ratios(self) -> tuple[float, float]: ...
    def cross_covariance(self, i: int, j: int, a: ArrayLike, b: ArrayLike) -> npt.NDArray[np.float64]: ...

class Transiogram:
    def __init__(self, proportions: Sequence[float], range: float) -> None: ...
    def matrix(self, h: float) -> npt.NDArray[np.float64]: ...

def experimental_variogram(
    coords: ArrayLike | PointSet | BlockModel,
    values: ArrayLike | Column,
    lag: float,
    max_lag: float,
    *,
    azimuth: float | None = None,
    dip: float = 0.0,
    tolerance: float = 22.5,
    bandwidth: float | None = None,
    estimator: str = "matheron",
    standardize: bool = False,
    other: ArrayLike | Column | None = None,
    other_coords: ArrayLike | PointSet | BlockModel | None = None,
    holes: Holes | Column | None = None,
    method: str | None = None,
    anisotropy: LocalAnisotropy | None = None,
) -> ExperimentalVariogram: ...
def dissemination(
    madogram: ExperimentalVariogram, variogram: ExperimentalVariogram
) -> npt.NDArray[np.float64]: ...
def experimental_variograms(
    coords: ArrayLike | PointSet | BlockModel,
    values: Sequence[ArrayLike | Column],
    lag: float,
    max_lag: float,
    *,
    directions: Sequence[tuple[float, float]] | None = None,
    tolerance: float = 22.5,
    bandwidth: float | None = None,
    estimator: str = "matheron",
    standardize: bool = False,
    method: str | None = None,
) -> VariogramSet: ...
def _realization_variograms(
    coords: ArrayLike | PointSet | BlockModel,
    realizations: ArrayLike,
    lag: float,
    max_lag: float,
    *,
    directions: Sequence[tuple[float, float]] | None = None,
    tolerance: float = 22.5,
    bandwidth: float | None = None,
    method: str | None = None,
) -> list[list[ExperimentalVariogram]]: ...
def variogram_map(
    coords: ArrayLike | PointSet | BlockModel,
    values: ArrayLike | Column,
    lag: float,
    max_lag: float,
    *,
    u: Sequence[float] = (1.0, 0.0, 0.0),
    v: Sequence[float] = (0.0, 1.0, 0.0),
    tolerance: float = 22.5,
    steps: int = 36,
    model: str = "spherical",
    estimator: str = "matheron",
) -> VariogramMap: ...
def variogram_volume(
    coords: ArrayLike | PointSet | BlockModel,
    values: ArrayLike | Column,
    lag: float,
    max_lag: float,
    *,
    tolerance: float = 20.0,
    directions: int = 200,
    model: str = "spherical",
    estimator: str = "matheron",
) -> VariogramVolume: ...
def experimental_transiogram(
    coords: ArrayLike | PointSet | BlockModel,
    categories: Sequence[int] | ArrayLike | Column,
    lag: float,
    max_lag: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def change_of_support(
    anamorphosis: HermiteAnamorphosis,
    variogram: Variogram,
    size: Sequence[float],
    *,
    discretization: tuple[int, int, int] = (4, 4, 1),
) -> tuple[float, HermiteAnamorphosis]: ...
def block_correlation(
    variogram: Variogram, size: Sequence[float], *, discretization: tuple[int, int, int] = (4, 4, 1)
) -> float: ...

class HighGrade:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> HighGrade: ...
    def __init__(
        self,
        threshold: float,
        radius: float | tuple[float, float, float],
        *,
        rotation: tuple[float, float, float] | None = None,
        mode: Literal["drop", "clamp"] = "drop",
    ) -> None: ...
    @property
    def threshold(self) -> float: ...
    @property
    def radius(self) -> float: ...
    @property
    def ranges(self) -> tuple[float, float, float] | None: ...
    @property
    def rotation(self) -> tuple[float, float, float] | None: ...
    @property
    def mode(self) -> Literal["drop", "clamp"]: ...

class Search:
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Search: ...
    def __init__(
        self,
        radius: float,
        *,
        max_samples: int = 16,
        min_samples: int = 1,
        octant: bool = False,
        max_per_hole: int | None = None,
        rotation: tuple[float, float, float] | None = None,
        ratios: tuple[float, float] | None = None,
        high_grade: HighGrade | tuple[float, float] | None = None,
        soft: float | Mapping[tuple[Label, Label], float] | None = None,
    ) -> None: ...
    @property
    def radius(self) -> float: ...
    @property
    def max_samples(self) -> int: ...
    @property
    def min_samples(self) -> int: ...
    @property
    def octant(self) -> bool: ...
    @property
    def max_per_hole(self) -> int | None: ...
    @property
    def high_grade(self) -> HighGrade | None: ...
    @property
    def rotation(self) -> tuple[float, float, float] | None: ...
    @property
    def ratios(self) -> tuple[float, float] | None: ...
    @property
    def soft(self) -> float | dict[tuple[Label, Label], float] | None: ...

class _Estimator:
    def to_parquet(self, path: Path, name: str) -> None: ...
    @staticmethod
    def from_parquet(path: Path, name: str) -> _Estimator: ...
    def __init__(
        self,
        method: str,
        search: Search | Sequence[Search],
        variogram: Variogram | None = None,
        **options: Any,
    ) -> None: ...
    def fit(
        self,
        coords: ArrayLike | PointSet | BlockModel,
        values: ArrayLike | Column,
        *,
        holes: Holes | Column | None = None,
        error_variance: ArrayLike | Column | None = None,
        domains: Labels | None = None,
        domain_column: Column | None = None,
    ) -> _Estimator: ...
    @property
    def values(self) -> npt.NDArray[np.float64]: ...
    def predict(
        self,
        targets: Any,
        *,
        return_variance: bool = False,
        anisotropy: LocalAnisotropy | None = None,
        diagnostics: bool = False,
        domains: Label | Labels | None = None,
        domain_column: Column | None = None,
        progress: bool = True,
    ) -> Any: ...
    def cross_validate(
        self, *, folds: int | None = None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def with_search(self, search: Search | Sequence[Search]) -> _Estimator: ...
    def _point_support(self) -> _Estimator: ...
    def _declustering(
        self, coords: ArrayLike | PointSet | BlockModel, values: ArrayLike | Column, targets: Any
    ) -> Declustering: ...
    @property
    def variogram(self) -> Variogram | None: ...
    @property
    def _sample_domains(self) -> list[Label] | None: ...

class DualKriging:
    def to_parquet(self, path: Path) -> None: ...
    @staticmethod
    def from_parquet(path: Path) -> DualKriging: ...
    def __init__(self, variogram: Variogram, *, degree: int = 0) -> None: ...
    def fit(self, coords: ArrayLike | PointSet | BlockModel, values: ArrayLike | Column) -> DualKriging: ...
    def predict(self, targets: Any) -> npt.NDArray[np.float64]: ...

def neighborhood_stats(
    targets: Any,
    coords: ArrayLike | PointSet | BlockModel,
    values: ArrayLike | Column,
    *,
    k: int = 8,
    radius: float = ...,
    variogram: Variogram | None = None,
    holes: Holes | Column | None = None,
) -> Table: ...
def hole_distance(
    targets: Any,
    coords: ArrayLike | PointSet | BlockModel,
    holes: Holes | Column,
    n: int | Sequence[int],
    *,
    search: Search | None = None,
    domains: tuple[Labels, Labels] | None = None,
    domain_column: Column | tuple[Column, Column] | None = None,
) -> npt.NDArray[np.float64]: ...

class SGS:
    def to_parquet(self, path: Path) -> None: ...
    @staticmethod
    def from_parquet(path: Path) -> SGS: ...
    def __init__(
        self, variogram: Variogram, search: Search | Sequence[Search], *, classes: int = 10
    ) -> None: ...
    @property
    def correlation(self) -> float | None: ...
    def fit(
        self,
        coords: ArrayLike | PointSet | BlockModel,
        values: ArrayLike | Column,
        *,
        weights: ArrayLike | Column | None = None,
        holes: Holes | Column | None = None,
        trend: ArrayLike | Column | None = None,
        domains: Label | Sequence[Label] | ArrayLike | None = None,
        domain_column: Column | None = None,
        secondary: ArrayLike | Column | None = None,
        correlation: float | None = None,
    ) -> SGS: ...
    def passes(
        self,
        targets: Any,
        *,
        anisotropy: LocalAnisotropy | None = None,
        domains: Label | Sequence[Label] | ArrayLike | None = None,
        domain_column: Column | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def simulate(
        self,
        targets: Any,
        *,
        n: int = 100,
        seed: int = 0,
        cutoffs: Sequence[float] = (),
        quantiles: Sequence[float] = (),
        keep: bool | Sequence[int] = False,
        anisotropy: LocalAnisotropy | None = None,
        blocks: BlockModel | None = None,
        trend: ArrayLike | Column | None = None,
        domains: Label | Sequence[Label] | ArrayLike | None = None,
        domain_column: Column | None = None,
        secondary: ArrayLike | Column | None = None,
        path: Literal["shared", "random"] | None = None,
        batch: int | None = None,
        progress: bool = True,
    ) -> SimulationSummary: ...

class TurningBands:
    def to_parquet(self, path: Path) -> None: ...
    @staticmethod
    def from_parquet(path: Path) -> TurningBands: ...
    def __init__(
        self,
        variogram: Variogram,
        *,
        bands: int = 300,
        step: float | None = None,
        search: Search | None = None,
        classes: int = 10,
    ) -> None: ...
    def fit(
        self,
        coords: ArrayLike | PointSet | BlockModel,
        values: ArrayLike | Column,
        *,
        weights: ArrayLike | Column | None = None,
        holes: Holes | Column | None = None,
        trend: ArrayLike | Column | None = None,
        domains: Label | Sequence[Label] | ArrayLike | None = None,
        domain_column: Column | None = None,
    ) -> TurningBands: ...
    def simulate(
        self,
        targets: Any,
        *,
        n: int = 100,
        seed: int = 0,
        cutoffs: Sequence[float] = (),
        quantiles: Sequence[float] = (),
        keep: bool | Sequence[int] = False,
        blocks: BlockModel | None = None,
        trend: ArrayLike | Column | None = None,
        domains: Label | Sequence[Label] | ArrayLike | None = None,
        domain_column: Column | None = None,
        progress: bool = True,
    ) -> SimulationSummary: ...
    def simulate_to_parquet(
        self,
        path: Path,
        out: Path,
        *,
        n: int = 100,
        seed: int = 0,
        cutoffs: Sequence[float] = (),
        quantiles: Sequence[float] = (),
        keep: bool | Sequence[int] = False,
        rows: int = 1_000_000,
        domains: Label | Sequence[Label] | ArrayLike | None = None,
        domain_column: Column | None = None,
        trend: Column | None = None,
        discretization: tuple[int, int, int] | None = None,
        progress: bool = True,
    ) -> dict[str, npt.NDArray[np.float64]]: ...

class MultivariateSimulation:
    def __init__(
        self,
        transform: PCA | MAF | StepwiseConditional | PPMT,
        simulators: Sequence[SGS | TurningBands],
    ) -> None: ...
    def fit(
        self,
        coords: ArrayLike | PointSet | BlockModel,
        data: ArrayLike | Sequence[Column],
        *,
        weights: ArrayLike | Column | None = None,
        holes: Holes | Column | None = None,
        impute: bool | GaussianImputer | None = None,
    ) -> MultivariateSimulation: ...
    def simulate(
        self,
        targets: Any,
        *,
        n: int = 100,
        seed: int = 0,
        cutoffs: Sequence[float] = (),
        quantiles: Sequence[float] = (),
        keep: bool | Sequence[int] = False,
        anisotropy: LocalAnisotropy | None = None,
        blocks: BlockModel | None = None,
        progress: bool = True,
    ) -> list[SimulationSummary]: ...

class SIS:
    def to_parquet(self, path: Path) -> None: ...
    @staticmethod
    def from_parquet(path: Path) -> SIS: ...
    def __init__(self, variograms: Sequence[Variogram], search: Search) -> None: ...
    def fit(
        self,
        coords: ArrayLike | PointSet | BlockModel,
        categories: ArrayLike | Column,
        *,
        holes: Holes | Column | None = None,
        proportions: ArrayLike | Table | None = None,
    ) -> SIS: ...
    def simulate(
        self,
        targets: Any,
        *,
        n: int = 100,
        seed: int = 0,
        keep: bool | Sequence[int] = False,
        blocks: BlockModel | None = None,
        proportions: ArrayLike | Table | None = None,
        progress: bool = True,
    ) -> CategoricalSummary: ...

class Plurigaussian:
    def to_parquet(self, path: Path) -> None: ...
    @staticmethod
    def from_parquet(path: Path) -> Plurigaussian: ...
    def __init__(
        self,
        variograms: Variogram | Sequence[Variogram],
        *,
        proportions: Sequence[float] | None = None,
        rule: PlurigaussianRule | None = None,
        regions: Sequence[tuple[Sequence[tuple[float, float]], int]] | None = None,
    ) -> None: ...
    @property
    def variograms(self) -> list[Variogram]: ...
    def fit_variograms(self, experimental: Sequence[ExperimentalVariogram | None]) -> Plurigaussian: ...
    def indicator_variograms(self, lags: ArrayLike) -> npt.NDArray[np.float64]: ...
    def fit(
        self,
        coords: ArrayLike | PointSet | BlockModel,
        categories: ArrayLike | Column,
        *,
        holes: Holes | Column | None = None,
        proportions: ArrayLike | Table | None = None,
    ) -> Plurigaussian: ...
    def simulate(
        self,
        targets: Any,
        *,
        n: int = 100,
        seed: int = 0,
        keep: bool | Sequence[int] = False,
        blocks: BlockModel | None = None,
        proportions: ArrayLike | Table | None = None,
        progress: bool = True,
    ) -> CategoricalSummary: ...

class SimulationSummary:
    def to_parquet(self, path: Path) -> None: ...
    @staticmethod
    def from_parquet(path: Path) -> SimulationSummary: ...
    @property
    def n(self) -> int: ...
    @property
    def mean(self) -> npt.NDArray[np.float64]: ...
    @property
    def variance(self) -> npt.NDArray[np.float64]: ...
    @property
    def std(self) -> npt.NDArray[np.float64]: ...
    @property
    def cutoffs(self) -> list[float]: ...
    @property
    def probability_above(self) -> npt.NDArray[np.float64]: ...
    @property
    def mean_above(self) -> npt.NDArray[np.float64]: ...
    @property
    def quantiles(self) -> list[float]: ...
    @property
    def quantile_values(self) -> npt.NDArray[np.float64]: ...
    @property
    def realization_mean(self) -> npt.NDArray[np.float64]: ...
    @property
    def realization_above(self) -> npt.NDArray[np.float64]: ...
    @property
    def kept(self) -> list[int]: ...
    @property
    def realizations(self) -> npt.NDArray[np.float64] | None: ...

class CategoricalSummary:
    def to_parquet(self, path: Path) -> None: ...
    @staticmethod
    def from_parquet(path: Path) -> CategoricalSummary: ...
    @property
    def n(self) -> int: ...
    @property
    def probabilities(self) -> npt.NDArray[np.float64]: ...
    @property
    def most_likely(self) -> npt.NDArray[np.int64]: ...
    @property
    def entropy(self) -> npt.NDArray[np.float64]: ...
    @property
    def proportions(self) -> npt.NDArray[np.float64]: ...
    @property
    def kept(self) -> list[int]: ...
    @property
    def realizations(self) -> npt.NDArray[np.int64] | None: ...

def gibbs(
    coords: ArrayLike,
    bounds: ArrayLike,
    variogram: Variogram,
    *,
    iterations: int = 200,
    burn_in: int = 50,
    seed: int = 0,
) -> npt.NDArray[np.float64]: ...
def localize(
    smus: BlockModel, ranking: Column, panels: BlockModel, realizations: ArrayLike, *, name: str = "localized"
) -> BlockModel: ...
def correct_distribution(
    values: SimulationSummary | ArrayLike,
    reference: ArrayLike | KernelDensity | GaussianMixture,
    *,
    weights: ArrayLike | None = None,
    strength: float = 1.0,
    realizations: Sequence[int] | None = None,
) -> npt.NDArray[np.float64]: ...

class Cokriging:
    def to_parquet(self, path: Path) -> None: ...
    @staticmethod
    def from_parquet(path: Path) -> Cokriging: ...
    def __init__(
        self, coregionalization: Coregionalization, search: Search, *, means: Sequence[float] | None = None
    ) -> None: ...
    def fit(
        self,
        coords: ArrayLike | PointSet | BlockModel,
        values: ArrayLike | Column,
        variables: Sequence[int] | ArrayLike | Column,
        *,
        holes: Holes | Column | None = None,
    ) -> Cokriging: ...
    def predict(
        self,
        targets: Any,
        *,
        variable: int = 0,
        return_variance: bool = False,
        collocated: Mapping[int, ArrayLike] | None = None,
        progress: bool = True,
    ) -> Any: ...

class DisjunctiveKriging:
    def to_parquet(self, path: Path) -> None: ...
    @staticmethod
    def from_parquet(path: Path) -> DisjunctiveKriging: ...
    def __init__(
        self, anamorphosis: HermiteAnamorphosis, variogram: Variogram, search: Search, *, order: int = 20
    ) -> None: ...
    def fit(
        self, coords: ArrayLike | PointSet | BlockModel, values: ArrayLike | Column
    ) -> DisjunctiveKriging: ...
    def predict(self, targets: Any, *, progress: bool = True) -> npt.NDArray[np.float64]: ...
    def predict_tonnage(
        self, targets: Any, cutoff: float, *, progress: bool = True
    ) -> npt.NDArray[np.float64]: ...

class MultipleIndicatorKriging:
    def to_parquet(self, path: Path) -> None: ...
    @staticmethod
    def from_parquet(path: Path) -> MultipleIndicatorKriging: ...
    def __init__(
        self,
        variogram: Variogram | Sequence[Variogram],
        search: Search | Sequence[Search],
        thresholds: Sequence[float],
        *,
        simple: bool = False,
        tails: tuple[float, float] | None = None,
        interpolation: Literal["global", "linear"] = "global",
        upper_tail: tuple[Literal["power", "hyperbolic"], float] | None = None,
    ) -> None: ...
    @property
    def thresholds(self) -> list[float]: ...
    def fit(
        self,
        coords: ArrayLike | PointSet | BlockModel,
        values: ArrayLike | Column,
        *,
        weights: ArrayLike | Column | None = None,
        holes: Holes | Column | None = None,
    ) -> MultipleIndicatorKriging: ...
    def predict(
        self,
        targets: Any,
        *,
        cutoffs: Sequence[float] = (),
        quantiles: Sequence[float] = (),
        anisotropy: LocalAnisotropy | None = None,
        diagnostics: bool = False,
        discretization: tuple[int, int, int] | None = None,
        progress: bool = True,
    ) -> IndicatorSummary: ...
    def cross_validate(self, *, folds: int | None = None) -> IndicatorCrossValidation: ...
    def localize(
        self,
        smus: BlockModel,
        ranking: Column,
        panels: BlockModel,
        *,
        variance_factor: float | Variogram | None = None,
        name: str = "localized",
        discretization: tuple[int, int, int] | None = None,
    ) -> BlockModel: ...

class MultigaussianKriging:
    def to_parquet(self, path: Path) -> None: ...
    @staticmethod
    def from_parquet(path: Path) -> MultigaussianKriging: ...
    def __init__(
        self,
        variogram: Variogram,
        search: Search | Sequence[Search],
        *,
        tails: tuple[float, float] | None = None,
    ) -> None: ...
    def fit(
        self,
        coords: ArrayLike | PointSet | BlockModel,
        values: ArrayLike | Column,
        *,
        weights: ArrayLike | Column | None = None,
        holes: Holes | Column | None = None,
        despike: bool = False,
    ) -> MultigaussianKriging: ...
    def predict(
        self,
        targets: Any,
        *,
        cutoffs: Sequence[float] = (),
        quantiles: Sequence[float] = (),
        discretization: tuple[int, int, int] | None = None,
        diagnostics: bool = False,
    ) -> IndicatorSummary: ...
    def cross_validate(self, *, folds: int | None = None) -> IndicatorCrossValidation: ...

class IndicatorSummary:
    def to_parquet(self, path: Path) -> None: ...
    @staticmethod
    def from_parquet(path: Path) -> IndicatorSummary: ...
    @property
    def mean(self) -> npt.NDArray[np.float64]: ...
    @property
    def variance(self) -> npt.NDArray[np.float64]: ...
    @property
    def std(self) -> npt.NDArray[np.float64]: ...
    @property
    def thresholds(self) -> list[float]: ...
    @property
    def cdf(self) -> npt.NDArray[np.float64]: ...
    @property
    def correction(self) -> npt.NDArray[np.float64]: ...
    @property
    def cutoffs(self) -> list[float]: ...
    @property
    def probability_above(self) -> npt.NDArray[np.float64]: ...
    @property
    def mean_above(self) -> npt.NDArray[np.float64]: ...
    @property
    def quantiles(self) -> list[float]: ...
    @property
    def quantile_values(self) -> npt.NDArray[np.float64]: ...
    @property
    def diagnostics(self) -> Table | None: ...

class CategoricalIndicatorKriging:
    def to_parquet(self, path: Path) -> None: ...
    @staticmethod
    def from_parquet(path: Path) -> CategoricalIndicatorKriging: ...
    def __init__(
        self,
        variograms: Variogram | Sequence[Variogram],
        search: Search | Sequence[Search],
        *,
        simple: bool = False,
        scheme: Categories | None = None,
    ) -> None: ...
    @property
    def names(self) -> list[str]: ...
    @property
    def scheme(self) -> Categories | None: ...
    def fit(
        self,
        coords: ArrayLike | PointSet | BlockModel,
        categories: ArrayLike | Column,
        *,
        weights: ArrayLike | Column | None = None,
        holes: Holes | Column | None = None,
        domains: Label | Sequence[Label] | ArrayLike | None = None,
        domain_column: Column | None = None,
    ) -> CategoricalIndicatorKriging: ...
    def predict(
        self,
        targets: Any,
        *,
        domains: Label | Sequence[Label] | ArrayLike | None = None,
        domain_column: Column | None = None,
        anisotropy: LocalAnisotropy | None = None,
        diagnostics: bool = False,
        progress: bool = True,
    ) -> CategoricalIndicatorSummary: ...
    def cross_validate(self, *, folds: int | None = None) -> CategoricalCrossValidation: ...

class CategoricalIndicatorSummary:
    def to_parquet(self, path: Path) -> None: ...
    @staticmethod
    def from_parquet(path: Path) -> CategoricalIndicatorSummary: ...
    @property
    def probabilities(self) -> npt.NDArray[np.float64]: ...
    @property
    def most_likely(self) -> npt.NDArray[np.float64]: ...
    @property
    def entropy(self) -> npt.NDArray[np.float64]: ...
    @property
    def correction(self) -> npt.NDArray[np.float64]: ...
    @property
    def proportions(self) -> npt.NDArray[np.float64]: ...
    @property
    def names(self) -> list[str]: ...
    @property
    def scheme(self) -> Categories | None: ...
    @property
    def diagnostics(self) -> Table | None: ...

class Drillholes:
    def __init__(
        self,
        collar: TableLike,
        survey: TableLike,
        intervals: TableLike | None = None,
        *,
        hole: str = "HOLE_ID",
        x: str = "X",
        y: str = "Y",
        z: str = "Z",
        at: str = "DEPTH",
        azimuth: str = "AZIMUTH",
        dip: str | None = "DIP",
        inclination: str | None = None,
        from_: str = "FROM",
        to: str = "TO",
        method: Literal["minimum_curvature", "tangential", "balanced_tangential"] = "minimum_curvature",
    ) -> None: ...
    @property
    def holes(self) -> list[str]: ...
    @property
    def interval_columns(self) -> tuple[str, str, str] | None: ...
    def paths(self) -> Table: ...
    def at(self, holes: Sequence[str], depths: ArrayLike) -> npt.NDArray[np.float64]: ...
    def samples(self) -> PointSet: ...
    def composite(
        self,
        length: float | None,
        grades: Sequence[str],
        *,
        domain: str | None = None,
        intervals: TableLike | None = None,
        residual: Literal["keep", "drop", "merge"] = "keep",
        min_fraction: float = 0.5,
        categories: Sequence[str] = (),
    ) -> PointSet: ...
    def runs(
        self,
        grade: str | None,
        *,
        cutoff: float | None = None,
        category: str | None = None,
        ore: Sequence[str] = (),
        min_length: float = 0.0,
        max_dilution: float = 0.0,
        edge: float = 0.0,
    ) -> Table: ...
    def mesh_intervals(self, mesh: Mesh, *, step: float = 1.0, tolerance: float = 0.01) -> Table: ...
    def __len__(self) -> int: ...

def closure(parts: ArrayLike, *, total: float = 1.0) -> npt.NDArray[np.float64]: ...
def clr(parts: ArrayLike) -> npt.NDArray[np.float64]: ...
def clr_inverse(coords: ArrayLike) -> npt.NDArray[np.float64]: ...
def alr(parts: ArrayLike) -> npt.NDArray[np.float64]: ...
def alr_inverse(coords: ArrayLike) -> npt.NDArray[np.float64]: ...
def ilr(parts: ArrayLike) -> npt.NDArray[np.float64]: ...
def ilr_inverse(coords: ArrayLike) -> npt.NDArray[np.float64]: ...
def aitchison_distance(a: ArrayLike, b: ArrayLike) -> npt.NDArray[np.float64]: ...

class Mesh:
    def __init__(self, vertices: ArrayLike, triangles: ArrayLike, *, crs: str | None = None) -> None: ...
    @property
    def vertices(self) -> npt.NDArray[np.float64]: ...
    @property
    def triangles(self) -> npt.NDArray[np.int64]: ...
    @property
    def bounds(self) -> tuple[list[float], list[float]] | None: ...
    @property
    def crs(self) -> str | None: ...
    @property
    def is_closed(self) -> bool: ...
    @property
    def analysis(self) -> dict[str, int | bool]: ...
    @property
    def area(self) -> float: ...
    @property
    def volume(self) -> float: ...
    @property
    def vertex_attributes(self) -> Table: ...
    @property
    def face_attributes(self) -> Table: ...
    def with_vertex_column(self, name: str, values: ArrayLike | Sequence[str]) -> Mesh: ...
    def with_face_column(self, name: str, values: ArrayLike | Sequence[str]) -> Mesh: ...
    def contains(self, points: ArrayLike) -> npt.NDArray[np.bool_]: ...
    def winding_number(self, points: ArrayLike) -> npt.NDArray[np.float64]: ...
    def distance(self, points: ArrayLike, *, signed: bool = False) -> npt.NDArray[np.float64]: ...
    def vertical_distance(self, points: Any) -> npt.NDArray[np.float64]: ...
    def repair(self, *, tolerance: float = 0.0) -> Mesh: ...
    def proportion(
        self,
        targets: Any,
        *,
        size: Sequence[float] | None = None,
        discretization: int | tuple[int, int, int] = 4,
    ) -> npt.NDArray[np.float64]: ...

class PolygonSelector:
    def __init__(
        self,
        rings: Sequence[ArrayLike] | Polylines,
        *,
        closed: bool = False,
        z_min: float | None = None,
        z_max: float | None = None,
    ) -> None: ...
    def contains(self, points: ArrayLike) -> npt.NDArray[np.bool_]: ...

class Unfold:
    def __init__(
        self,
        footwall: Mesh,
        hangingwall: Mesh,
        *,
        mode: Literal["proportional", "footwall", "hangingwall"] = "proportional",
        reference: Literal["footwall", "hangingwall"] | None = None,
        extrapolate: bool = False,
    ) -> None: ...
    def fit(self, coords: ArrayLike | PointSet | BlockModel) -> Unfold: ...
    def fit_transform(self, coords: ArrayLike | PointSet | BlockModel) -> npt.NDArray[np.float64]: ...
    def transform(self, coords: ArrayLike | PointSet | BlockModel) -> npt.NDArray[np.float64]: ...
    def inverse(self, coords: ArrayLike) -> npt.NDArray[np.float64]: ...

def point_in_polygon(points: ArrayLike, polygon: ArrayLike | Polylines) -> npt.NDArray[np.bool_]: ...
def polygon_distance(
    points: ArrayLike, polygon: ArrayLike | Polylines, *, signed: bool = False
) -> npt.NDArray[np.float64]: ...
def assign_domain(
    targets: Any,
    *,
    coords: ArrayLike | PointSet | BlockModel | None = None,
    domains: Labels | None = None,
    domain_column: Column | None = None,
    method: str = "nearest",
    mesh: Mesh | None = None,
    n: int = 5,
) -> tuple[list[str], npt.NDArray[np.float64]]: ...
def block_shell(model: BlockModel, *, column: str | None = None) -> Mesh: ...
def convex_hull(points: ArrayLike) -> Mesh: ...
def grid_surface(model: BlockModel, column: str) -> Mesh: ...
def merge_intervals(
    left: TableLike, right: TableLike, *, hole: str = "HOLE_ID", from_: str = "FROM", to: str = "TO"
) -> Table: ...
def check_drillholes(
    collar: TableLike,
    survey: TableLike | None = None,
    intervals: TableLike | Mapping[str, TableLike] | None = None,
    *,
    hole: str = "HOLE_ID",
    x: str = "X",
    y: str = "Y",
    z: str = "Z",
    at: str = "DEPTH",
    azimuth: str = "AZIMUTH",
    dip: str | None = "DIP",
    inclination: str | None = None,
    from_: str = "FROM",
    to: str = "TO",
    max_depth: str | None = None,
    grades: Sequence[str] | None = None,
    nodata: Sequence[float] = (-99.0, -999.0, -9999.0, 1e21),
    max_deviation: float = 20.0,
    tolerance: float = 1e-6,
) -> tuple[dict[str, Table], Table, Table]: ...
def fix_drillholes(
    flags: Mapping[str, Table],
    tables: Mapping[str, TableLike],
    *,
    missing: Literal["drop", "keep"] = "drop",
    duplicates: Literal["drop", "keep"] = "drop",
    inverted: Literal["drop", "keep"] = "drop",
    out_of_range: Literal["drop", "keep"] = "drop",
    overlaps: Literal["keep_first", "keep"] = "keep_first",
    sentinels: Literal["null", "drop", "keep"] = "null",
    deviation: Literal["drop", "keep"] = "drop",
    no_collar: Literal["drop", "keep"] = "drop",
    past_depth: Literal["drop", "keep"] = "keep",
    dip_sign: Literal["keep", "negate"] = "keep",
    id_mismatch: Literal["rename", "keep"] = "rename",
    text_values: Literal["null", "half", "limit", "keep"] = "null",
) -> tuple[dict[str, Table], Table]: ...

class LocalAnisotropy:
    def __init__(
        self,
        coords: ArrayLike,
        angles: ArrayLike,
        ratios: ArrayLike,
        *,
        scales: ArrayLike | float | None = None,
    ) -> None: ...
    @staticmethod
    def from_grid(
        model: BlockModel, column: str, *, window: int = 2, ratios: tuple[float, float] | None = None
    ) -> LocalAnisotropy: ...
    @staticmethod
    def from_points(
        coords: ArrayLike, *, k: int = 20, ratios: tuple[float, float] | None = None
    ) -> LocalAnisotropy: ...
    @staticmethod
    def from_mesh(
        mesh: Mesh, targets: Any, *, major: str = "dip", ratios: tuple[float, float] = (1.0, 0.2)
    ) -> LocalAnisotropy: ...
    def smooth(self, radius: float) -> LocalAnisotropy: ...
    def at(self, targets: Any) -> LocalAnisotropy: ...
    def to_parquet(self, path: Path) -> None: ...
    @staticmethod
    def from_parquet(path: Path) -> LocalAnisotropy: ...
    @property
    def coords(self) -> npt.NDArray[np.float64]: ...
    @property
    def angles(self) -> npt.NDArray[np.float64]: ...
    @property
    def ratios(self) -> npt.NDArray[np.float64]: ...
    @property
    def scales(self) -> npt.NDArray[np.float64]: ...
    def __len__(self) -> int: ...

def local_variogram_parameters(
    coords: ArrayLike | PointSet | BlockModel,
    values: ArrayLike | Column,
    grid: BlockModel | PointSet | ArrayLike,
    *,
    variogram: Variogram,
    window: float,
    lag: float,
    max_lag: float | None = None,
    anisotropy: LocalAnisotropy | None = None,
    sectors: int = 8,
    min_pairs: int = 100,
) -> LocalAnisotropy: ...

class ImplicitModel:
    def __init__(
        self,
        engine: str = "rbf",
        *,
        kernel: str = "biharmonic",
        variogram: Variogram | None = None,
        degree: int = 1,
        smoothing: float = 0.0,
        rotation: tuple[float, float, float] | None = None,
        ratios: tuple[float, float] | None = None,
    ) -> None: ...
    def to_parquet(self, path: Path) -> None: ...
    @staticmethod
    def from_parquet(path: Path) -> ImplicitModel: ...
    def fit(
        self,
        coords: ArrayLike | None = None,
        values: ArrayLike | None = None,
        *,
        cutoff: float | None = None,
        boundaries: ArrayLike | None = None,
        planes: ArrayLike | None = None,
        lineations: ArrayLike | None = None,
    ) -> ImplicitModel: ...
    def predict(
        self, targets: Any, *, gradient: bool = False, variance: bool = False, progress: bool = True
    ) -> Any: ...
    def isosurface(
        self, model: BlockModel, *, isovalue: float = 0.0, closed: bool = False, progress: bool = True
    ) -> Mesh: ...
    @property
    def report(self) -> dict[str, Any] | None: ...

def write_parquet(
    path: Path, data: PointSet | BlockModel | Polylines | TableLike, *, progress: bool = True
) -> None: ...
def read_parquet(path: Path, *, progress: bool = True) -> PointSet | BlockModel | Polylines | Table: ...

class BlockModelFile:
    def __init__(self, path: Path) -> None: ...
    @property
    def origin(self) -> list[float]: ...
    @property
    def size(self) -> list[float]: ...
    @property
    def count(self) -> list[int]: ...
    @property
    def rotation(self) -> list[float]: ...
    @property
    def crs(self) -> str | None: ...
    @property
    def column_names(self) -> list[str]: ...
    def chunks(
        self, *, rows: int = 1_000_000, columns: Sequence[str] | None = None
    ) -> Iterator[BlockModel]: ...
    def __len__(self) -> int: ...

def map_blocks(
    path: Path,
    out: Path,
    func: Callable[[BlockModel], Mapping[str, ArrayLike]],
    *,
    rows: int = 1_000_000,
    keep: bool = True,
) -> None: ...
def read_mesh(path: Path, *, progress: bool = True) -> Mesh: ...
def write_mesh(path: Path, mesh: Mesh, *, ascii: bool = False, progress: bool = True) -> None: ...
def read_geotiff(path: Path, *, nodata: float | None = None) -> BlockModel: ...
def write_geotiff(path: Path, model: BlockModel, *, nodata: float = -9999.0) -> None: ...
def read_segy(
    path: Path,
    *,
    column: str = "amplitude",
    inline_byte: int = 189,
    crossline_byte: int = 193,
    x_byte: int = 181,
    y_byte: int = 185,
    nodata: float | None = None,
) -> BlockModel: ...
def write_segy(path: Path, model: BlockModel, column: str, *, nodata: float = 0.0) -> None: ...
def read_shapefile(path: Path, *, nodata: Sequence[float | str] | None = None) -> PointSet | Polylines: ...
def write_shapefile(path: Path, data: PointSet | Polylines) -> None: ...

class Categories:
    def __init__(
        self,
        names: Sequence[Label],
        *,
        colors: Sequence[str] | None = None,
        mapping: Mapping[Label, Label] | None = None,
        other: Label | None = None,
    ) -> None: ...
    @staticmethod
    def from_values(
        values: ArrayLike,
        *,
        weights: ArrayLike | None = None,
        min_share: float = 0.0,
        mapping: Mapping[Label, Label] | None = None,
        other: str = "other",
        colors: Sequence[str] | None = None,
    ) -> Categories: ...
    def encode(self, values: ArrayLike) -> npt.NDArray[np.float64]: ...
    def decode(self, codes: ArrayLike) -> list[str | None]: ...
    def lump(self, names: Sequence[str], *, into: str = "other") -> Categories: ...
    def shares(self, codes: ArrayLike, *, weights: ArrayLike | None = None) -> npt.NDArray[np.float64]: ...
    @property
    def names(self) -> list[str]: ...
    @property
    def colors(self) -> list[str] | None: ...
    @property
    def other(self) -> str | None: ...
    @property
    def mapping(self) -> dict[str, str]: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(text: str) -> Categories: ...
    def __len__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

def vertical_proportions(
    coords: ArrayLike | PointSet | BlockModel,
    categories: Labels | Column,
    *,
    size: float,
    elevation: ArrayLike | Column | None = None,
    weights: ArrayLike | Column | None = None,
    scheme: Categories | None = None,
) -> Table: ...
def combine_proportions(
    coords: ArrayLike | PointSet | BlockModel,
    vertical: Table,
    areal: Table | ArrayLike,
    *,
    elevation: ArrayLike | Column | None = None,
) -> Table: ...
def describe(
    values: ArrayLike | Column,
    *,
    weights: ArrayLike | Column | None = None,
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    data: Data | None = None,
) -> dict[str, float]: ...
def describe_by(
    values: ArrayLike | Column,
    categories: Labels | Column,
    *,
    weights: ArrayLike | Column | None = None,
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    data: Data | None = None,
) -> Table: ...
def grade_tonnage(
    values: ArrayLike | Column,
    cutoffs: ArrayLike,
    *,
    weights: ArrayLike | Column | None = None,
    density: float | ArrayLike | Column = 1.0,
    categories: Labels | Column | None = None,
    data: Data | None = None,
) -> Table: ...
def domain_change(
    before: Labels | Column,
    after: Labels | Column,
    *,
    weights: ArrayLike | Column | None = None,
    density: float | ArrayLike | Column | None = None,
    grades: ArrayLike | Column | None = None,
    scheme: Categories | None = None,
    data: Data | None = None,
) -> Table: ...
def transition_matrix(
    depth: ArrayLike | Column,
    categories: Labels | Column,
    holes: Labels | Column,
    *,
    lag: float,
    tolerance: float = 0.0,
    scheme: Categories | None = None,
    data: Data | None = None,
) -> Table: ...
def compare_models(
    model: BlockModel,
    columns: Sequence[Column] | Mapping[str, ArrayLike | Column],
    cutoffs: ArrayLike,
    *,
    categories: Labels | Column | None = None,
    reference: str | None = None,
    density: float | ArrayLike | Column = 1.0,
) -> Table: ...
def capping_report(
    values: ArrayLike | Column,
    caps: Mapping[int | str, float],
    *,
    domains: Labels | None = None,
    domain_column: Column | None = None,
    weights: ArrayLike | Column | None = None,
    data: Data | None = None,
) -> Table: ...
def swath(
    coords: PointSet | BlockModel | ArrayLike,
    values: ArrayLike | Column,
    width: float,
    *,
    azimuth: float | None = None,
    axis: str | None = None,
    weights: ArrayLike | Column | None = None,
    density: float | ArrayLike | Column = 1.0,
) -> Table: ...
def contact(
    coords: PointSet | ArrayLike,
    values: ArrayLike | Column,
    *,
    domains: Labels | None = None,
    domain_column: Column | None = None,
    holes: Holes | Column,
    inside: int | str,
    outside: int | str,
    max_distance: float,
    bin: float,
) -> Table: ...
def soft_boundary(
    coords: PointSet | ArrayLike,
    values: ArrayLike | Column,
    *,
    domains: Labels | None = None,
    domain_column: Column | None = None,
    target: Any,
    buffer: float,
    weights: ArrayLike | Column | None = None,
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> tuple[npt.NDArray[np.bool_], Table]: ...
def capping(
    values: ArrayLike | Column,
    *,
    weights: ArrayLike | Column | None = None,
    caps: ArrayLike | None = None,
    data: Data | None = None,
) -> Table: ...
def h_scatter(
    coords: PointSet | ArrayLike,
    values: ArrayLike | Column,
    lag: float,
    tolerance: float,
    *,
    azimuth: float | None = None,
    dip: float = 0.0,
    angle_tolerance: float = 22.5,
    other: ArrayLike | Column | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]: ...
def correlation(
    data: ArrayLike | Data,
    *,
    columns: Sequence[Column] | None = None,
    weights: ArrayLike | Column | None = None,
    method: Literal["pearson", "spearman", "covariance"] = "pearson",
) -> npt.NDArray[np.float64]: ...
@overload
def duplicates(
    coords: PointSet | ArrayLike, *, tolerance: float = 0.0, merge: None = None, weights: None = None
) -> tuple[Table, npt.NDArray[np.int64]]: ...
@overload
def duplicates(
    coords: PointSet,
    *,
    tolerance: float = 0.0,
    merge: Literal["mean", "first", "max"],
    weights: ArrayLike | Column | None = None,
) -> PointSet: ...
def pairs(
    a: PointSet | ArrayLike,
    b: PointSet | ArrayLike,
    max_distance: float,
    *,
    values: Column | tuple[ArrayLike | Column, ArrayLike | Column] | None = None,
    unique: bool = True,
    holes: Column | tuple[Holes | Column, Holes | Column] | None = None,
) -> Table: ...
def paired_bias(pairs: Table, bins: int | ArrayLike) -> Table: ...
def data_spacing(
    coords: PointSet | ArrayLike,
    *,
    n: int = 1,
    targets: Any | None = None,
    horizontal: bool = False,
) -> npt.NDArray[np.float64]: ...
def validate_model(
    model: BlockModel,
    grade: Column | ArrayLike,
    data: Data,
    values: Column | ArrayLike,
    *,
    weights: ArrayLike | Column | None = None,
    domain_column: Column | tuple[Column, Column] | None = None,
    density: float | ArrayLike | Column = 1.0,
    reference: Column | ArrayLike | None = None,
) -> Table: ...
def smooth_classes(
    model: BlockModel,
    classes: ArrayLike | Column,
    *,
    window: tuple[int, int, int] = (3, 3, 1),
    iterations: int = 1,
    domains: Labels | None = None,
    domain_column: Column | None = None,
) -> npt.NDArray[Any]: ...
def remove_small_units(
    model: BlockModel,
    classes: ArrayLike | Column,
    *,
    min_volume: float | None = None,
    min_blocks: int | None = None,
    connectivity: int = 6,
    domains: Labels | None = None,
    domain_column: Column | None = None,
) -> npt.NDArray[Any]: ...
def contact_distance(
    model: BlockModel,
    classes: ArrayLike | Column,
    *,
    target: Any | None = None,
    signed: bool = True,
) -> npt.NDArray[np.float64]: ...
def buffer_domains(
    model: BlockModel,
    classes: ArrayLike | Column,
    *,
    distance: float,
    label: Any = "contact",
    target: Any | None = None,
) -> npt.NDArray[Any]: ...
