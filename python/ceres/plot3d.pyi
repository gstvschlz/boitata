from typing import Any, TypeAlias

from ceres._ceres import BlockModel, Drillholes, Mesh, PointSet

Container: TypeAlias = PointSet | Drillholes | BlockModel | Mesh

def to_pyvista(data: Container) -> Any: ...
def plot(data: Container | Any, scalars: str | None = None, plotter: Any = None, **kwargs: Any) -> Any: ...
def slices(
    model: BlockModel,
    scalars: str | None = None,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    plotter: Any = None,
    **kwargs: Any,
) -> Any: ...
