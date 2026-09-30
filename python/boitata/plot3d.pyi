from typing import Any, TypeAlias

from boitata._boitata import BlockModel, Drillholes, Mesh, PointSet

Container: TypeAlias = PointSet | Drillholes | BlockModel | Mesh

def to_pyvista(data: Container) -> Any: ...
def plot(data: Container | Any, values: str | None = None, *, plotter: Any = None, **kwargs: Any) -> Any: ...
def slices(
    model: BlockModel,
    values: str | None = None,
    *,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    plotter: Any = None,
    **kwargs: Any,
) -> Any: ...
