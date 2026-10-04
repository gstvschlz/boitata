from pathlib import Path
from typing import Any, Literal, Self, TypeAlias

from boitata._boitata import BlockModel, Drillholes, Mesh, PointSet

Container: TypeAlias = PointSet | Drillholes | BlockModel | Mesh
Style: TypeAlias = Literal["surface", "wireframe", "points", "points_gaussian", "volume"]

class Scene:
    plotter: Any
    colors: dict[str, Any]
    def __init__(self, *, plotter: Any = None, **kwargs: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def add(
        self,
        data: Container | Any,
        values: str | None = None,
        *,
        style: Style | None = None,
        radius: float | None = None,
        labels: bool = False,
        **kwargs: Any,
    ) -> Self: ...
    def show(self, *, browser: bool = False, **kwargs: Any) -> Path | Any: ...

def to_pyvista(data: Container) -> Any: ...
def plot(
    data: Container | Any, values: str | None = None, *, plotter: Scene | Any = None, **kwargs: Any
) -> Scene: ...
def slices(
    model: BlockModel,
    values: str | None = None,
    *,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    plotter: Scene | Any = None,
    **kwargs: Any,
) -> Scene: ...
