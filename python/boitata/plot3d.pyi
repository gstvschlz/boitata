from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self, TypeAlias

from boitata._boitata import BlockModel, Drillholes, Mesh, PointSet

__all__ = ["Scene", "plot"]

Container: TypeAlias = PointSet | Drillholes | BlockModel | Mesh
Theme: TypeAlias = Literal["auto", "light", "dark"] | Mapping[str, str]
MotionQuality: TypeAlias = Literal["auto", "full"] | float
Condition: TypeAlias = tuple[float | None, float | None] | Sequence[str]
Representation: TypeAlias = Literal["lines", "tubes", "points", "spheres", "surface", "wireframe", "cells"]

class Scene:
    motion_quality: MotionQuality
    filters: dict[str, dict[str, Condition]]
    sections: dict[str, Any] | None
    @property
    def picked(self) -> dict[str, Any] | None: ...
    def __init__(
        self, *, theme: Theme = "auto", height: int = 600, motion_quality: MotionQuality = "auto"
    ) -> None: ...
    def add(
        self,
        data: Container,
        values: str | None = None,
        *,
        name: str | None = None,
        representation: Representation | None = None,
        color: str | None = None,
        opacity: float = 1.0,
        cmap: str | None = None,
        clim: tuple[float, float] | None = None,
        point_size: float | None = None,
        line_width: float | None = None,
        radius: float | None = None,
        label: str | None = None,
        visible: bool = True,
        columns: Sequence[str] | None = None,
        filter: Mapping[str, Condition] | None = None,
    ) -> Self: ...
    def view(self, *, azimuth: float = 45.0, dip: float = 30.0) -> Self: ...
    def section(self, points: Any | None, *, width: float | None = None, dip: float = 90.0) -> Self: ...
    def save(self, path: str | Path) -> Path: ...
    def show(self) -> Path | None: ...
    def screenshot(
        self, path: str | Path, *, scale: float = 1, panel: bool = False, transparent: bool = False
    ) -> Path: ...

def plot(
    data: Container, values: str | None = None, *, theme: Theme = "auto", height: int = 600, **kwargs: Any
) -> Scene: ...
