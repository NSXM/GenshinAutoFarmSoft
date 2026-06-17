"""
Формат маршрута и его чтение/запись.

Маршрут — список точек (waypoints) в метрах в системе fused_pos (отсчёт от точки
старта записи). Вместе с точками сохраняем minimap_meters_per_px, в котором
маршрут писан, — чтобы потом было видно масштаб и при необходимости пересчитать.

JSON-схема:
    {
      "name": "recorded",
      "minimap_meters_per_px": 5.869,
      "waypoints": [{"x": 0.0, "y": 0.0}, {"x": 12.3, "y": -4.1}, ...]
    }
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Waypoint:
    x: float
    y: float


@dataclass
class Route:
    name: str = "recorded"
    minimap_meters_per_px: float = 0.0
    waypoints: List[Waypoint] = field(default_factory=list)


def load_route(path: str) -> Route:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    wps = [Waypoint(float(p["x"]), float(p["y"])) for p in data.get("waypoints", [])]
    return Route(
        name=data.get("name", "recorded"),
        minimap_meters_per_px=float(data.get("minimap_meters_per_px", 0.0)),
        waypoints=wps,
    )


def save_route(route: Route, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = {
        "name": route.name,
        "minimap_meters_per_px": round(route.minimap_meters_per_px, 4),
        "waypoints": [{"x": round(w.x, 2), "y": round(w.y, 2)} for w in route.waypoints],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
