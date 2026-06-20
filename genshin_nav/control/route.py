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
from typing import List, Optional


@dataclass
class Waypoint:
    x: float
    y: float


@dataclass
class Route:
    name: str = "recorded"
    minimap_meters_per_px: float = 0.0
    waypoints: List[Waypoint] = field(default_factory=list)
    # Курс по стрелке (0=север, по часовой) в момент НАЧАЛА движения по маршруту
    # при записи. Раннер берёт смещение курса ИЗ ЭТОГО значения, а не из «как ты
    # встал при запуске» — поэтому стартовая поза больше не важна, бот сам
    # довернётся вдоль маршрута. None у старых маршрутов (фолбэк на старую
    # калибровку «стой лицом вдоль маршрута»). См. route_runner.run().
    start_heading: Optional[float] = None


def load_route(path: str) -> Route:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    wps = [Waypoint(float(p["x"]), float(p["y"])) for p in data.get("waypoints", [])]
    sh = data.get("start_heading", None)
    return Route(
        name=data.get("name", "recorded"),
        minimap_meters_per_px=float(data.get("minimap_meters_per_px", 0.0)),
        waypoints=wps,
        start_heading=(float(sh) if sh is not None else None),
    )


def save_route(route: Route, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = {
        "name": route.name,
        "minimap_meters_per_px": round(route.minimap_meters_per_px, 4),
        "start_heading": (round(route.start_heading, 1)
                          if route.start_heading is not None else None),
        "waypoints": [{"x": round(w.x, 2), "y": round(w.y, 2)} for w in route.waypoints],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
