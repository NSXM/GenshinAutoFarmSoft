"""
Формат маршрута и его чтение/запись.

Маршрут — список точек (waypoints) в метрах в системе fused_pos (отсчёт от точки
старта записи). Вместе с точками сохраняем minimap_meters_per_px, в котором
маршрут писан, — чтобы потом было видно масштаб и при необходимости пересчитать.

JSON-схема:
    {
      "name": "recorded",
      "minimap_meters_per_px": 5.869,
      "waypoints": [
        {"x": 0.0, "y": 0.0, "heading": 170.8},
        {"x": 12.3, "y": -4.1, "heading": 205.4, "turn": 34.6},
        ...
      ]
    }
heading — курс камеры (0=север) в точке; turn — поворот камеры относительно
предыдущей точки в градусах (со знаком). Оба поля опциональны.
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
    # Действие при достижении точки. None — обычная точка. "teleport" — активировать
    # телепорт (route_runner._do_teleport): стоп → F → закрыть всплывашку → дальше.
    action: Optional[str] = None
    # Курс камеры (0=север, по часовой) в момент записи точки.
    heading: Optional[float] = None
    # На сколько градусов повёрнута камера относительно ПРЕДЫДУЩЕЙ точки (со знаком:
    # + по часовой, − против). None у первой точки и у старых маршрутов.
    turn: Optional[float] = None
    # Длительность действия в секундах. Для action="climb" — сколько держать W
    # вверх (карабканье по стене/скале); высоту миникарта не видит, поэтому подъём
    # задаётся временем, а не точками. None у обычных точек.
    dur: Optional[float] = None


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
    # Путь файла маршрута (runtime, не сериализуется) — чтобы найти рядом отпечатки
    # миникарты <маршрут>.fp.npz для абсолютной локализации (localizer.py).
    source_path: Optional[str] = None


def load_route(path: str) -> Route:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    wps = [Waypoint(float(p["x"]), float(p["y"]), p.get("action"),
                    heading=(float(p["heading"]) if p.get("heading") is not None else None),
                    turn=(float(p["turn"]) if p.get("turn") is not None else None),
                    dur=(float(p["dur"]) if p.get("dur") is not None else None))
           for p in data.get("waypoints", [])]
    sh = data.get("start_heading", None)
    return Route(
        name=data.get("name", "recorded"),
        minimap_meters_per_px=float(data.get("minimap_meters_per_px", 0.0)),
        waypoints=wps,
        start_heading=(float(sh) if sh is not None else None),
        source_path=path,
    )


def save_route(route: Route, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _wp_dict(w: Waypoint) -> dict:
        d = {"x": round(w.x, 2), "y": round(w.y, 2)}
        if w.action:
            d["action"] = w.action
        if w.heading is not None:
            d["heading"] = round(w.heading, 1)
        if w.turn is not None:
            d["turn"] = round(w.turn, 1)
        if w.dur is not None:
            d["dur"] = round(w.dur, 1)
        return d

    out = {
        "name": route.name,
        "minimap_meters_per_px": round(route.minimap_meters_per_px, 4),
        "start_heading": (round(route.start_heading, 1)
                          if route.start_heading is not None else None),
        "waypoints": [_wp_dict(w) for w in route.waypoints],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
