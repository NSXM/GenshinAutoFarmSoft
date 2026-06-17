"""Геометрические утилиты: углы, повороты, работа с пиксельными лучами."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def wrap_deg(a: float) -> float:
    """Привести угол к диапазону (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return a + 360.0 if a <= -180.0 else a


def angle_diff_deg(a: float, b: float) -> float:
    """Кратчайшая разница углов a-b в градусах (-180, 180]."""
    return wrap_deg(a - b)


def rot2d(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


@dataclass
class Intrinsics:
    """Пинхол-модель камеры. Угол по горизонтали -> фокус в пикселях."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_fov(width: int, height: int, fov_h_deg: float) -> "Intrinsics":
        fx = (width / 2.0) / math.tan(math.radians(fov_h_deg) / 2.0)
        fy = fx  # квадратные пиксели
        return Intrinsics(width, height, fx, fy, width / 2.0, height / 2.0)

    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """Пиксель -> единичный луч в системе камеры (x вправо, y вниз, z вперёд)."""
        d = np.array([(u - self.cx) / self.fx,
                      (v - self.cy) / self.fy,
                      1.0], dtype=np.float64)
        return d / np.linalg.norm(d)

    def bearing_deg(self, u: float) -> float:
        """Горизонтальный пеленг пикселя относительно оптической оси (град)."""
        return math.degrees(math.atan2((u - self.cx), self.fx))


def meters_to_minimap_px(dx_m: float, dy_m: float, meters_per_px: float) -> Tuple[float, float]:
    return dx_m / meters_per_px, dy_m / meters_per_px
