"""
Шаг 4-5 ТЗ — триангуляция объектов через известное движение камеры/персонажа.

ВАЖНАЯ ГЕОМЕТРИЯ. Чистый поворот камеры вокруг её собственной оптической оси
параллакса НЕ даёт: близкие и далёкие объекты смещаются на экране на один и тот
же угловой Δyaw. Но камера в Genshin — от 3-го лица: она вращается вокруг
ПЕРСОНАЖА на радиусе R (orbit_radius_m). Поэтому покачивание камеры по yaw
двигает оптический центр по дуге окружности радиуса R → возникает база
(baseline) → возникает параллакс → можно триангулировать глубину.

База при повороте на Δyaw:   b = 2 * R * sin(Δyaw / 2)

Три источника известного смещения камеры (всё — без доступа к памяти игры):
  1. Покачивание камеры на известный угол (угол виден на миникарте/из мыши).
  2. Известное число шагов персонажа (скорость бега в м/с * время).
  3. Прыжок (известная дальность прыжка).

Во всех случаях база известна боту заранее → классическая two-view
триангуляция: глубина из диспаритета пеленгов и базы.

Ожидаемое vs фактическое смещение:
  Зная позу до/после и гипотезу позиции объекта, считаем ОЖИДАЕМЫЙ пиксельный
  сдвиг. SAM/optical-flow дают ФАКТИЧЕСКИЙ. Разница:
    * ~0 у статичных объектов на плоскости движения,
    * систематическая → объект выше/ниже или ближе/дальше, чем гипотеза,
    * хаотичная/большая → объект сам движется (монстр).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..utils.geom import Intrinsics, wrap_deg


@dataclass
class CameraPose:
    """Поза камеры в мировой плоскости (вид сверху). yaw: 0=север, по часовой."""
    x: float            # метры
    y: float            # метры
    yaw_deg: float

    def forward(self) -> np.ndarray:
        r = math.radians(self.yaw_deg)
        return np.array([math.sin(r), math.cos(r)])   # (восток, север)

    def right(self) -> np.ndarray:
        r = math.radians(self.yaw_deg)
        return np.array([math.cos(r), -math.sin(r)])


def orbit_baseline(R: float, dyaw_deg: float) -> float:
    """Длина базы при орбитальном повороте камеры 3-го лица на dyaw вокруг игрока."""
    return 2.0 * R * math.sin(math.radians(abs(dyaw_deg)) / 2.0)


def orbit_camera_position(player_xy: Tuple[float, float], R: float,
                          yaw_deg: float) -> np.ndarray:
    """
    Позиция оптического центра камеры: на расстоянии R позади игрока вдоль
    курса (камера 3-го лица смотрит вперёд через плечо персонажа).
    """
    r = math.radians(yaw_deg)
    fwd = np.array([math.sin(r), math.cos(r)])
    return np.array(player_xy) - R * fwd


def depth_from_disparity(bearing1_deg: float, bearing2_deg: float,
                         baseline_m: float) -> Optional[float]:
    """
    Глубина (вдоль оптической оси первой камеры) из двух пеленгов и базы.
    Модель: база перпендикулярна оптической оси (как в стереопаре).
      disparity = bearing1 - bearing2 (град)
      depth = baseline / ( tan(b1) - tan(b2) )   при малых углах ~ baseline / disp
    """
    d = wrap_deg(bearing1_deg - bearing2_deg)
    if abs(d) < 1e-4:
        return None                      # нет параллакса -> бесконечность
    t1 = math.tan(math.radians(bearing1_deg))
    t2 = math.tan(math.radians(bearing2_deg))
    denom = t1 - t2
    if abs(denom) < 1e-6:
        return None
    return abs(baseline_m / denom)


def triangulate_two_view(intr: Intrinsics,
                         u1: float, pose1: CameraPose,
                         u2: float, pose2: CameraPose) -> Optional[Tuple[float, float]]:
    """
    Общая 2-видовая триангуляция позиции объекта на плоскости (вид сверху).
    Берём горизонтальные пеленги пикселей u1,u2 и две позы камеры → пересечение
    двух лучей в мировых координатах. Работает для покачивания камеры, шага и
    прыжка одинаково — нужны лишь две позы и два пеленга.
    """
    # абсолютные направления лучей в мире = yaw камеры + пеленг пикселя
    b1 = intr.bearing_deg(u1)
    b2 = intr.bearing_deg(u2)
    a1 = math.radians(pose1.yaw_deg + b1)
    a2 = math.radians(pose2.yaw_deg + b2)
    # луч из позы: p + t * dir, dir = (sin a, cos a)
    d1 = np.array([math.sin(a1), math.cos(a1)])
    d2 = np.array([math.sin(a2), math.cos(a2)])
    p1 = np.array([pose1.x, pose1.y])
    p2 = np.array([pose2.x, pose2.y])
    # решаем p1 + t*d1 = p2 + s*d2
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    det = np.linalg.det(A)
    if abs(det) < 1e-9:
        return None                      # лучи параллельны
    ts = np.linalg.solve(A, p2 - p1)
    pt = p1 + ts[0] * d1
    return (float(pt[0]), float(pt[1]))


def expected_pixel_u(intr: Intrinsics, obj_xy: Tuple[float, float],
                     pose: CameraPose) -> Optional[float]:
    """
    Ожидаемая горизонтальная пиксельная координата объекта при данной позе.
    Нужна, чтобы сравнить ОЖИДАЕМОЕ смещение с ФАКТИЧЕСКИМ (от SAM/flow).
    """
    rel = np.array(obj_xy) - np.array([pose.x, pose.y])
    # перевести в систему камеры: вперёд (z) и вправо (x)
    r = math.radians(pose.yaw_deg)
    fwd = np.array([math.sin(r), math.cos(r)])
    right = np.array([math.cos(r), -math.sin(r)])
    z = float(rel @ fwd)
    xr = float(rel @ right)
    if z <= 1e-3:
        return None                      # объект позади камеры
    return intr.cx + intr.fx * (xr / z)


def expected_shift_from_motion(intr: Intrinsics, obj_xy: Tuple[float, float],
                               pose1: CameraPose, pose2: CameraPose) -> Optional[float]:
    """Ожидаемый сдвиг объекта в пикселях между двумя позами (u2 - u1)."""
    u1 = expected_pixel_u(intr, obj_xy, pose1)
    u2 = expected_pixel_u(intr, obj_xy, pose2)
    if u1 is None or u2 is None:
        return None
    return u2 - u1


@dataclass
class TriangulationResult:
    world_xy: Optional[Tuple[float, float]]    # позиция объекта, метры
    depth_m: Optional[float]                   # дальность вдоль оси камеры
    residual_px: float                         # |факт - ожид| смещения
    is_moving: bool                            # похоже на самодвижущийся (монстр)


class Triangulator:
    def __init__(self, cfg_camera):
        self.cfg = cfg_camera
        self.intr = Intrinsics.from_fov(cfg_camera.image_width,
                                        cfg_camera.image_height,
                                        cfg_camera.fov_horizontal_deg)

    def from_camera_wobble(self, player_xy: Tuple[float, float],
                           yaw1: float, u1: float,
                           yaw2: float, u2: float,
                           expected_static_shift_px: Optional[float] = None
                           ) -> TriangulationResult:
        """
        Триангуляция по покачиванию камеры. player_xy фиксирована (персонаж стоит),
        двигается только камера по орбите радиуса R.
        """
        R = self.cfg.orbit_radius_m
        c1 = orbit_camera_position(player_xy, R, yaw1)
        c2 = orbit_camera_position(player_xy, R, yaw2)
        pose1 = CameraPose(c1[0], c1[1], yaw1)
        pose2 = CameraPose(c2[0], c2[1], yaw2)
        world = triangulate_two_view(self.intr, u1, pose1, u2, pose2)
        depth = None
        if world is not None:
            rel = np.array(world) - c1
            depth = float(np.linalg.norm(rel))
        # факт vs ожид: ожидаемый сдвиг для статичного объекта на найденной глубине
        residual = 0.0
        moving = False
        if expected_static_shift_px is not None:
            actual = u2 - u1
            residual = abs(actual - expected_static_shift_px)
            moving = residual > 8.0
        return TriangulationResult(world, depth, residual, moving)

    def from_known_step(self, pose1: CameraPose, u1: float,
                        pose2: CameraPose, u2: float) -> TriangulationResult:
        """
        Триангуляция по известному смещению персонажа (шаг/бег/прыжок). Позы
        камеры известны заранее: при беге R одинаков, центр сместился на пройденный
        вектор; при прыжке — на дальность прыжка.
        """
        world = triangulate_two_view(self.intr, u1, pose1, u2, pose2)
        depth = None
        moving = False
        residual = 0.0
        if world is not None:
            rel = np.array(world) - np.array([pose1.x, pose1.y])
            depth = float(np.linalg.norm(rel))
            exp = expected_shift_from_motion(self.intr, world, pose1, pose2)
            if exp is not None:
                residual = abs((u2 - u1) - exp)
                moving = residual > 8.0
        return TriangulationResult(world, depth, residual, moving)
