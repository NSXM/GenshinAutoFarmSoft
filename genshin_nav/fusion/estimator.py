"""
Пункт 3 ответа — объединение данных миникарты и триангуляции SAM в единую оценку.

ИДЕЯ ФЬЮЖНА (комплементарная фильтрация + Kalman):

  Позиция персонажа:
    * Миникарта  — АБСОЛЮТНЫЙ, но грубый и низкочастотный источник (большая
      дисперсия minimap_pos_var, ~5-10 Гц). Убирает дрейф, но «дёргается».
    * Optical flow / триангуляция — ОТНОСИТЕЛЬНЫЙ, но точный и высокочастотный
      источник скорости (малая дисперсия flow_vel_var, ~30-60 Гц). Гладкий, но
      копит дрейф.
    Линейный Kalman (state = [x, y, vx, vy]) сшивает их: predict по скорости от
    flow, correct по абсолютной позиции от миникарты. Получаем гладкую и
    несмещённую траекторию персонажа.

  Позиция объекта:
    * Триангуляция даёт объект в системе КАМЕРЫ (относительно персонажа).
    * Берём отфильтрованную мировую позу персонажа (выше) и переносим объект в
      МИРОВЫЕ координаты: world_obj = player_world + R(yaw) * obj_rel.
    * Один и тот же объект, увиденный с разных поз, усредняем с весом по
      уверенности (обратно дисперсии) → устойчивая мировая карта препятствий.

  Так миникарта задаёт глобальный якорь координат, а триангуляция SAM уточняет
  и персонажа, и объекты внутри этого якоря.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
#  Kalman для позы персонажа: state = [x, y, vx, vy]                           #
# --------------------------------------------------------------------------- #
class PoseKalman:
    def __init__(self, fusion_cfg):
        self.x = np.zeros(4)                       # [x, y, vx, vy]
        self.P = np.eye(4) * 100.0
        self.q = fusion_cfg.process_var
        self.r_pos = fusion_cfg.minimap_pos_var
        self.r_vel = fusion_cfg.flow_vel_var
        self._init = False

    def predict(self, dt: float):
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        self.x = F @ self.x
        Q = np.eye(4) * self.q * dt
        self.P = F @ self.P @ F.T + Q

    def update_position(self, xy: Tuple[float, float]):
        """Коррекция абсолютной позицией с миникарты."""
        if not self._init:
            self.x[0], self.x[1] = xy
            self._init = True
            return
        z = np.array(xy)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        R = np.eye(2) * self.r_pos
        self._kalman_step(z, H, R)

    def update_velocity(self, vxy: Tuple[float, float]):
        """Коррекция скоростью от оптического потока/триангуляции."""
        z = np.array(vxy)
        H = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], float)
        R = np.eye(2) * self.r_vel
        self._kalman_step(z, H, R)

    def _kalman_step(self, z, H, R):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    @property
    def xy(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def vxy(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])


# --------------------------------------------------------------------------- #
#  Мировая карта объектов                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class WorldObject:
    xy: Tuple[float, float]
    category: str = "other"          # monster | obstacle | other
    var: float = 4.0                 # дисперсия оценки позиции (м^2)
    hits: int = 1
    moving: bool = False


@dataclass
class FusedState:
    player_xy: Tuple[float, float]
    player_vxy: Tuple[float, float]
    heading_deg: float
    objects: List[WorldObject] = field(default_factory=list)


class FusionEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.kf = PoseKalman(cfg.fusion)
        self.heading_deg: float = 0.0
        self._objects: Dict[int, WorldObject] = {}
        self._next_id = 1
        self._merge_dist_m = 1.5         # объекты ближе этого считаем одним

    # ---- персонаж ----------------------------------------------------------
    def step_player(self, dt: float,
                    minimap_xy: Optional[Tuple[float, float]],
                    flow_vxy: Optional[Tuple[float, float]],
                    heading: Optional[float]):
        self.kf.predict(dt)
        if flow_vxy is not None:
            self.kf.update_velocity(flow_vxy)
        if minimap_xy is not None:
            self.kf.update_position(minimap_xy)
        if heading is not None:
            self.heading_deg = heading

    # ---- объект: из камеры в мир -------------------------------------------
    def add_object_relative(self, obj_rel_xy: Tuple[float, float],
                            category: str, var: float, moving: bool):
        """
        obj_rel_xy — позиция объекта относительно персонажа в системе камеры
        (вперёд=+z->север-эквивалент после поворота, вправо=+x). Переносим в мир.
        """
        yaw = math.radians(self.heading_deg)
        fwd = np.array([math.sin(yaw), math.cos(yaw)])
        right = np.array([math.cos(yaw), -math.sin(yaw)])
        px, py = self.kf.xy
        world = np.array([px, py]) + obj_rel_xy[1] * fwd + obj_rel_xy[0] * right
        self._integrate(tuple(world), category, var, moving)

    def add_object_world(self, world_xy: Tuple[float, float],
                         category: str, var: float, moving: bool):
        self._integrate(world_xy, category, var, moving)

    def _integrate(self, world_xy, category, var, moving):
        # найти близкий существующий объект и слить (взвешенно по дисперсии)
        best_id, best_d = None, self._merge_dist_m
        for oid, o in self._objects.items():
            d = math.hypot(o.xy[0] - world_xy[0], o.xy[1] - world_xy[1])
            if d < best_d:
                best_id, best_d = oid, d
        if best_id is None:
            self._objects[self._next_id] = WorldObject(world_xy, category, var, 1, moving)
            self._next_id += 1
            return
        o = self._objects[best_id]
        # объединение двух гауссиан (information filter)
        w_old, w_new = 1.0 / o.var, 1.0 / var
        nx = (o.xy[0] * w_old + world_xy[0] * w_new) / (w_old + w_new)
        ny = (o.xy[1] * w_old + world_xy[1] * w_new) / (w_old + w_new)
        o.xy = (nx, ny)
        o.var = 1.0 / (w_old + w_new)
        o.hits += 1
        o.moving = moving or o.moving
        if category != "other":
            o.category = category

    def decay(self, max_objects: int = 200):
        """Подчистить переполнение карты (оставить самые «уверенные»)."""
        if len(self._objects) <= max_objects:
            return
        keep = sorted(self._objects.items(), key=lambda kv: (-kv[1].hits, kv[1].var))[:max_objects]
        self._objects = dict(keep)

    def snapshot(self) -> FusedState:
        return FusedState(self.kf.xy, self.kf.vxy, self.heading_deg,
                          list(self._objects.values()))
