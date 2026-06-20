"""
Чистая логика следования по маршруту (без захвата экрана и без ввода).

На вход — текущая поза (позиция в метрах + курс), на выход — решение: доворот,
движение прямо или маршрут пройден. Никаких побочных эффектов, поэтому логику
легко проверить синтетикой и переиспользовать как в dry-run, так и при живом вводе.

Конвенция углов — ровно как считается курс в minimap_reader: heading = atan2(-sx, sy),
где (sx, sy) — сдвиг карты. fused_pos копит сырой (sx, sy), поэтому в его системе
СЕВЕР = +y, но ВОСТОК = -x (ось x зеркальная). Чтобы пеленг на точку был согласован
с курсом, берём atan2(-dx, dy), dx=wp.x-px, dy=wp.y-py (а НЕ atan2(dx, dy)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from ..utils.geom import angle_diff_deg
from .route import Route


@dataclass
class FollowDecision:
    done: bool                 # маршрут пройден целиком
    wp_idx: int                # индекс текущей цели (или len при done)
    dist_m: float              # расстояние до текущей точки, м
    bearing_deg: float         # пеленг на точку, 0..360 (0=север, по часовой)
    heading_err_deg: float     # ошибка курса (bearing - heading), -180..180
    action: str                # "arrived_all" | "turn" | "go"
    turn_deg: float            # на сколько повернуть (= heading_err_deg); >0 по часовой


class RouteFollower:
    def __init__(self, route: Route, control_cfg):
        self.route = route
        self.cfg = control_cfg
        self.wp_idx = 0

    def _dist(self, player_xy, wp) -> float:
        return math.hypot(wp.x - player_xy[0], wp.y - player_xy[1])

    def step(self, player_xy: Tuple[float, float], heading_deg: float) -> FollowDecision:
        wps = self.route.waypoints
        tol = self.cfg.waypoint_tolerance_m
        look = getattr(self.cfg, "lookahead_m", 6.0)

        # Прогрессия: снап к ближайшей точке ВПЕРЁД (текущая достигнута ИЛИ следующая
        # уже ближе — значит прошли мимо). Привязывается к фактической позиции, не
        # залипает, если бот прошёл точку не ровно через её радиус. (Дикие проскоки
        # раньше были из-за взорванной/шумной позиции — её починили.)
        # ВАЖНО: "next ближе" разрешаем ТОЛЬКО когда мы уже в окрестности текущей
        # точки (d_cur < look). Иначе, стоя в начале, цикл каскадом проскакивает
        # пол-маршрута за один кадр (так бот улетал сразу на wp#9). Снап на одну
        # точку за вызов — прогрессия плавная.
        while self.wp_idx < len(wps) - 1:
            d_cur = self._dist(player_xy, wps[self.wp_idx])
            d_next = self._dist(player_xy, wps[self.wp_idx + 1])
            if d_cur < tol or (d_cur < look and d_next < d_cur):
                self.wp_idx += 1
            else:
                break

        last = len(wps) - 1
        if self.wp_idx >= last and self._dist(player_xy, wps[last]) < tol:
            return FollowDecision(True, last, 0.0, 0.0, 0.0, "arrived_all", 0.0)

        # Pure-pursuit: целимся в точку на ~look метров впереди (первая от wp_idx,
        # что дальше look; если все ближе — последняя). Дальняя цель = стабильный
        # пеленг, бот не дёргается у близких точек.
        tgt = self.wp_idx
        for i in range(self.wp_idx, len(wps)):
            tgt = i
            if self._dist(player_xy, wps[i]) >= look:
                break
        wp = wps[tgt]
        dx, dy = wp.x - player_xy[0], wp.y - player_xy[1]
        dist_cur = self._dist(player_xy, wps[self.wp_idx])   # до текущей (для лога/прогресса)
        # -dx: ось x в системе fused_pos зеркальная (восток=-x), см. докстринг модуля
        bearing = (math.degrees(math.atan2(-dx, dy)) + 360.0) % 360.0
        err = angle_diff_deg(bearing, heading_deg)           # кратчайший доворот, -180..180

        action = "turn" if abs(err) > self.cfg.align_tolerance_deg else "go"
        return FollowDecision(False, self.wp_idx, dist_cur, bearing, err, action, err)
