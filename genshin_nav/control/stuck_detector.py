"""
Шаг 6 ТЗ — детект застревания.

Если кнопка движения зажата, а позиция персонажа (из fusion: миникарта + flow)
за контрольное окно сместилась меньше порога — фиксируем застревание и отдаём
рекомендацию по выходу (прыжок / поворот / смена траектории).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass
class StuckVerdict:
    stuck: bool
    displacement_m: float
    suggestion: str          # "" | "jump" | "turn" | "reroute"


class StuckDetector:
    def __init__(self, cfg_control):
        self.cfg = cfg_control
        self._hist: Deque[Tuple[float, float, float]] = deque()  # (t, x, y)
        self._recoveries = 0
        self._last_recovery_t = 0.0

    def reset(self):
        self._hist.clear()
        self._recoveries = 0

    def update(self, t: float, xy: Tuple[float, float],
               is_moving_cmd: bool) -> StuckVerdict:
        """t — монотонное время (сек), xy — позиция персонажа, is_moving_cmd —
        зажата ли клавиша движения прямо сейчас."""
        self._hist.append((t, xy[0], xy[1]))
        win = self.cfg.stuck_window_s
        while self._hist and t - self._hist[0][0] > win:
            self._hist.popleft()

        if not is_moving_cmd or len(self._hist) < 2 or (t - self._hist[0][0]) < win * 0.8:
            return StuckVerdict(False, 0.0, "")

        x0, y0 = self._hist[0][1], self._hist[0][2]
        disp = ((xy[0] - x0) ** 2 + (xy[1] - y0) ** 2) ** 0.5
        if disp >= self.cfg.stuck_min_displacement_m:
            self._recoveries = 0
            return StuckVerdict(False, disp, "")

        # застряли — эскалация рекавери
        self._recoveries += 1
        if self._recoveries == 1:
            sug = "jump"
        elif self._recoveries in (2, 3):
            sug = "turn"
        else:
            sug = "reroute"
        self._hist.clear()           # дать рекавери время сработать
        return StuckVerdict(True, disp, sug)
