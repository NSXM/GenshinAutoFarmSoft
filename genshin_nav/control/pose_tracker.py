"""
Общий источник позы персонажа для рекордера и follower'а.

Оборачивает рабочий пайплайн миникарты (ScreenCapture -> MinimapReader) и на
каждый poll() отдаёт текущую позу: позицию в метрах (одометрия), курс (0=север,
по часовой) и флаг движения. Так рекордер, follower и исполнитель не дублируют
один и тот же цикл захвата.

ВАЖНО (одометрия): позицию копим НАПРЯМУЮ суммой delta_xy_m (готовое смещение
карты в метрах за кадр). Раньше шли через скорость v=delta/dt и Kalman, который
интегрировал v×dt_следующий — при скачках dt (рывки камеры, пол max(1e-3,...))
это умножало смещение в десятки раз и позиция «взрывалась» на сотни метров.
Прямая сумма delta dt-независима и устойчива. (Фьюжн с атласом/SAM вернём позже,
когда появится абсолютный источник позиции.)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from ..capture.screen_capture import ScreenCapture
from ..minimap.minimap_reader import MinimapReader


@dataclass
class Pose:
    player_xy: Tuple[float, float]   # позиция в метрах (одометрия), отсчёт от старта
    heading_deg: float               # курс: 0=север, по часовой
    moving: bool                     # детектор движения миникарты (гейт одометрии)
    conf: float                      # уверенность чтения миникарты
    dt: float                        # секунд с прошлого poll()


class PoseTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cap = ScreenCapture(cfg.capture.region, cfg.capture.backend,
                                 cfg.capture.target_fps, cfg.capture.window_title)
        self.minimap = MinimapReader(cfg.minimap)
        self._pos = [0.0, 0.0]            # накопленная позиция (метры)
        self._last_heading = 0.0
        self._last_t = time.monotonic()

    def set_position(self, x: float, y: float):
        """Задать текущую позицию (метры). Нужно для выравнивания одометрии с
        системой координат маршрута: ставим бота в первую точку перед стартом."""
        self._pos = [float(x), float(y)]

    def poll(self) -> Optional[Pose]:
        """Один кадр -> Pose. None, если кадр ещё не готов (dxcam может вернуть None)."""
        frame = self.cap.grab()
        if frame is None:
            return None
        now = time.monotonic()
        dt = max(1e-3, now - self._last_t)
        self._last_t = now

        r = self.minimap.read(frame)
        if r.delta_xy_m is not None:          # delta уже гейтится по движению в MinimapReader
            self._pos[0] += r.delta_xy_m[0]
            self._pos[1] += r.delta_xy_m[1]
        if r.heading_deg is not None:
            self._last_heading = r.heading_deg
        return Pose(player_xy=(self._pos[0], self._pos[1]),
                    heading_deg=self._last_heading,
                    moving=self.minimap._moving, conf=r.confidence, dt=dt)

    def close(self):
        try:
            self.cap.close()
        except Exception:
            pass
