"""
Захват экрана.

Бэкенды:
  - dxcam: DXGI Desktop Duplication, самый быстрый на Windows (до 240 FPS),
           отдаёт кадр напрямую как numpy-массив. Рекомендуется для realtime.
  - mss:   кроссплатформенный фолбэк, медленнее, но без дополнительных драйверов.

Возвращает кадры в формате BGR (как привык OpenCV).
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def find_window_region(title: str) -> Optional[Tuple[int, int, int, int]]:
    """Найти прямоугольник окна игры по заголовку (left, top, width, height)."""
    try:
        import pygetwindow as gw
    except ImportError:
        return None
    wins = [w for w in gw.getWindowsWithTitle(title) if w.width > 0]
    if not wins:
        return None
    w = wins[0]
    return (w.left, w.top, w.width, w.height)


class ScreenCapture:
    def __init__(self, region: Optional[Tuple[int, int, int, int]] = None,
                 backend: str = "dxcam", target_fps: int = 60,
                 window_title: str = "Genshin Impact"):
        self.region = region or find_window_region(window_title)
        self.backend = backend
        self.target_fps = target_fps
        self._cam = None
        self._sct = None
        self._init_backend()

    def _init_backend(self):
        if self.backend == "dxcam":
            try:
                import dxcam
                self._cam = dxcam.create(output_color="BGR")
                sw, sh = self._cam.width, self._cam.height
                # dxcam region формат: (left, top, right, bottom), строго в пределах экрана
                self._dx_region = None
                if self.region:
                    l, t, w, h = self.region
                    # клампим к границам экрана (maximized/borderless окна часто
                    # дают left=-8 или край за пределами -> dxcam ругается)
                    left = max(0, min(l, sw - 1))
                    top = max(0, min(t, sh - 1))
                    right = max(left + 1, min(l + w, sw))
                    bottom = max(top + 1, min(t + h, sh))
                    if (left, top, right, bottom) != (0, 0, sw, sh):
                        self._dx_region = (left, top, right, bottom)
                    # иначе полноэкранный режим: dx_region=None -> grab всего экрана
                    if (l, t, w, h) != (left, top, right - left, bottom - top):
                        print(f"[capture] регион {self.region} склампен под экран "
                              f"{sw}x{sh} -> {self._dx_region or 'весь экран'}")
                return
            except Exception as e:  # pragma: no cover
                print(f"[capture] dxcam недоступен ({e}), переключаюсь на mss")
                self.backend = "mss"
        # mss
        import mss
        self._sct = mss.mss()

    def grab(self) -> Optional[np.ndarray]:
        """Один кадр как BGR ndarray (H, W, 3) или None, если кадр не готов."""
        if self.backend == "dxcam":
            frame = self._cam.grab(region=self._dx_region) if self._dx_region else self._cam.grab()
            return frame  # может быть None, если кадр не обновился — это нормально
        # mss
        if self.region:
            l, t, w, h = self.region
            mon = {"left": l, "top": t, "width": w, "height": h}
        else:
            mon = self._sct.monitors[1]
        img = np.asarray(self._sct.grab(mon))  # BGRA
        return img[:, :, :3]

    def grab_blocking(self) -> np.ndarray:
        """Гарантированно вернуть кадр (повторяет grab, пока не получит непустой)."""
        for _ in range(1000):
            f = self.grab()
            if f is not None:
                return f
            time.sleep(0.002)          # не жечь CPU busy-loop'ом, пока кадр не готов
        raise RuntimeError("Не удалось получить кадр с экрана")

    @staticmethod
    def crop(frame: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        l, t, w, h = region
        return frame[t:t + h, l:l + w]

    def close(self):
        if self._cam is not None:
            try:
                self._cam.release()
            except Exception:
                pass
        if self._sct is not None:
            self._sct.close()
