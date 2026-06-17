"""
Быстрый дамп: захват кропов миникарты на ПОЛНОЙ частоте (как живой debug_view) с
реальными таймстампами. По нему офлайн воспроизводим живой сбой курса и чиним по
настоящим данным, а не по синтетике.

Запуск:
    .venv\\Scripts\\python.exe scripts\\dump_fast.py
Иди РОВНО ВПЕРЁД (зажми W) всю запись (~3 сек). Камеру не крути.
Результат: diag/fast_dump.npz  -> пришли/оставь, я разберу сам.
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genshin_nav.config import Config
from genshin_nav.capture.screen_capture import ScreenCapture


def main():
    cfg = Config.load("config.yaml")
    cap = ScreenCapture(cfg.capture.region, cfg.capture.backend,
                        cfg.capture.target_fps, cfg.capture.window_title)
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diag")
    os.makedirs(outdir, exist_ok=True)
    l, t, w, h = cfg.minimap.region

    print("[fast] 3 сек на переключение в игру, потом ИДИ ВПЕРЁД (W) ~3 сек...")
    time.sleep(3)

    frames, times = [], []
    t_end = time.monotonic() + 3.0
    print("[fast] ИДУ! записываю на полной частоте (ЦВЕТ)...")
    while time.monotonic() < t_end:
        frame = cap.grab()
        if frame is None:
            continue
        mm = frame[t:t + h, l:l + w]
        frames.append(mm.copy())                 # BGR — нужен голубой конус камеры
        times.append(time.monotonic())
    cap.close()

    arr = np.stack(frames).astype(np.uint8)       # (N, H, W, 3) BGR
    ts = np.array(times, np.float64)
    path = os.path.join(outdir, "fast_dump.npz")
    np.savez_compressed(path, frames=arr, times=ts)
    dt = np.diff(ts)
    print(f"[fast] записано {len(frames)} кадров за 3с | средний FPS={1/dt.mean():.0f} "
          f"| dt медиана={np.median(dt)*1000:.1f}мс")
    print(f"[fast] сохранено -> {path}")


if __name__ == "__main__":
    main()
