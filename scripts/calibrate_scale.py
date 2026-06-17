"""
Калибровка масштаба миникарты: minimap_meters_per_px (метров на 1 пиксель сдвига).

Как меряем: бежишь РОВНО ПРЯМО известное время T, скрипт суммирует векторы сдвига
текстуры карты (для прямого бега |сумма| = пройденный путь в пикселях). Тогда
    масштаб = (скорость × T) / пиксели.
Скорость бега в Genshin ≈ 6 м/с (зависит от персонажа — можно задать --speed).

Маскирование/интервалы берутся те же, что в рантайме (MinimapReader), чтобы
масштаб совпадал с реально используемым сдвигом.

Запуск (в открытом мире, БЕГ по прямой, камеру не крутить):
    .venv\\Scripts\\python.exe scripts\\calibrate_scale.py --write
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genshin_nav.config import Config, update_config_value
from genshin_nav.capture.screen_capture import ScreenCapture
from genshin_nav.minimap.minimap_reader import MinimapReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--duration", type=float, default=5.0, help="сколько секунд бежать")
    ap.add_argument("--speed", type=float, default=6.0, help="скорость бега, м/с (Genshin ~6)")
    ap.add_argument("--write", action="store_true", help="записать minimap_meters_per_px")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = ScreenCapture(cfg.capture.region, cfg.capture.backend,
                        cfg.capture.target_fps, cfg.capture.window_title)
    mm = MinimapReader(cfg.minimap)
    l, t, w, h = cfg.minimap.region

    print(f"[scale] 3 сек на переключение в игру, потом БЕГИ ПРЯМО {args.duration:.0f} сек "
          f"(камеру НЕ крути)...")
    time.sleep(3)
    print("[scale] БЕГИ!")

    prev = None
    prev_t = 0.0
    acc = np.zeros(2)
    n = 0
    t_start = None       # время кадра-ИСТОЧНИКА первого учтённого сэмпла
    t_last = None        # время последнего учтённого сэмпла
    t_end = time.monotonic() + args.duration
    while time.monotonic() < t_end:
        frame = cap.grab()
        if frame is None:
            continue
        now = time.monotonic()
        crop = frame[t:t + h, l:l + w]
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) * mm._motion_window(
            (crop.shape[0], crop.shape[1]))
        if prev is None:
            prev = g; prev_t = now
            continue
        if now - prev_t < cfg.minimap.heading_sample_dt:
            continue
        from_t = prev_t                # время кадра, ОТ которого считаем этот сдвиг
        (sx, sy), resp = cv2.phaseCorrelate(prev, g)
        prev = g; prev_t = now
        if resp >= cfg.minimap.heading_min_response:
            acc += (sx, sy)
            n += 1
            if t_start is None:
                t_start = from_t       # начало интервала, покрытого суммарным сдвигом
            t_last = now
    cap.close()

    px = math.hypot(acc[0], acc[1])
    print(f"[scale] сэмплов: {n}, суммарный сдвиг карты: {px:.1f} px "
          f"(вектор {acc[0]:+.1f},{acc[1]:+.1f})")
    if px < 3.0 or t_start is None or t_last is None or t_last <= t_start:
        print("[scale] слишком малый сдвиг — мало текстуры или не бежал. Повтори в "
              "текстурной зоне, бегом по прямой.")
        return
    elapsed = t_last - t_start         # фактическое время накопления (без прогрева/пропусков)
    dist = args.speed * elapsed
    scale = dist / px
    print(f"[scale] дистанция ≈ {dist:.1f} м (скорость {args.speed} × {elapsed:.2f}с фактич.)")
    print(f"[scale] minimap_meters_per_px ≈ {scale:.3f}")
    if args.write:
        ok = update_config_value(args.config, "minimap.minimap_meters_per_px", round(scale, 3))
        print(f"[scale] {'записано' if ok else 'НЕ найден ключ'} в {args.config}")
    else:
        print("[scale] запусти с --write, чтобы сохранить в config.yaml")


if __name__ == "__main__":
    main()
