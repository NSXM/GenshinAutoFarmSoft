"""
Диагностика по diag/fast_dump.npz — РЕАЛЬНОЕ движение прямо вперёд (W зажат, камера
не крутится). При ходьбе прямо стрелка(=камера) и сдвиг карты обязаны давать ОДИН и
тот же курс. Это даёт однозначный ответ:
  * какой move_sign согласует сдвиг карты со стрелкой;
  * насколько стабильны оба источника на полной частоте;
  * есть ли лаг/рассинхрон, дающий «кривое» руление.

Запуск:  .venv\\Scripts\\python.exe scripts\\diag_fast_dump.py
"""
from __future__ import annotations
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from genshin_nav.config import Config  # noqa: E402
from scripts.diag_heading_sources import (  # noqa: E402
    cyan_arrow_heading, map_shift_heading, ang_diff,
)


def circ_stats(degs):
    """Круговое среднее и круговой СКО (в градусах) по списку углов."""
    rad = np.radians(degs)
    c, s = np.cos(rad).mean(), np.sin(rad).mean()
    mean = (math.degrees(math.atan2(s, c)) + 360.0) % 360.0
    R = math.hypot(c, s)
    std = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(R, 1e-9)))))
    return mean, std


def main():
    cfg = Config.load(os.path.join(ROOT, "config.yaml"))
    mcfg = cfg.minimap
    path = os.path.join(ROOT, "diag", "fast_dump.npz")
    d = np.load(path)
    frames, times = d["frames"], d["times"]
    n = len(frames)
    dt = np.diff(times)
    print(f"кадров={n}  длительность={times[-1]-times[0]:.2f}с  "
          f"FPS={ (n-1)/(times[-1]-times[0]):.1f}  dt медиана={np.median(dt)*1000:.1f}мс")
    print(f"move_sign в конфиге={mcfg.move_sign}  "
          f"floor={mcfg.heading_move_floor_px}px\n")

    # --- курс по стрелке покадрово ---
    arrow = []
    for f in frames:
        a, ar = cyan_arrow_heading(f, mcfg.arrow_hsv_low, mcfg.arrow_hsv_high,
                                   mcfg.arrow_roi_frac)
        arrow.append(a)
    valid_arrow = [a for a in arrow if a is not None]
    if valid_arrow:
        am, asd = circ_stats(valid_arrow)
        print(f"СТРЕЛКА: валидных {len(valid_arrow)}/{n}  "
              f"средн={am:.1f}°  кругСКО={asd:.1f}°")

    # --- курс по сдвигу карты, оба знака; собираем только где shift>floor ---
    pos_h, neg_h, shifts, resps = [], [], [], []
    d_pos, d_neg = [], []
    for i in range(n - 1):
        hp, rp, sh, (sx, sy) = map_shift_heading(frames[i], frames[i+1], mcfg, +1)
        hn, _, _, _ = map_shift_heading(frames[i], frames[i+1], mcfg, -1)
        shifts.append(sh)
        resps.append(rp)
        if sh < mcfg.heading_move_floor_px:
            continue
        pos_h.append(hp)
        neg_h.append(hn)
        ap = arrow[i]
        if ap is not None:
            d_pos.append(abs(ang_diff(hp, ap)))
            d_neg.append(abs(ang_diff(hn, ap)))

    shifts = np.array(shifts)
    print(f"СДВИГ КАРТЫ: shift медиана={np.median(shifts):.2f}px  "
          f"max={shifts.max():.2f}px  кадров>floor={len(pos_h)}/{n-1}  "
          f"resp медиана={np.median(resps):.2f}\n")

    if pos_h:
        pm, psd = circ_stats(pos_h)
        nm, nsd = circ_stats(neg_h)
        print(f"  sign=+1: средн={pm:.1f}°  кругСКО={psd:.1f}°")
        print(f"  sign=-1: средн={nm:.1f}°  кругСКО={nsd:.1f}°")

    if d_pos:
        mp, mn = np.mean(d_pos), np.mean(d_neg)
        print(f"\n|стрелка - карта| (движение прямо => должно быть ~0):")
        print(f"  sign=+1: средн={mp:.1f}°  медиана={np.median(d_pos):.1f}°")
        print(f"  sign=-1: средн={mn:.1f}°  медиана={np.median(d_neg):.1f}°")
        win = "+1" if mp < mn else "-1"
        print(f"\n  => ПРАВИЛЬНЫЙ move_sign = {win}  "
              f"(стрелка и карта совпадают при нём)")
        if min(mp, mn) > 25:
            print("  ВНИМАНИЕ: даже лучший знак даёт >25° расхождения — "
                  "источники в разных системах ИЛИ камера всё же крутилась.")


if __name__ == "__main__":
    main()
