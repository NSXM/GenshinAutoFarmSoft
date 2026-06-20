"""
Диагностика согласованности двух источников курса на shots/*.png.

Берём кадры по порядку как последовательность движения и считаем:
  * курс по cyan-СТРЕЛКЕ игрока (как в minimap_reader.read_heading) — по каждому кадру;
  * курс по СДВИГУ карты (как в _heading_from_motion) — по каждой соседней паре,
    при обоих знаках move_sign (+1 и -1).

Цель — увидеть, СОВПАДАЮТ ли численно стрелка и сдвиг карты и при каком знаке.
Если основной источник курса (стрелка) расходится с конвенцией follower'а
(сдвиг карты, восток=-x) — это и есть причина «кривого» движения.

Запуск:  python scripts/diag_heading_sources.py [region_l region_t region_w region_h]
По умолчанию регион миникарты берётся из config.yaml.
"""
from __future__ import annotations
import glob
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from genshin_nav.config import Config  # noqa: E402


def cyan_arrow_heading(mm_bgr, lo, hi, roi_frac):
    """Повтор read_heading: самый ЦЕНТРАЛЬНЫЙ cyan-блоб, центроид->дальняя точка."""
    hsv = cv2.cvtColor(mm_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    h, w = mask.shape
    cx, cy = w // 2, h // 2
    r = max(8, int(min(w, h) * roi_frac))
    roi = np.zeros_like(mask)
    cv2.circle(roi, (cx, cy), r, 255, -1)
    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_d = None, 1e9
    for c in cnts:
        if cv2.contourArea(c) < 30:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        gx, gy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        d = np.hypot(gx - cx, gy - cy)
        if d < best_d:
            best, best_d = c, d
    if best is None or best_d > r:
        return None, 0
    M = cv2.moments(best)
    gx, gy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    pts = best.reshape(-1, 2).astype(np.float64)
    tip = pts[np.argmax(np.hypot(pts[:, 0] - gx, pts[:, 1] - gy))]
    dx, dy = tip[0] - gx, tip[1] - gy
    heading = math.degrees(math.atan2(dx, -dy))
    return (heading + 360.0) % 360.0, int(cv2.contourArea(best))


def motion_window(shape, inner, outer):
    h, w = shape
    cx, cy = w // 2, h // 2
    rad = min(h, w)
    ring = np.zeros(shape, np.float32)
    cv2.circle(ring, (cx, cy), int(rad * outer), 1.0, -1)
    cv2.circle(ring, (cx, cy), int(rad * inner), 0.0, -1)
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    return ring * hann


def map_shift_heading(prev_mm, cur_mm, mcfg, sign):
    """Курс по сдвигу карты при заданном знаке (повтор _heading_from_motion ядра)."""
    win = motion_window(prev_mm.shape[:2], mcfg.motion_ring_inner_frac,
                        mcfg.motion_ring_outer_frac)
    g0 = (cv2.cvtColor(prev_mm, cv2.COLOR_BGR2GRAY).astype(np.float32)) * win
    g1 = (cv2.cvtColor(cur_mm, cv2.COLOR_BGR2GRAY).astype(np.float32)) * win
    (sx, sy), resp = cv2.phaseCorrelate(g0, g1)
    shift = math.hypot(sx, sy)
    ang = math.atan2(sign * sx, -(sign * sy))
    heading = (math.degrees(ang) + 360.0) % 360.0
    return heading, resp, shift, (sx, sy)


def ang_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def main():
    cfg = Config.load(os.path.join(ROOT, "config.yaml"))
    mcfg = cfg.minimap
    if len(sys.argv) >= 5:
        l, t, w, h = map(int, sys.argv[1:5])
    else:
        l, t, w, h = mcfg.region
    print(f"регион миникарты: ({l},{t},{w},{h})  move_sign в конфиге={mcfg.move_sign}")
    print(f"arrow HSV: {mcfg.arrow_hsv_low}-{mcfg.arrow_hsv_high} roi={mcfg.arrow_roi_frac}\n")

    files = sorted(glob.glob(os.path.join(ROOT, "shots", "*.png")))
    files = [f for f in files if os.path.basename(f).lower().startswith(("снимок", "shot", "mm_")) or "(" in f]
    files = [f for f in glob.glob(os.path.join(ROOT, "shots", "*.png"))
             if not os.path.basename(f).startswith(("view_", "mm_"))]
    files = sorted(files)
    if not files:
        print("нет png в shots/")
        return

    mms = []
    print("=== курс по СТРЕЛКЕ (по каждому кадру) ===")
    for f in files:
        full = cv2.imread(f)
        if full is None:
            # кириллические имена: читаем через numpy
            data = np.fromfile(f, dtype=np.uint8)
            full = cv2.imdecode(data, cv2.IMREAD_COLOR)
        mm = full[t:t + h, l:l + w]
        mms.append(mm)
        a, area = cyan_arrow_heading(mm, mcfg.arrow_hsv_low, mcfg.arrow_hsv_high,
                                     mcfg.arrow_roi_frac)
        astr = f"{a:6.1f}" if a is not None else "  None"
        print(f"  {os.path.basename(f):28s} arrow={astr}  area={area}")

    print("\n=== курс по СДВИГУ КАРТЫ (по соседним парам), оба знака ===")
    print("  pair                         resp  shift_px   (sx,sy)        sign=+1  sign=-1   arrow_prev")
    for i in range(len(mms) - 1):
        hp, _ = cyan_arrow_heading(mms[i], mcfg.arrow_hsv_low, mcfg.arrow_hsv_high,
                                   mcfg.arrow_roi_frac)
        hpos, rp, sh, (sx, sy) = map_shift_heading(mms[i], mms[i + 1], mcfg, +1)
        hneg, _, _, _ = map_shift_heading(mms[i], mms[i + 1], mcfg, -1)
        ap = f"{hp:6.1f}" if hp is not None else "  None"
        line = (f"  {i}->{i+1}  resp={rp:4.2f}  shift={sh:6.2f}  "
                f"({sx:+6.2f},{sy:+6.2f})  +1={hpos:6.1f}  -1={hneg:6.1f}  arrow={ap}")
        print(line)
        if hp is not None:
            dp = abs(ang_diff(hpos, hp))
            dn = abs(ang_diff(hneg, hp))
            better = "+1" if dp < dn else "-1"
            print(f"        |arrow-карта|: +1={dp:5.1f}°  -1={dn:5.1f}°  -> ближе к стрелке: sign={better}")

    print("\nИтог: если стрелка и сдвиг карты совпадают только при ОДНОМ знаке —")
    print("это правильный move_sign. Если расходятся всегда — источники в разных")
    print("системах, и follower надо согласовать с выбранным источником курса.")


if __name__ == "__main__":
    main()
