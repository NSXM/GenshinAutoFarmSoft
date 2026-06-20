"""
Живой калибровщик HSV-порогов стрелки игрока — ПОВЕРХ игры, на cv2-трекбарах
(без PIL/tkinter; используем наш ScreenCapture). Подбирай пороги так, чтобы в правом
окне (маска) светилась ТОЛЬКО синяя стрелка — без конуса камеры, воды, пинов. Чем
чище маска, тем меньше глюков курса (медиана в minimap_reader добивает остаток).

Зачем: глюки стрелки (спайки до 147°, затяжные до 4 кадров) идут от мусора в маске
(конус/вода). Тут лечим в ИСТОЧНИКЕ; читается тот же регион и формула угла, что в
проде (MinimapReader.read_heading: самый центральный cyan-блоб, нос = дальняя точка).

Запуск:  .venv\\Scripts\\python.exe scripts\\calibrate_arrow.py
Клавиши: S — напечатать значения для config.yaml | Q — выход.
"""
from __future__ import annotations
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from genshin_nav.config import Config              # noqa: E402
from genshin_nav.capture.screen_capture import ScreenCapture  # noqa: E402

WIN = "calibrate arrow (S=save Q=quit)"
ROI_FRAC = 0.22   # как cfg.minimap.arrow_roi_frac


def _nothing(_):
    pass


def detect_angle(mm_bgr, lo, hi):
    """Повтор read_heading: самый ЦЕНТРАЛЬНЫЙ cyan-блоб, нос = дальняя точка."""
    hsv = cv2.cvtColor(mm_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    h, w = mask.shape
    cx, cy = w // 2, h // 2
    r = max(8, int(min(w, h) * ROI_FRAC))
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
        d = math.hypot(gx - cx, gy - cy)
        if d < best_d:
            best, best_d = c, d
    ang = None
    if best is not None and best_d <= r:
        M = cv2.moments(best)
        gx, gy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        pts = best.reshape(-1, 2).astype(np.float64)
        tip = pts[np.argmax(np.hypot(pts[:, 0] - gx, pts[:, 1] - gy))]
        ang = (math.degrees(math.atan2(tip[0] - gx, -(tip[1] - gy))) + 360) % 360
    return ang, mask, best


def main():
    cfg = Config.load(os.path.join(ROOT, "config.yaml"))
    l, t, w, h = cfg.minimap.region
    lo0 = list(cfg.minimap.arrow_hsv_low)
    hi0 = list(cfg.minimap.arrow_hsv_high)

    cap = ScreenCapture(cfg.capture.region, cfg.capture.backend,
                        cfg.capture.target_fps, cfg.capture.window_title)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 760, 470)
    try:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    for name, val, mx in (("H lo", lo0[0], 179), ("H hi", hi0[0], 179),
                          ("S lo", lo0[1], 255), ("S hi", hi0[1], 255),
                          ("V lo", lo0[2], 255), ("V hi", hi0[2], 255)):
        cv2.createTrackbar(name, WIN, val, mx, _nothing)

    print(f"[calib] регион миникарты=({l},{t},{w},{h}); старт HSV {lo0}-{hi0}")
    print("[calib] двигай трекбары: в правом окне (маска) — ТОЛЬКО стрелка. S — сохранить, Q — выход.")
    positioned = False
    try:
        while True:
            frame = cap.grab()
            if frame is None:
                cv2.waitKey(1); continue
            mm = frame[t:t + h, l:l + w]
            g = cv2.getTrackbarPos
            lo = [g("H lo", WIN), g("S lo", WIN), g("V lo", WIN)]
            hi = [g("H hi", WIN), g("S hi", WIN), g("V hi", WIN)]
            ang, mask, best = detect_angle(mm, lo, hi)

            vis = mm.copy()
            H, Wd = mm.shape[:2]
            cv2.circle(vis, (Wd // 2, H // 2), 2, (255, 255, 255), -1)
            if best is not None:
                cv2.drawContours(vis, [best], -1, (0, 255, 0), 1)
            if ang is not None:
                a = math.radians(ang)
                cv2.arrowedLine(vis, (Wd // 2, H // 2),
                                (int(Wd // 2 + 40 * math.sin(a)), int(H // 2 - 40 * math.cos(a))),
                                (0, 200, 255), 2, tipLength=0.3)
            S = 2
            visb = cv2.resize(vis, (Wd * S, H * S), interpolation=cv2.INTER_NEAREST)
            maskb = cv2.cvtColor(cv2.resize(mask, (Wd * S, H * S), interpolation=cv2.INTER_NEAREST),
                                 cv2.COLOR_GRAY2BGR)
            cv2.putText(visb, f"angle={'--' if ang is None else f'{ang:.1f}'}", (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)
            cv2.putText(maskb, "MASK: only arrow!", (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1)
            out = np.hstack([visb, maskb])
            cv2.imshow(WIN, out)
            if not positioned:
                cv2.moveWindow(WIN, 1100, 80); positioned = True

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), ord('Q')):
                break
            if k in (ord('s'), ord('S')):
                print("\n=== В config.yaml -> minimap: ===")
                print(f"  arrow_hsv_low:  [{lo[0]}, {lo[1]}, {lo[2]}]")
                print(f"  arrow_hsv_high: [{hi[0]}, {hi[1]}, {hi[2]}]")
                print("=================================\n")
    finally:
        cap.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()