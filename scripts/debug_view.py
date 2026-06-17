"""
Живой оверлей отладки — БЕЗ какого-либо ввода в игру.

Главный инструмент проверки калибровки: подтвердить, что регион миникарты задан
верно и стрелка/курс детектятся. В реальном времени рисует:
  * кроп миникарты крупно;
  * обнаруженную стрелку игрока и вектор heading;
  * численные heading / world_xy / delta + FPS;
  * (опц.) точки optical-flow на уменьшенном полном кадре.

Запуск:
    python scripts/debug_view.py
    python scripts/debug_view.py --flow      # ещё и точки optical-flow

Выход: клавиша Q или ESC в окне.
"""
from __future__ import annotations

import argparse
import math
import time

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genshin_nav.config import Config
from genshin_nav.capture.screen_capture import ScreenCapture
from genshin_nav.minimap.minimap_reader import MinimapReader
from genshin_nav.fusion.estimator import FusionEstimator


def _arrow(panel, cx, cy, deg, color, length_frac, thick):
    a = math.radians(deg)
    ln = int(min(panel.shape[:2]) * length_frac)
    tx = int(cx + ln * math.sin(a))
    ty = int(cy - ln * math.cos(a))
    cv2.arrowedLine(panel, (cx, cy), (tx, ty), color, thick, tipLength=0.25)


def draw_minimap_panel(mm_bgr, reading, scale=2, player_arrow_deg=None):
    panel = cv2.resize(mm_bgr, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_NEAREST)
    H, W = panel.shape[:2]
    cx, cy = W // 2, H // 2
    cv2.circle(panel, (cx, cy), 4, (0, 0, 255), -1)
    # конус камеры — тонкий, пурпурный, для сравнения
    if reading.arrow_heading_deg is not None:
        _arrow(panel, cx, cy, reading.arrow_heading_deg, (255, 0, 255), 0.3, 1)
    # СТРЕЛКА ИГРОКА (read_heading) — ЖЁЛТАЯ: кандидат в быстрый датчик курса
    if player_arrow_deg is not None:
        _arrow(panel, cx, cy, player_arrow_deg, (0, 255, 255), 0.36, 2)
    # курс из движения (сдвиг карты) — толстый зелёный
    if reading.heading_deg is not None:
        _arrow(panel, cx, cy, reading.heading_deg, (0, 255, 0), 0.42, 2)
    return panel


def put_lines(img, lines, org=(8, 20), color=(255, 255, 255)):
    y = org[1]
    for ln in lines:
        cv2.putText(img, ln, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, ln, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
        y += 22


WIN_MAIN = "minimap debug"
WIN_FLOW = "optical flow"


def make_topmost(name: str):
    """Окно поверх всех — чтобы висело над игрой в режиме 'без рамок'."""
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass  # старые сборки OpenCV без TOPMOST — не критично


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--flow", action="store_true", help="показать точки optical-flow")
    ap.add_argument("--no-topmost", action="store_true",
                    help="не держать окно поверх всех")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = ScreenCapture(cfg.capture.region, cfg.capture.backend,
                        cfg.capture.target_fps, cfg.capture.window_title)
    mm = MinimapReader(cfg.minimap)
    fusion = FusionEstimator(cfg)

    print("[debug] 3 секунды на переключение в игру...")
    time.sleep(3)

    if not args.no_topmost:
        make_topmost(WIN_MAIN)
        if args.flow:
            make_topmost(WIN_FLOW)

    positioned = False
    flow_pts = None
    prev_gray = None
    last_t = time.monotonic()
    fps = 0.0
    print("[debug] окно поверх игры (always-on-top). Q/ESC — выход. Ввод в игру не идёт.")
    print("[debug] если мешает — закрой и запусти с --no-topmost.")
    try:
        while True:
            frame = cap.grab()
            if frame is None:
                time.sleep(0.001)
                continue
            now = time.monotonic()
            dt = max(1e-3, now - last_t); last_t = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

            r = mm.read(frame)
            vx = (r.delta_xy_m[0] / dt, r.delta_xy_m[1] / dt) if r.delta_xy_m else None
            fusion.step_player(dt, r.world_xy_m, vx, r.heading_deg)
            st = fusion.snapshot()

            l, t, w, h = cfg.minimap.region
            mm_bgr = frame[t:t + h, l:l + w]
            player_arrow = mm.read_heading(mm_bgr)      # стрелка игрока — тест быстрого датчика
            panel = draw_minimap_panel(mm_bgr, r, player_arrow_deg=player_arrow)

            hd = f"{r.heading_deg:.1f}" if r.heading_deg is not None else "stand"
            ar = f"{r.arrow_heading_deg:.1f}" if r.arrow_heading_deg is not None else "-"
            pa = f"{player_arrow:.1f}" if player_arrow is not None else "-"
            wx = f"({r.world_xy_m[0]:.1f},{r.world_xy_m[1]:.1f})" if r.world_xy_m else "no atlas"
            dl = f"({r.delta_xy_m[0]:+.2f},{r.delta_xy_m[1]:+.2f})" if r.delta_xy_m else "-"
            put_lines(panel, [
                f"FPS: {fps:4.1f}",
                f"heading(FILT): {hd}",   # GREEN = отфильтрованный курс (идёт в бота)
                f"arrow(RAW): {pa}",      # YELLOW = сырая стрелка без фильтра
                f"cone(cam): {ar}",       # magenta arrow = конус камеры
                f"delta_m: {dl}",
                f"fused_pos: ({st.player_xy[0]:.1f},{st.player_xy[1]:.1f})",
            ])
            cv2.imshow(WIN_MAIN, panel)

            # позиционируем один раз: правый-верхний угол, подальше от миникарты
            # игры (она слева-сверху) — иначе оверлей перекроет то, что мы же
            # захватываем, и поймаем сам себя в кадр.
            if not positioned:
                screen_w = frame.shape[1]
                px = max(0, screen_w - panel.shape[1] - 30)
                cv2.moveWindow(WIN_MAIN, px, 50)
                if args.flow:
                    cv2.moveWindow(WIN_FLOW, px, 50 + panel.shape[0] + 40)
                positioned = True

            if args.flow:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    if flow_pts is None or len(flow_pts) < 30:
                        flow_pts = cv2.goodFeaturesToTrack(
                            prev_gray, maxCorners=200, qualityLevel=0.01,
                            minDistance=8, blockSize=7)
                    if flow_pts is not None:
                        p1, stt, _ = cv2.calcOpticalFlowPyrLK(
                            prev_gray, gray, flow_pts, None)
                        vis = cv2.resize(frame, None, fx=0.5, fy=0.5)
                        if p1 is not None:
                            good = p1[stt.reshape(-1).astype(bool)]
                            for px, py in good.reshape(-1, 2):
                                cv2.circle(vis, (int(px * 0.5), int(py * 0.5)),
                                           2, (0, 255, 255), -1)
                            flow_pts = good.reshape(-1, 1, 2)
                        cv2.imshow(WIN_FLOW, vis)
                prev_gray = gray

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
    finally:
        cap.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
