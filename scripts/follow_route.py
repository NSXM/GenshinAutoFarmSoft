"""
Follower в режиме DRY-RUN: проверка «мозгов» следования по маршруту БЕЗ ввода.

Бот НЕ трогает мышь/клавиатуру. Он читает маршрут, на каждом кадре считает, где он
(fused_pos), куда следующая точка, какой нужен доворот, и показывает это решение —
в ОВЕРЛЕЕ поверх игры (маленькое окно в углу, always-on-top) и в консоли. Ты бежишь
сам и сверяешь: правильно ли бот понимает «куда и на сколько повернуть» и
переключается ли на следующую точку при подходе.

Реальный ввод (поворот камеры мышью + W), детект застревания и обход препятствий —
отдельные следующие фазы; здесь их нет.

Запуск:
    .venv\\Scripts\\python.exe scripts\\follow_route.py --route routes\\recorded_route.json
Выход: F9, либо Q в окне оверлея, либо Ctrl+C (ESC не годится — откроет меню Genshin).
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

from genshin_nav.config import Config
from genshin_nav.control.pose_tracker import PoseTracker
from genshin_nav.control.route import load_route
from genshin_nav.control.route_follower import RouteFollower

WIN = "follow compass (dry-run)"


def make_topmost(name: str):
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass


def _text(img, s, org, scale, color, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _compass(panel, cx, cy, R, heading_deg, bearing_deg):
    """Север-вверх компас: зелёная стрелка=heading, оранжевая=пеленг на точку.
    Сравнивается напрямую с миникартой Genshin (тоже север-вверх)."""
    cv2.circle(panel, (cx, cy), R, (90, 90, 90), 1)
    for lbl, ang in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
        lx = int(cx + (R + 12) * math.sin(math.radians(ang)))
        ly = int(cy - (R + 12) * math.cos(math.radians(ang)))
        _text(panel, lbl, (lx - 6, ly + 5), 0.5, (160, 160, 160), 1)

    def arrow(deg, color, thick):
        a = math.radians(deg)
        tx = int(cx + R * math.sin(a))
        ty = int(cy - R * math.cos(a))
        cv2.arrowedLine(panel, (cx, cy), (tx, ty), color, thick, tipLength=0.3)

    arrow(heading_deg, (0, 230, 0), 3)     # зелёная = куда смотрит персонаж
    arrow(bearing_deg, (0, 170, 255), 2)   # оранжевая = куда точка
    cv2.circle(panel, (cx, cy), 3, (255, 255, 255), -1)


def draw_overlay(pose, d, route):
    """Панель: действие + компас (heading vs bearing) + сырые числа."""
    n_total = len(route.waypoints)
    W, H = 300, 330
    panel = np.full((H, W, 3), 30, np.uint8)

    if d.done:
        _text(panel, "ARRIVED", (30, 120), 1.3, (0, 255, 255), 3)
        return panel

    if d.action == "go":
        col, big = (0, 230, 0), "GO"
    else:
        col = (0, 170, 255)
        big = "TURN RIGHT" if d.turn_deg > 0 else "TURN LEFT"

    _text(panel, big, (12, 40), 0.9, col, 2)
    if d.action == "turn":
        _text(panel, f"{abs(d.turn_deg):.0f} deg", (12, 70), 0.7, col, 2)

    _compass(panel, W // 2, 165, 60, pose.heading_deg, d.bearing_deg)

    wp = route.waypoints[d.wp_idx]
    px, py = pose.player_xy
    dx, dy = wp.x - px, wp.y - py
    _text(panel, f"wp #{d.wp_idx + 1}/{n_total}  d={d.dist_m:.1f}m", (12, 252), 0.5, (255, 255, 255), 1)
    _text(panel, f"hdg={pose.heading_deg:.0f} brg={d.bearing_deg:.0f} err={d.heading_err_deg:+.0f}",
          (12, 273), 0.5, (220, 220, 220), 1)
    _text(panel, f"pos=({px:.1f},{py:.1f}) wp=({wp.x:.1f},{wp.y:.1f})", (12, 294), 0.45, (180, 180, 180), 1)
    _text(panel, f"dx={dx:+.1f} dy={dy:+.1f}  move={'Y' if pose.moving else 'n'}",
          (12, 315), 0.45, (180, 180, 180), 1)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--route", default="routes/recorded_route.json")
    ap.add_argument("--log-dt", type=float, default=0.5, help="период консольного лога, сек")
    ap.add_argument("--no-overlay", action="store_true", help="без окна-оверлея, только консоль")
    args = ap.parse_args()

    route = load_route(args.route)
    if not route.waypoints:
        print(f"[follow] маршрут пуст: {args.route}")
        return

    try:
        import keyboard
        have_kb = True
    except Exception:
        have_kb = False
        print("[follow] модуль keyboard недоступен — выход по Q в окне или Ctrl+C")

    cfg = Config.load(args.config)
    follower = RouteFollower(route, cfg.control)
    tracker = PoseTracker(cfg)
    overlay = not args.no_overlay

    if overlay:
        make_topmost(WIN)
    print(f"[follow] DRY-RUN. Точек: {len(route.waypoints)}. 3 секунды на переключение в игру...")
    time.sleep(3)
    print("[follow] поехали (реального ввода НЕТ). F9 / Q в окне — выход.")

    positioned = False
    last_log = 0.0
    try:
        while True:
            pose = tracker.poll()
            if pose is None:
                time.sleep(0.001)
                continue
            d = follower.step(pose.player_xy, pose.heading_deg)

            if overlay:
                panel = draw_overlay(pose, d, route)
                cv2.imshow(WIN, panel)
                if not positioned:
                    cv2.moveWindow(WIN, 1500, 60)   # правый-верхний угол (далеко от миникарты)
                    positioned = True
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), ord('Q')):
                    print("[follow] Q — стоп")
                    break

            if d.done:
                print("[follow] МАРШРУТ ПРОЙДЕН (arrived_all)")
                if overlay:
                    time.sleep(1.5)
                break

            now = time.monotonic()
            if now - last_log >= args.log_dt:
                last_log = now
                turn_dir = "RIGHT" if d.turn_deg > 0 else "LEFT"
                print(
                    f"[dry] pos=({pose.player_xy[0]:7.2f},{pose.player_xy[1]:7.2f}) "
                    f"hdg={pose.heading_deg:6.1f} move={'Y' if pose.moving else 'n'} | "
                    f"wp#{d.wp_idx} d={d.dist_m:6.2f}m bearing={d.bearing_deg:6.1f} "
                    f"err={d.heading_err_deg:+6.1f} -> {d.action.upper():4s}"
                    + (f" {abs(d.turn_deg):4.1f} {turn_dir}" if d.action == 'turn' else "")
                )

            if have_kb and keyboard.is_pressed("f9"):
                print("[follow] F9 — стоп")
                break
            if not overlay:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[follow] прервано (Ctrl+C)")
    finally:
        tracker.close()
        if overlay:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
