"""
Запись маршрута: бегаешь по миру руками, роняешь точки маршрута.

  F8  — записать текущую позицию (fused_pos) как точку маршрута.
  F9  — сохранить и выйти.
  --auto N — дополнительно ронять точку автоматически каждые N метров пути
             (гуще «крошки» -> глаже следование). По умолчанию только F8.

ВАЖНО: клавиши ловятся глобально, но игра в фокусе тоже их получает. Поэтому
ПРОБЕЛ (=прыжок) и ESC (=меню) не годятся — используем F8/F9, которые Genshin
не занимает. Если у тебя F8/F9 чем-то заняты — задай свою через --key.

Использует тот же рабочий пайплайн, что и debug_view (PoseTracker: миникарта +
fusion), поэтому точки пишутся в тех же метрах, в которых потом работает follower.

Запуск:
    .venv\\Scripts\\python.exe scripts\\record_route.py
    .venv\\Scripts\\python.exe scripts\\record_route.py --auto 3 --out routes\\my_route.json
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genshin_nav.config import Config
from genshin_nav.control.pose_tracker import PoseTracker
from genshin_nav.control.route import Route, Waypoint, save_route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="routes/recorded_route.json")
    ap.add_argument("--name", default="recorded")
    ap.add_argument("--auto", type=float, default=0.0,
                    help="ронять точку каждые N метров пути (0 = выкл, только клавиша)")
    ap.add_argument("--key", default="f8", help="клавиша записи точки (Genshin её не должен занимать)")
    ap.add_argument("--stop-key", default="f9", help="клавиша сохранить и выйти")
    args = ap.parse_args()

    try:
        import keyboard
    except Exception:
        print("Нужен модуль keyboard: pip install keyboard")
        return

    cfg = Config.load(args.config)
    tracker = PoseTracker(cfg)

    wps: list[Waypoint] = []
    last_auto_xy = None

    print("[record] 3 секунды на переключение в игру...")
    time.sleep(3)
    print(f"[record] {args.key.upper()} — записать точку, {args.stop_key.upper()} — сохранить и выйти"
          + (f" | авто-точка каждые {args.auto:.1f} м" if args.auto > 0 else ""))

    try:
        while True:
            pose = tracker.poll()
            if pose is None:
                time.sleep(0.001)
                continue
            px, py = pose.player_xy

            # ручная точка по клавише
            if keyboard.is_pressed(args.key):
                wps.append(Waypoint(round(px, 2), round(py, 2)))
                last_auto_xy = (px, py)
                print(f"  +точка #{len(wps)} (ручная): ({px:.2f}, {py:.2f})")
                time.sleep(0.3)            # антидребезг

            # авто-точка по пройденному пути
            elif args.auto > 0:
                if last_auto_xy is None:
                    last_auto_xy = (px, py)
                elif math.hypot(px - last_auto_xy[0], py - last_auto_xy[1]) >= args.auto:
                    wps.append(Waypoint(round(px, 2), round(py, 2)))
                    last_auto_xy = (px, py)
                    print(f"  +точка #{len(wps)} (авто): ({px:.2f}, {py:.2f})")

            if keyboard.is_pressed(args.stop_key):
                break
            time.sleep(0.02)
    finally:
        tracker.close()

    route = Route(name=args.name,
                  minimap_meters_per_px=cfg.minimap.minimap_meters_per_px,
                  waypoints=wps)
    save_route(route, args.out)
    print(f"[record] сохранено {len(wps)} точек -> {args.out}")


if __name__ == "__main__":
    main()
