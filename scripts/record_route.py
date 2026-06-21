"""
Запись маршрута: пробегаешь путь руками, бот запоминает его и потом повторяет.

ВАЖНО (модель позиции): по умолчанию маршрут пишется DEAD-RECKONING'ом — позиция
интегрируется из КУРСА СТРЕЛКИ × скорость, ТОЧНО ТАК ЖЕ, как потом едет
исполнитель (route_runner). Поэтому система координат записи и езды совпадают,
между ними нет поворота/зеркала, и бот воспроизводит ровно твой путь. Одометрия
миникарты на зуме ~4 м/px искажена и поворачивает маршрут — поэтому НЕ она.

  → Зажми W и БЕГИ по маршруту НЕПРЕРЫВНО (не останавливаясь), поворачивая
    камеру/мышь по ходу. Скорость бега держи обычную (= control.dead_reckon_speed).

  F8  — записать точку вручную.
  F7  — записать точку-ТЕЛЕПОРТ = КОНЕЦ маршрута (сразу сохраняет и выходит).
  F9  — сохранить и выйти.
  --auto N — ронять точку автоматически каждые N метров пути (рекомендуется, напр. 2).
  --mode odometry — писать по одометрии миникарты (старое; обычно хуже).

Клавиши ловятся глобально; ПРОБЕЛ/ESC заняты игрой, поэтому F8/F9.

Запуск:
    .venv\\Scripts\\python.exe scripts\\record_route.py --auto 2
    .venv\\Scripts\\python.exe scripts\\record_route.py --auto 2 --out routes\\my_route.json
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
from genshin_nav.minimap.localizer import make_fingerprint, save_fingerprints, fp_path_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="routes/recorded_route.json")
    ap.add_argument("--name", default="recorded")
    ap.add_argument("--auto", type=float, default=0.0,
                    help="ронять точку каждые N метров пути (0 = выкл, только клавиша). Реком. 2")
    ap.add_argument("--mode", choices=["dead_reckon", "odometry"], default="dead_reckon",
                    help="dead_reckon (как едет бот; реком.) | odometry (сдвиг миникарты)")
    ap.add_argument("--key", default="f8", help="клавиша записи точки (Genshin её не должен занимать)")
    ap.add_argument("--tp-key", default="f7", help="клавиша записи точки-ТЕЛЕПОРТА (action=teleport)")
    ap.add_argument("--stop-key", default="f9", help="клавиша сохранить и выйти")
    args = ap.parse_args()

    try:
        import keyboard
    except Exception:
        print("Нужен модуль keyboard: pip install keyboard")
        return

    cfg = Config.load(args.config)
    tracker = PoseTracker(cfg)
    spd = cfg.control.dead_reckon_speed       # та же скорость, что у исполнителя

    wps: list[Waypoint] = []
    fps: list = []                # отпечаток миникарты на каждой точке (для локализации)
    last_auto_xy = None
    start_heading = None          # курс по стрелке в начале движения по маршруту
    head_at_first = None          # фолбэк: курс на самой первой точке

    def grab_fp():
        """Отпечаток миникарты текущего кадра (для teach-and-repeat локализации)."""
        f = tracker.last_frame
        if f is None:
            return None
        l, t, w, h = cfg.minimap.region
        return make_fingerprint(f[t:t + h, l:l + w])

    # DEAD-RECKONING позиции (та же формула, что в route_runner): смещение на курс
    # θ = (-sinθ, cosθ)·spd·dt. Система координат совпадает с follower (восток=-x).
    dr = [0.0, 0.0]
    dr_t = None
    started = False                       # запись начнётся только после первого нажатия W
    move_key = cfg.control.move_key

    print("[record] 3 секунды на переключение в игру...")
    time.sleep(3)
    mode_str = ("DEAD-RECKON (зажми W и БЕГИ непрерывно!)" if args.mode == "dead_reckon"
                else "ODOMETRY (сдвиг миникарты)")
    print(f"[record] режим: {mode_str}")
    print(f"[record] запись точек начнётся, как только нажмёшь {move_key.upper()} (движение).")
    print(f"[record] {args.key.upper()} — точка, {args.tp_key.upper()} — точка-ТЕЛЕПОРТ (= конец, сохранить и выйти), "
          f"{args.stop_key.upper()} — сохранить и выйти"
          + (f" | авто-точка каждые {args.auto:.1f} м" if args.auto > 0 else ""))

    try:
        while True:
            pose = tracker.poll()
            if pose is None:
                time.sleep(0.001)
                continue

            # ждём первого нажатия W — до него ничего не пишем (чистый старт без
            # фантомного дрейфа, пока стоишь). F9 работает и тут.
            if not started:
                if keyboard.is_pressed(args.stop_key):
                    break
                if keyboard.is_pressed(move_key):
                    started = True
                    dr_t = None           # сбросить часы, чтобы первый dt не был огромным
                    print(f"[record] {move_key.upper()} нажат — старт записи точек")
                else:
                    time.sleep(0.01)
                    continue

            # позиция: dead-reckon (по курсу) или одометрия миникарты
            now = time.monotonic()
            if args.mode == "dead_reckon":
                if dr_t is not None:
                    dt = min(0.2, max(0.0, now - dr_t))
                    rad = math.radians(pose.heading_deg)
                    dr[0] += -math.sin(rad) * spd * dt
                    dr[1] += math.cos(rad) * spd * dt
                dr_t = now
                px, py = dr[0], dr[1]
            else:
                px, py = pose.player_xy

            # точка-ТЕЛЕПОРТ по клавише — это КОНЕЦ маршрута: записываем и выходим
            if keyboard.is_pressed(args.tp_key):
                wps.append(Waypoint(round(px, 2), round(py, 2), action="teleport"))
                fps.append(grab_fp())
                if len(wps) == 1:
                    head_at_first = pose.heading_deg
                elif len(wps) == 2 and start_heading is None:
                    start_heading = pose.heading_deg
                print(f"  +точка #{len(wps)} (ТЕЛЕПОРТ): ({px:.2f}, {py:.2f})  hdg={pose.heading_deg:.0f}")
                print("[record] точка-телепорт = конец маршрута → сохраняю и выхожу")
                break

            # ручная точка по клавише
            elif keyboard.is_pressed(args.key):
                wps.append(Waypoint(round(px, 2), round(py, 2)))
                fps.append(grab_fp())
                last_auto_xy = (px, py)
                if len(wps) == 1:
                    head_at_first = pose.heading_deg
                elif len(wps) == 2 and start_heading is None:
                    start_heading = pose.heading_deg     # курс вдоль 1-го сегмента
                print(f"  +точка #{len(wps)} (ручная): ({px:.2f}, {py:.2f})  hdg={pose.heading_deg:.0f}")
                time.sleep(0.3)            # антидребезг

            # авто-точка по пройденному пути
            elif args.auto > 0:
                if last_auto_xy is None:
                    last_auto_xy = (px, py)
                elif math.hypot(px - last_auto_xy[0], py - last_auto_xy[1]) >= args.auto:
                    wps.append(Waypoint(round(px, 2), round(py, 2)))
                    fps.append(grab_fp())
                    last_auto_xy = (px, py)
                    if len(wps) == 1:
                        head_at_first = pose.heading_deg
                    elif len(wps) == 2 and start_heading is None:
                        start_heading = pose.heading_deg
                    print(f"  +точка #{len(wps)} (авто): ({px:.2f}, {py:.2f})  hdg={pose.heading_deg:.0f}")

            if keyboard.is_pressed(args.stop_key):
                break
            time.sleep(0.02)
    finally:
        tracker.close()

    sh = start_heading if start_heading is not None else head_at_first
    route = Route(name=args.name,
                  minimap_meters_per_px=cfg.minimap.minimap_meters_per_px,
                  waypoints=wps,
                  start_heading=sh)
    save_route(route, args.out)
    sh_str = f"{sh:.1f}°" if sh is not None else "нет"
    print(f"[record] сохранено {len(wps)} точек ({args.mode}), start_heading={sh_str} -> {args.out}")
    # отпечатки миникарты для абсолютной локализации (teach-and-repeat)
    if any(f is not None for f in fps):
        fp_out = fp_path_for(args.out)
        n_ok = save_fingerprints(fp_out, fps)
        print(f"[record] отпечатки миникарты: {n_ok}/{len(fps)} -> {fp_out}")
    else:
        print("[record] отпечатки миникарты не сохранены (нет cv2/кадров)")
    if sh is None or abs(sh) < 0.5:
        print("[record] ⚠ ВНИМАНИЕ: start_heading отсутствует/≈0 — курс при записи не "
              "считался (стрелка не найдена?). Сегмент поедет НЕ ТУДА. Перезапиши: "
              "убедись, что миникарта видна и персонаж двигается с первых метров.")


if __name__ == "__main__":
    main()
