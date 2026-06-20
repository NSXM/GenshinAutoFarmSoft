"""
Калибровка чувствительности мыши: camera.deg_per_mouse_unit
(сколько градусов поворота камеры приходится на 1 единицу сырого мышемува).

Как меряем: персонаж СТОИТ, двигаем мышь на известное dx и смотрим, насколько
повернулась КАМЕРА. Направление камеры берём по ГОЛУБОМУ КОНУСУ на миникарте
(MinimapReader.read().arrow_heading_deg) — он крутится вместе с камерой даже на
север-залоченной миникарте Genshin. (Старый метод через вращение текстуры
миникарты тут не работает: карта север-up и при повороте камеры не вращается.)

Чувствительность в Genshin нелинейна (ускорение мыши), поэтому пробуем несколько
величин мышемува и печатаем таблицу град/ед. + медиану. Значение СО ЗНАКОМ:
+ означает, что мышь вправо поворачивает камеру по часовой (курс растёт).

Запуск (в открытом мире, персонаж СТОИТ, окно игры активно):
    .venv\\Scripts\\python.exe scripts\\calibrate_mouse.py --write
"""
from __future__ import annotations

import argparse
import math
import statistics
import time

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genshin_nav.config import Config, update_config_value
from genshin_nav.capture.screen_capture import ScreenCapture
from genshin_nav.minimap.minimap_reader import MinimapReader
from genshin_nav.control.input_sim import InputSimulator
from genshin_nav.utils.geom import angle_diff_deg


def read_cam_dir(cap, mm, n=5):
    """Медианное (круговое) направление камеры по конусу, из n кадров. None если не вышло."""
    vs = []
    for _ in range(n):
        frame = cap.grab()
        if frame is None:
            time.sleep(0.003)
            continue
        r = mm.read(frame)
        if r.arrow_heading_deg is not None:
            vs.append(math.radians(r.arrow_heading_deg))
        time.sleep(0.01)
    if not vs:
        return None
    # круговое среднее
    s = sum(math.sin(a) for a in vs)
    c = sum(math.cos(a) for a in vs)
    if s == 0 and c == 0:
        return None
    return (math.degrees(math.atan2(s, c)) + 360.0) % 360.0


def sweep(cap, mm, inp, step, n_steps, settle=0.18):
    """
    Плавный свип камеры в ОДНУ сторону мелкими шагами по `step` единиц. После
    каждого шага читаем направление конуса и копим скорость поворота d/step.
    Возвращаем список (град/ед.) по шагам — медиана устойчива к грубым промахам
    детекта конуса. В конце возвращаем камеру назад.
    """
    rates = []
    prev = read_cam_dir(cap, mm, n=3)
    moved = 0
    for _ in range(n_steps):
        inp.move_mouse_raw(step, 0)
        moved += step
        time.sleep(settle)
        h = read_cam_dir(cap, mm, n=3)
        if h is not None and prev is not None:
            d = angle_diff_deg(h, prev)        # поворот за этот шаг, -180..180
            if abs(d) < 120.0:                 # явный промах конуса отсекаем
                rates.append(d / step)
        prev = h
    inp.move_mouse_raw(-moved, 0)              # вернуть камеру на место
    time.sleep(settle)
    return rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--write", action="store_true",
                    help="записать camera.deg_per_mouse_unit в config.yaml")
    ap.add_argument("--step", type=int, default=60, help="шаг мышемува за итерацию свипа")
    ap.add_argument("--steps", type=int, default=12, help="сколько шагов в свипе")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cap = ScreenCapture(cfg.capture.region, cfg.capture.backend,
                        cfg.capture.target_fps, cfg.capture.window_title)
    mm = MinimapReader(cfg.minimap)
    inp = InputSimulator(cfg.control, cfg.camera)   # реальный ввод — нужен поворот

    print("[calib] через 3с плавно проверну камеру свипом. Персонаж СТОИТ, окно игры активно.")
    time.sleep(3)

    try:
        rates = sweep(cap, mm, inp, args.step, args.steps)
    finally:
        inp.release_all()
        cap.close()

    if len(rates) < 3:
        print("[calib] мало валидных шагов — конус плохо виден. Проверь scripts/debug_view.py "
              "(конус на миникарте) и что окно игры активно.")
        return

    rec = statistics.median(rates)
    lo, hi = min(rates), max(rates)
    sign = "вправо=по часовой (норма)" if rec > 0 else "мышь вправо крутит ПРОТИВ часовой (учтём знаком)"
    print(f"[calib] шагов учтено: {len(rates)}  | град/ед. по шагам: "
          f"медиана={rec:+.4f}  разброс [{lo:+.4f}..{hi:+.4f}]")
    print(f"[calib] рекомендованное deg_per_mouse_unit = {rec:+.4f}   [{sign}]")
    print("        Точность некритична — поворот в исполнителе замкнут по курсу и сам добирает.")

    if args.write:
        ok = update_config_value(args.config, "camera.deg_per_mouse_unit", round(rec, 4))
        print(f"[calib] {'записано' if ok else 'НЕ найден ключ'} camera.deg_per_mouse_unit в {args.config}")


if __name__ == "__main__":
    main()
