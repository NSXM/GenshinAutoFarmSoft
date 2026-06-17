"""
Интерактивный пикер прямоугольных регионов под своё разрешение.

Захватывает один кадр экрана и даёт выделить мышкой регион (миникарты или окна
игры). Печатает [left, top, width, height] готовой строкой для config.yaml и, по
флагу --write, точечно вписывает значение в нужный ключ (комментарии конфига
сохраняются).

Примеры:
    python scripts/pick_region.py                       # просто показать координаты
    python scripts/pick_region.py --write minimap.region
    python scripts/pick_region.py --write capture.region

Управление в окне выбора: выделить рамку мышью, ENTER/ПРОБЕЛ — подтвердить,
C — отмена.
"""
from __future__ import annotations

import argparse
import sys

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genshin_nav.config import Config, update_config_value
from genshin_nav.capture.screen_capture import ScreenCapture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--write", default=None,
                    help="ключ для записи: minimap.region | capture.region")
    ap.add_argument("--full-screen", action="store_true",
                    help="захватить весь экран (а не регион окна из конфига)")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    region = None if args.full_screen else cfg.capture.region
    cap = ScreenCapture(region, cfg.capture.backend,
                        cfg.capture.target_fps, cfg.capture.window_title)
    print("[pick] захватываю кадр... (переключись в игру, если нужно)")
    frame = cap.grab_blocking()
    cap.close()

    # selectROI ждёт BGR-изображение
    win = "Выдели регион: ENTER/ПРОБЕЛ=ок, C=отмена"
    x, y, w, h = cv2.selectROI(win, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        print("[pick] отменено")
        sys.exit(0)

    region = [int(x), int(y), int(w), int(h)]
    print(f"[pick] выбран регион: {region}")
    print(f"       строка для config.yaml:  region: {region}")

    if args.write:
        ok = update_config_value(args.config, args.write, region)
        if ok:
            print(f"[pick] записано в {args.config}: {args.write} = {region}")
        else:
            print(f"[pick] НЕ нашёл ключ {args.write} в {args.config} — впиши вручную")


if __name__ == "__main__":
    main()
