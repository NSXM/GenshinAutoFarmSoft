"""
ЖИВОЙ оверлей гибридного детектора курса (heading_detector.HeadingDetector) через
НАШ захват экрана. Встроенный в модуль `python heading_detector.py 0` берёт ВЕБКУ —
игру не снимет, поэтому смотрим тут.

Окно показывает увеличенный кроп миникарты с тремя стрелками:
  ЗЕЛЁНАЯ  — курс по движению (motion, PRIMARY)
  ГОЛУБАЯ  — курс по стрелке игрока (arrow, FALLBACK)
  ОРАНЖЕВАЯ— итоговый EMA (что пойдёт в бота)
+ числа motion/arrow/heading/source/conf.

Прогони и походи/побегай по РАЗНЫМ местам (вода, трава, повороты, рядом с метками).
Смотри: оранжевая стрелка должна стабильно смотреть туда, куда реально идёшь, и НЕ
прыгать на воде/у пинов. Спринт даёт более чистый motion.

Запуск:  .venv\\Scripts\\python.exe scripts\\probe_heading.py
Выход:   F9 или Q в окне
"""
from __future__ import annotations
import math
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from genshin_nav.config import Config              # noqa: E402
from genshin_nav.capture.screen_capture import ScreenCapture  # noqa: E402
import heading_detector as hd                      # noqa: E402

WIN = "heading probe (motion=green arrow=cyan ema=orange)"


def arrow(img, cx, cy, R, deg, color, thick):
    a = math.radians(deg)
    cv2.arrowedLine(img, (cx, cy),
                    (int(cx + R * math.sin(a)), int(cy - R * math.cos(a))),
                    color, thick, tipLength=0.3)


def main():
    cfg = Config.load(os.path.join(ROOT, "config.yaml"))
    l, t, w, h = cfg.minimap.region
    # синхронизируем Cfg детектора с нашим регионом (на случай crop=True не нужен —
    # мы сами кропим и отдаём crop=False)
    det = hd.HeadingDetector(crop=False)

    cap = ScreenCapture(cfg.capture.region, cfg.capture.backend,
                        cfg.capture.target_fps, cfg.capture.window_title)
    try:
        import keyboard
        have_kb = True
    except Exception:
        have_kb = False

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 360, 480)
    try:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)   # поверх игры (borderless)
    except Exception:
        print("[probe] always-on-top не поддержан этой сборкой OpenCV")
    print("[probe] 3с на переключение в игру... ходи по разным местам. F9/Q — стоп.")
    print("[probe] окно поверх игры; игра ДОЛЖНА быть в режиме 'без рамки' (borderless)")
    time.sleep(3)
    n = 0
    positioned = False
    try:
        while True:
            frame = cap.grab()
            if frame is None:
                time.sleep(0.001)
                continue
            mm = frame[t:t + h, l:l + w]
            r = det.update(mm)
            n += 1

            vis = cv2.resize(mm, (360, 360), interpolation=cv2.INTER_NEAREST)
            cx, cy, R = 180, 180, 150
            if r.arrow_hdg is not None:
                arrow(vis, cx, cy, R, r.arrow_hdg, (255, 180, 0), 2)
            if r.motion_hdg is not None:
                arrow(vis, cx, cy, R, r.motion_hdg, (0, 255, 0), 3)
            arrow(vis, cx, cy, R - 20, r.heading, (0, 160, 255), 2)

            panel = np.full((120, 360, 3), 30, np.uint8)
            txt = [
                f"EMA hdg={r.heading:6.1f}  [{r.source}] conf={r.confidence:.2f}",
                f"motion={r.motion_hdg}   arrow={r.arrow_hdg}",
                f"delta=({r.delta_px[0]:+.2f},{r.delta_px[1]:+.2f})",
            ]
            for i, s in enumerate(txt):
                cv2.putText(panel, s, (8, 30 + i * 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (230, 230, 230), 1, cv2.LINE_AA)
            out = np.vstack([vis, panel])
            cv2.imshow(WIN, out)
            if not positioned:
                cv2.moveWindow(WIN, 1520, 80)        # правый-верхний угол, мимо миникарты
                positioned = True
            if n % 60 == 0:                          # переутверждаем topmost (игра перехватывает)
                try:
                    cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
                except Exception:
                    pass

            if n % 30 == 0:
                print(f"[probe] hdg={r.heading:6.1f} [{r.source:8s}] "
                      f"motion={r.motion_hdg} arrow={r.arrow_hdg} conf={r.confidence:.2f}")

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), ord('Q')):
                break
            if have_kb and keyboard.is_pressed("f9"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.close()
        cv2.destroyAllWindows()
        print(f"[probe] кадров={n}. Если оранжевая держала курс на воде/у меток — "
              f"вшиваем детектор в бота (PoseTracker).")


if __name__ == "__main__":
    main()
