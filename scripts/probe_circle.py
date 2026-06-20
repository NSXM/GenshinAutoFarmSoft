"""
Калибровочный дамп для формулы «сдвиг карты → курс».

Проблема: на прямой ходьбе (ОДНО направление) motion-курс совпадал со стрелкой, но
на поворотах уходил до 180° — значит формула (sx,sy)->курс не та (поворот/отражение).
Из одного направления её не определить. Нужен дамп в РАЗНЫХ направлениях.

Что логирует на каждый кадр (CSV diag/circle_dump.csv):
  arrow  — курс по стрелке игрока (read_heading) = ИСТИННЫЙ курс, когда валиден;
  sx,sy  — сдвиг текстуры карты за скользящее окно (ring-masked phaseCorrelate);
  dist   — длина сдвига за окно (для отбора кадров с реальным движением).

КАК СНИМАТЬ: беги, поворачивая камеру/персонажа так, чтобы пройти ВСЕ стороны
(север/восток/юг/запад) — проще всего описать круг или «восьмёрку» секунд за 20-30.
Стрелка должна детектиться (не на воде). Потом я по этому дампу подберу формулу.

Запуск:  .venv\\Scripts\\python.exe scripts\\probe_circle.py
Выход:   F9 или Ctrl+C
"""
from __future__ import annotations
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from genshin_nav.config import Config              # noqa: E402
from genshin_nav.capture.screen_capture import ScreenCapture  # noqa: E402
from genshin_nav.minimap.minimap_reader import MinimapReader  # noqa: E402

WIN = 6   # кадров в окне накопления сдвига


def main():
    cfg = Config.load(os.path.join(ROOT, "config.yaml"))
    l, t, w, h = cfg.minimap.region
    reader = MinimapReader(cfg.minimap)   # для read_heading (стрелка) и кольцевой маски

    cap = ScreenCapture(cfg.capture.region, cfg.capture.backend,
                        cfg.capture.target_fps, cfg.capture.window_title)
    try:
        import keyboard
        have_kb = True
    except Exception:
        have_kb = False

    csv = open(os.path.join(ROOT, "diag", "circle_dump.csv"), "w", encoding="utf-8")
    csv.write("t,arrow,sx,sy,dist\n")

    print("[circle] 3с на переключение. БЕГИ ПО КРУГУ/ВОСЬМЁРКЕ (все стороны), 20-30с. F9 — стоп.")
    time.sleep(3)
    t0 = time.monotonic()
    prev = None
    positions = deque(maxlen=WIN + 1)
    cur = np.array([0.0, 0.0])
    n = 0
    try:
        while True:
            frame = cap.grab()
            if frame is None:
                time.sleep(0.001)
                continue
            now = time.monotonic() - t0
            mm = frame[t:t + h, l:l + w]
            arrow = reader.read_heading(mm)            # истинный курс (когда валиден)

            gray = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY).astype(np.float32)
            g = gray * reader._motion_window(gray.shape)   # кольцевая маска (без центра/углов)
            if prev is not None and prev.shape == g.shape:
                (dx, dy), _ = cv2.phaseCorrelate(prev, g)
                cur = cur + np.array([dx, dy])
            prev = g
            positions.append(cur.copy())

            sx = sy = dist = 0.0
            if len(positions) >= 2:
                d = positions[-1] - positions[0]
                sx, sy = float(d[0]), float(d[1])
                dist = float(np.hypot(sx, sy))

            n += 1
            csv.write(f"{now:.2f},{'' if arrow is None else f'{arrow:.1f}'},{sx:.3f},{sy:.3f},{dist:.3f}\n")
            csv.flush()
            if n % 30 == 0:
                print(f"[circle] t={now:5.1f}s кадров={n} arrow={('--' if arrow is None else f'{arrow:5.1f}')} "
                      f"sx={sx:+6.2f} sy={sy:+6.2f} dist={dist:5.2f}")

            if have_kb and keyboard.is_pressed("f9"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.close()
        csv.close()
        print(f"\n[circle] кадров={n} -> diag/circle_dump.csv. Пришли его — подберу формулу курса.")


if __name__ == "__main__":
    main()
