"""
Диагностика: сохранить серию кропов миникарты, пока ты идёшь в ОДНУ сторону.
По этой серии видно, как реально двигается карта (и не мешают ли углы/HUD).

Запуск:
    .venv\\Scripts\\python.exe scripts\\dump_minimap.py

Что делать: запусти, переключись в игру и ИДИ РОВНО ВПЕРЁД (зажми W) всё время
записи (~6 секунд). Лучше по прямой, не крутя камеру. После — пришли папку diag\\.
"""
from __future__ import annotations

import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genshin_nav.config import Config
from genshin_nav.capture.screen_capture import ScreenCapture


def main():
    cfg = Config.load("config.yaml")
    cap = ScreenCapture(cfg.capture.region, cfg.capture.backend,
                        cfg.capture.target_fps, cfg.capture.window_title)
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diag")
    os.makedirs(outdir, exist_ok=True)

    print("[dump] 3 секунды на переключение в игру, потом ИДИ РОВНО ВПЕРЁД (W)...")
    time.sleep(3)

    # один полный кадр — чтобы проверить регион миникарты на фоне всего экрана
    full = cap.grab_blocking()
    cv2.imwrite(os.path.join(outdir, "full_frame.png"), full)

    l, t, w, h = cfg.minimap.region
    n = 18
    print(f"[dump] записываю {n} кропов миникарты по ~0.3с. ИДИ ВПЕРЁД!")
    for i in range(n):
        frame = cap.grab_blocking()
        mm = frame[t:t + h, l:l + w]
        cv2.imwrite(os.path.join(outdir, f"mm_{i:02d}.png"), mm)
        time.sleep(0.30)
    cap.close()
    print(f"[dump] готово -> {outdir}")
    print("[dump] пришли папку diag\\ (или хотя бы mm_00.png, mm_09.png, mm_17.png и full_frame.png)")


if __name__ == "__main__":
    main()
