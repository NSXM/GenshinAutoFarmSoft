"""
calibrate_click.py — узнать ЭКРАННЫЕ координаты точки для авто-клика.

Нужно для config.control.teleport_dismiss_xy — куда бот должен кликнуть ЛКМ,
чтобы закрыть всплывашку «Точка телепортации» (синий ромбик).

Как пользоваться:
  1. Запусти скрипт.
  2. В игре доведи дело до состояния, где видна всплывашка телепорта с ромбиком
     (или просто наведи курсор туда, куда надо будет кликать).
  3. Нажми F8 — скрипт напечатает текущие координаты курсора.
  4. Впиши их в config.yaml:  control.teleport_dismiss_xy: [X, Y]
  F9 — выход.

    .venv\\Scripts\\python.exe scripts\\calibrate_click.py
"""
from __future__ import annotations

import time


def main():
    try:
        import keyboard
    except Exception:
        print("Нужен модуль keyboard: pip install keyboard")
        return
    # позиция курсора: pydirectinput/pyautogui или ctypes (Windows)
    get_pos = None
    try:
        import pyautogui
        get_pos = pyautogui.position
    except Exception:
        try:
            import ctypes

            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

            def get_pos():
                p = POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(p))
                return (p.x, p.y)
        except Exception:
            print("Не удалось получить позицию курсора (нет pyautogui и не Windows)")
            return

    print("[calib] F8 — напечатать координаты курсора, F9 — выход")
    last = None
    while True:
        if keyboard.is_pressed("f8"):
            x, y = get_pos()
            print(f"  координаты курсора: [{x}, {y}]   -> config: teleport_dismiss_xy: [{x}, {y}]")
            time.sleep(0.4)
        if keyboard.is_pressed("f9"):
            break
        # лёгкий показ текущей позиции (не спамим)
        cur = get_pos()
        if cur != last:
            last = cur
        time.sleep(0.03)
    print("[calib] выход")


if __name__ == "__main__":
    main()
