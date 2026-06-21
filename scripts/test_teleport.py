"""
test_teleport.py — проверить ТОЛЬКО телепорт-меню, без всего маршрута.

Запускает только часть с картой: M (открыть карту) → пауза → клик по жёлтому
ромбику → клик по иконке «Точка телепортации» слева → клик по кнопке «Телепорт».
Удобно подбирать пороги/тайминги/смещения кликов (config.control.teleport_*),
не гоняя весь маршрут.

ВНИМАНИЕ: выполняет РЕАЛЬНЫЕ клики мышью. Открой карту-меню в игре (или будь готов
к тому, что скрипт сам нажмёт M). Окно игры — borderless, активно.

Запуск:
    .venv\\Scripts\\python.exe scripts\\test_teleport.py
    .venv\\Scripts\\python.exe scripts\\test_teleport.py --with-f   # ещё и нажать F перед картой
    .venv\\Scripts\\python.exe scripts\\test_teleport.py --delay 5  # 5 секунд на переключение в игру
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genshin_nav.config import Config
from genshin_nav.control.route import Route, Waypoint
from genshin_nav.control.route_runner import RouteRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--with-f", action="store_true",
                    help="сначала нажать F (активировать точку), потом карту")
    ap.add_argument("--delay", type=float, default=3.0,
                    help="секунд на переключение в окно игры перед стартом")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    # фиктивный маршрут — RouteRunner нужен только ради захвата/ввода/шаблонов
    dummy = Route(name="tp-test", waypoints=[Waypoint(0.0, 0.0), Waypoint(0.0, 1.0)])
    runner = RouteRunner(cfg, dummy, hud=False)

    try:
        import keyboard
        keyboard.add_hotkey("f9", lambda: (print("\n[test] F9 — СТОП"), runner.stop()))
    except Exception:
        pass

    print(f"[test] РЕАЛЬНЫЕ клики! {args.delay:.0f} сек на переключение в игру "
          f"(встань на точку телепорта{' — F нажму сам' if args.with_f else ''})...")
    time.sleep(args.delay)
    try:
        if args.with_f:
            # F → пауза → карта и клики
            c = cfg.control
            print("[test] нажимаю F (активировать точку)")
            runner.inp.tap(getattr(c, "teleport_activate_key", "f"), 0.06)
            time.sleep(getattr(c, "teleport_wait_menu_s", 1.5))
        runner._map_teleport()
        print("[test] ГОТОВО — смотри лог выше: какие клики прошли (score) / не нашлись")
    except KeyboardInterrupt:
        print("\n[test] прервано")
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
