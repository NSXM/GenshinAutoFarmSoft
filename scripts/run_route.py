"""
Запуск ЖИВОГО прохождения маршрута: бот сам едет по точкам (реальный ввод).

    .venv\\Scripts\\python.exe scripts\\run_route.py --route routes\\recorded_route.json
    .venv\\Scripts\\python.exe scripts\\run_route.py --route ... --dry-run   # без ввода, только лог

ЦЕПОЧКА сегментов (продолжение после телепорта, без перезаписи старого маршрута):
    ...run_route.py --route routes\\seg1.json routes\\seg2.json --hud
Бот пройдёт seg1 (с телепортом в конце), затем перепривяжется и поедет seg2.

ВНИМАНИЕ: в боевом режиме бот перехватывает мышь и клавиатуру (зажимает W, крутит
камеру). Запускай в БЕЗОПАСНОМ открытом месте. Аварийная остановка:
    F9  — стоп (отпустить клавиши),  либо Ctrl+C в консоли.
Окно игры должно быть активно (borderless), иначе ввод не дойдёт.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genshin_nav.config import Config
from genshin_nav.control.route import load_route
from genshin_nav.control.route_runner import RouteRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--route", nargs="+", default=["routes/recorded_route.json"],
                    help="один или НЕСКОЛЬКО маршрутов подряд (цепочка сегментов после телепорта)")
    ap.add_argument("--dry-run", action="store_true",
                    help="всё считать и логировать, но НЕ выполнять реальный ввод")
    ap.add_argument("--hud", action="store_true",
                    help="показать окно HUD поверх игры (карта маршрута + позиция/курс бота)")
    args = ap.parse_args()

    routes = []
    for rp in args.route:
        r = load_route(rp)
        if not r.waypoints:
            print(f"[run] маршрут пуст, пропускаю: {rp}")
            continue
        routes.append(r)
    if not routes:
        print("[run] нет непустых маршрутов")
        return

    cfg = Config.load(args.config)
    if args.dry_run:
        cfg.control.dry_run = True

    runner = RouteRunner(cfg, routes, hud=args.hud)

    try:
        import keyboard
        keyboard.add_hotkey("f9", lambda: (print("\n[run] F9 — АВАРИЙНАЯ ОСТАНОВКА"), runner.stop()))
    except Exception:
        print("[run] модуль keyboard недоступен; аварийная остановка только Ctrl+C")

    mode = "DRY-RUN (без ввода)" if cfg.control.dry_run else "БОЕВОЙ (ввод активен!)"
    total_pts = sum(len(r.waypoints) for r in routes)
    print(f"[run] режим: {mode}. Сегментов: {len(routes)}, точек всего: {total_pts}.")
    print("[run] 3 секунды на переключение в окно игры (открытое безопасное место)...")
    time.sleep(3)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n[run] прервано (Ctrl+C)")
        runner.stop()


if __name__ == "__main__":
    main()
