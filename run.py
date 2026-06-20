"""
Точка входа: запуск автономной навигации по маршруту.

Использование:
    python run.py --config config.yaml --route routes/example_route.json

ВНИМАНИЕ: запускать при свёрнутой обычной активности — бот перехватывает мышь и
клавиатуру. Аварийная остановка — Ctrl+C в консоли (или горячая клавиша ниже).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading

from genshin_nav.config import Config
from genshin_nav.control.navigator import Navigator, Waypoint


def load_route(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Waypoint(float(p["x"]), float(p["y"])) for p in data["waypoints"]]


def install_panic_key(nav: Navigator):
    """Аварийная остановка по F12 (если доступен keyboard)."""
    try:
        import keyboard
        keyboard.add_hotkey("f12", lambda: (print("\n[panic] F12 — стоп"), nav.stop()))
    except Exception:
        print("[panic] модуль keyboard недоступен; аварийная остановка — Ctrl+C")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--route", default="routes/example_route.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="всё считать и логировать, но НЕ выполнять реальный ввод")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.dry_run:
        cfg.control.dry_run = True
    route = load_route(args.route)
    if not route:
        print("Маршрут пуст."); sys.exit(1)

    nav = Navigator(cfg, route)
    install_panic_key(nav)
    mode = "DRY-RUN (без ввода)" if cfg.control.dry_run else "БОЕВОЙ (ввод активен)"
    print(f"[run] режим: {mode}")
    print(f"[run] старт. Точек маршрута: {len(route)}. Бэкенд захвата: {cfg.capture.backend}")
    print("[run] 3 секунды на переключение в окно игры...")
    import time; time.sleep(3)
    try:
        nav.run()
    except KeyboardInterrupt:
        print("\n[run] прервано пользователем")
        nav.stop()


if __name__ == "__main__":
    main()
