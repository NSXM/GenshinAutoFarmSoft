"""
Симуляция ввода в DirectX-игру.

Genshin (как и большинство DirectX-игр) часто игнорирует обычные эмуляторы
вроде pyautogui/keyboard, которые шлют WM-сообщения. Нужен SendInput на уровне
скан-кодов — это умеет pydirectinput. Поворот камеры — относительным движением
мыши (mouseMove с relative=True).

Если pydirectinput недоступен — мягкий фолбэк на pynput (может не сработать в
некоторых играх, но не падает при импорте).
"""
from __future__ import annotations

import time
from typing import Optional


class InputSimulator:
    def __init__(self, cfg_control, cfg_camera, dry_run: bool = False):
        self.cfg = cfg_control
        self.cam = cfg_camera
        # dry-run: бэкенд-вызовы не выполняются, но состояние _held и логи живут,
        # чтобы навигатор и stuck_detector работали как обычно.
        self.dry_run = dry_run or getattr(cfg_control, "dry_run", False)
        self._held: set = set()
        self._backend = None
        self._init_backend(cfg_control.input_backend)
        if self.dry_run:
            print("[input] DRY-RUN: реальный ввод отключён, только логи")

    def _init_backend(self, name: str):
        if name == "pydirectinput":
            try:
                import pydirectinput as pdi
                pdi.PAUSE = 0.0
                pdi.FAILSAFE = False
                self._pdi = pdi
                self._backend = "pydirectinput"
                return
            except Exception as e:
                print(f"[input] pydirectinput недоступен ({e}), пробую pynput")
        try:
            from pynput.keyboard import Controller as KB, Key
            from pynput.mouse import Controller as MS
            self._kb = KB(); self._ms = MS(); self._Key = Key
            self._backend = "pynput"
        except Exception as e:
            print(f"[input] нет бэкенда ввода ({e}); режим dry-run")
            self._backend = "dryrun"

    # ---- клавиши -----------------------------------------------------------
    def key_down(self, key: str):
        if key in self._held:
            return
        self._held.add(key)
        if self.dry_run:
            print(f"[dry] key_down {key}")
            return
        if self._backend == "pydirectinput":
            self._pdi.keyDown(key)
        elif self._backend == "pynput":
            self._kb.press(self._map_key(key))

    def key_up(self, key: str):
        if key not in self._held:
            return
        self._held.discard(key)
        if self.dry_run:
            print(f"[dry] key_up {key}")
            return
        if self._backend == "pydirectinput":
            self._pdi.keyUp(key)
        elif self._backend == "pynput":
            self._kb.release(self._map_key(key))

    def tap(self, key: str, hold_s: float = 0.05):
        self.key_down(key)
        time.sleep(hold_s)
        self.key_up(key)

    def _map_key(self, key: str):
        if key == "space":
            return self._Key.space
        if key == "shift":
            return self._Key.shift
        return key

    # ---- движение ----------------------------------------------------------
    def start_moving(self):
        self.key_down(self.cfg.move_key)

    def stop_moving(self):
        self.key_up(self.cfg.move_key)

    def jump(self):
        self.tap(self.cfg.jump_key, 0.06)

    def sprint_tap(self):
        self.tap(self.cfg.sprint_key, 0.05)

    # ---- камера ------------------------------------------------------------
    def move_mouse_raw(self, dx: int, dy: int = 0):
        """
        Сырой относительный мышемув в ЕДИНИЦАХ ввода (без пересчёта в градусы).
        Нужен для калибровки (мышемув -> измеренный Δyaw). Дробим на мелкие шаги —
        игра плавнее реагирует и меньше теряет события.
        """
        dx = int(round(dx)); dy = int(round(dy))
        if dx == 0 and dy == 0:
            return
        if self.dry_run:
            print(f"[dry] move_mouse_raw dx={dx} dy={dy}")
            return
        if self._backend == "pydirectinput":
            step = 40 if dx >= 0 else -40
            rem = dx
            while abs(rem) > abs(step):
                self._pdi.moveRel(step, 0, relative=True)
                rem -= step
                time.sleep(0.005)
            self._pdi.moveRel(rem, dy, relative=True)
        elif self._backend == "pynput":
            self._ms.move(dx, dy)

    # ВРЕМЕННО для диагностики: yaw_sign читается из cfg.camera, если там есть
    # поле (см. rotate_camera), а если нет — берётся эта константа класса.
    # Поменяй на -1.0 здесь напрямую и перезапусти — это гарантированно
    # сработает независимо от того, как config.py парсит yaml.
    YAW_SIGN_FALLBACK = 1.0

    def rotate_camera(self, dyaw_deg: float):
        """Повернуть камеру по горизонтали на dyaw градусов (через сырой мышемув).

        ВАЖНО про знак: dyaw_deg приходит из навигатора в компасной системе
        координат (0=север, по часовой; положительный err = цель ПРАВЕЕ
        текущего курса -> камеру нужно повернуть вправо/по часовой).
        Если на твоей системе/в игре ось мыши инвертирована, это лечится
        ЗДЕСЬ, в одном месте — через cam.yaw_sign в config.yaml ИЛИ,
        если конфиг это поле не подхватывает, через YAW_SIGN_FALLBACK выше.
        Не лезь чинить знак в navigator.py/minimap_reader.py — там система
        курса используется ещё в нескольких формулах (bearing, AVOID), и
        инверсия в одном из них собьёт остальные. Единственная точка правды
        "градусы курса -> физическое движение мыши" — это функция ниже.
        """
        if self.dry_run:
            # без печати: навигатор зовёт это каждый кадр -> залило бы консоль
            return
        yaw_sign = getattr(self.cam, "yaw_sign", None)
        if yaw_sign is None:
            yaw_sign = self.YAW_SIGN_FALLBACK
        dx = int(round(yaw_sign * dyaw_deg / self.cam.deg_per_mouse_unit))
        # ВРЕМЕННАЯ ДИАГНОСТИКА — убрать после калибровки. Смотрим, что
        # реально приходит на вход (dyaw_deg) и что реально летит в мышь (dx),
        # и сверяем глазами с тем, куда крутится камера в игре в этот момент.
        print(f"[rotate_camera] dyaw_deg={dyaw_deg:+7.2f}  yaw_sign={yaw_sign:+.0f}  -> dx={dx:+5d}")
        self.move_mouse_raw(dx, 0)

    def release_all(self):
        for k in list(self._held):
            self.key_up(k)
