"""
Живой исполнитель маршрута — ведёт персонажа по точкам РЕАЛЬНЫМ вводом.

Схема (замкнутый контур, без зрения):
  * Бот зажимает W и едет вперёд. КУРС берём из сдвига карты — он надёжен ИМЕННО
    в движении (конус нужен только когда стоишь, а мы едем).
  * Каждый такт: ошибка = пеленг на точку − курс. Подруливаем мышью к её уменьшению.
  * Знак и масштаб руления НЕ калибруем заранее: на старте делаем пробный доворот в
    движении и меряем, на сколько изменился курс (units_per_deg = K/Δкурс, СО ЗНАКОМ).
    Поэтому точная калибровка мыши не нужна, а направление мыши определяется само.
  * Мини-защита от застревания: если позиция не меняется при зажатой W — прыжок+доворот.

Обход препятствий/монстров (CV, шаги 2–5 плана) — СЛЕДУЮЩАЯ фаза, здесь его нет.
Аварийная остановка — снаружи (раннер вешает F9) и через stop().
"""
from __future__ import annotations

import math
import time
from typing import Optional

from .pose_tracker import PoseTracker
from .route_follower import RouteFollower
from .input_sim import InputSimulator
from ..utils.geom import angle_diff_deg


class RouteRunner:
    def __init__(self, cfg, route):
        self.cfg = cfg
        self.route = route
        self.tracker = PoseTracker(cfg)
        self.follower = RouteFollower(route, cfg.control)
        self.inp = InputSimulator(cfg.control, cfg.camera, dry_run=cfg.control.dry_run)

        c = cfg.control
        self.deadzone = c.turn_deadzone_deg
        self.gain = c.turn_gain
        self.throttle = c.turn_throttle_s
        self.max_turn = c.turn_max_units
        # крупнее этого угла — доворачиваем СТОЯ (камера слушается на 100%, без
        # авто-доворота игры и без пируэта), затем едем дальше. Мелкие правки — на ходу.
        self.align_big_deg = 40.0
        self.max_pivot_units = 700      # потолок ед. мыши за разовый доворот стоя

        # Фиксированное усиление руля: ~0.16°/ед (≈6.25 ед/°) — многократно измерено.
        # Per-run калибровку убрали (шумела и угоняла бота). Знак уточняет auto-flip.
        self.units_per_deg: float = 1.0 / max(0.02, cfg.camera.deg_per_mouse_unit)
        self._running = False
        self._last_steer_t = 0.0
        self._last_dx = 0.0              # последняя поданная команда руля (для трейса)
        self._stuck_ref = None
        self._stuck_t = 0.0
        # детект разноса (неверный знак руления) -> авто-инверсия
        self._diverge_since: Optional[float] = None
        self._diverge_minabs = 0.0
        self._flips = 0
        # ВЫКЛ авто-инверсия знака. Знак руля = физ. константа (units_per_deg>0 верен,
        # подтверждено трейсом: dx<0 уменьшает курс). Глюк детекта курса (стрелка
        # прыгает на 100°+) задирал |err| и провоцировал СПУРЬЕЗНЫЙ флип знака посреди
        # прогона → положительный dx гнал курс не туда → runaway-разворот (виден в
        # хвосте трейса). Вреда от авто-флипа больше, чем пользы.
        self._sign_locked = True

    # ---- курс: круговое среднее за secs (надёжен только в движении) ----------
    def _avg_heading(self, secs: float, require_moving: bool = False) -> Optional[float]:
        """Круговое среднее курса за secs. require_moving=True учитывает только кадры
        с надёжным курсом (move=Y) — иначе среднее засоряется конусом при остановке."""
        s = c = 0.0
        n = 0
        t_end = time.monotonic() + secs
        while time.monotonic() < t_end:
            p = self.tracker.poll()
            if p is None:
                time.sleep(0.002)
                continue
            if require_moving and not p.moving:
                continue
            a = math.radians(p.heading_deg)
            s += math.sin(a)
            c += math.cos(a)
            n += 1
        if n < 2:
            return None
        return (math.degrees(math.atan2(s, c)) + 360.0) % 360.0

    # ---- авто-калибровка руления пробным доворотом В ДВИЖЕНИИ ----------------
    def _calibrate_turn(self) -> bool:
        k = 150
        for _ in range(4):
            h0 = self._avg_heading(0.35, require_moving=True)
            if h0 is None:
                continue
            self.inp.move_mouse_raw(k, 0)
            time.sleep(0.35)               # дать персонажу снова поехать в новом направлении
            h1 = self._avg_heading(0.35, require_moving=True)
            if h1 is None:
                continue
            dh = angle_diff_deg(h1, h0)
            if abs(dh) >= 8.0:
                self.units_per_deg = k / dh         # со знаком: dx = err * units_per_deg
                print(f"[run] калибровка руления: +{k}ед -> Δкурс {dh:+.1f}° "
                      f"=> {self.units_per_deg:+.2f} ед/°")
                return True
            k = int(k * 1.7)                        # повернуло мало — толкаем сильнее
        dpu = self.cfg.camera.deg_per_mouse_unit or 0.15
        self.units_per_deg = 1.0 / dpu
        print(f"[run] авто-калибровка не удалась — фолбэк {self.units_per_deg:+.2f} ед/° "
              f"(знак может быть неверным, проверь по поведению)")
        return False

    def _pivot(self, err_deg: float):
        """Крупный доворот НА ХОДУ (W НЕ отпускаем). Раньше доворачивали стоя — но
        стоя мышь крутит только КАМЕРУ, а персонаж (и стрелка-курс) за ней не идёт,
        поэтому контур молотил вхолостую на месте (видно в трейсе: ~2.7с болтанки при
        неизменной позиции). Дамп подтвердил: стрелка достоверна именно В ДВИЖЕНИИ.
        Поэтому держим W и крутим камеру до 0.8·err порциями — персонаж едет по дуге,
        доворачивает, курс читается надёжно. Throttle не ждём (это разовый крупный
        доворот). Знак ошибётся — поймает auto-flip."""
        self.inp.start_moving()         # гарантированно едем (стрелка надёжна только в движении)
        dx = 0.8 * err_deg * self.units_per_deg
        dx = max(-self.max_pivot_units, min(self.max_pivot_units, dx))
        self._last_dx = float(dx)       # для трейса
        self._last_steer_t = time.monotonic()
        self.inp.move_mouse_raw(int(round(dx)), 0)

    def _steer(self, err_deg: float, now: float):
        if self.units_per_deg is None:
            return
        # МЯГКАЯ зона нечувствительности: вычитаем deadzone из модуля ошибки. У края
        # зоны команда ≈0 и плавно растёт — нет «чанковых» поворотов (жёсткая зона
        # давала минимум ~10° за раз). Внутри зоны (mag<=0) руль молчит → едем прямо.
        mag = abs(err_deg) - self.deadzone
        if mag <= 0:
            return
        if now - self._last_steer_t < self.throttle:    # не чаще лага курса
            return
        self._last_steer_t = now
        eff = math.copysign(mag, err_deg)                # ошибка за вычетом зоны
        dx = eff * self.gain * self.units_per_deg
        dx = max(-self.max_turn, min(self.max_turn, dx))
        self._last_dx = float(dx)                        # для трейса
        self.inp.move_mouse_raw(int(round(dx)), 0)

    def _auto_sign(self, err_deg: float, now: float):
        """
        Если знак руления неверен, контур разносит: ошибка большая и НЕ падает.
        Следим: при |err|>60° запоминаем лучшую (минимальную) достигнутую |err|;
        если за 1.5с она не улучшилась, а стала ещё хуже на 15° — руль наоборот,
        инвертируем знак. На сходящемся (правильном) контуре |err| быстро падает
        ниже 60° и таймер сбрасывается, так что ложных инверсий нет.
        """
        if self.units_per_deg is None or self._sign_locked:
            return                                # знак уже установлен пробой
        if abs(err_deg) <= 60.0:
            self._diverge_since = None
            return
        if self._diverge_since is None:
            self._diverge_since = now
            self._diverge_minabs = abs(err_deg)
            return
        self._diverge_minabs = min(self._diverge_minabs, abs(err_deg))
        if now - self._diverge_since > 1.2 and abs(err_deg) > self._diverge_minabs + 12.0:
            if self._flips < 3:
                self.units_per_deg = -self.units_per_deg
                self._flips += 1
                print(f"[run] контур разносит -> инвертирую знак руления (flip #{self._flips}, "
                      f"теперь {self.units_per_deg:+.2f} ед/°)")
            self._diverge_since = None

    def _stuck(self, now: float, pos) -> bool:
        win = self.cfg.control.stuck_window_s
        mind = self.cfg.control.stuck_min_displacement_m
        if self._stuck_ref is None:
            self._stuck_ref, self._stuck_t = pos, now
            return False
        if math.hypot(pos[0] - self._stuck_ref[0], pos[1] - self._stuck_ref[1]) >= mind:
            self._stuck_ref, self._stuck_t = pos, now
            return False
        if now - self._stuck_t >= win:
            self._stuck_ref, self._stuck_t = pos, now
            return True
        return False

    def _start_bearing(self) -> Optional[float]:
        """Пеленг на цель ~lookahead впереди от стартовой точки (та же конвенция, что
        в follower: 0=север, восток=-x). Для стартовой калибровки смещения курса."""
        wps = self.route.waypoints
        if len(wps) < 2:
            return None
        p0 = wps[0]
        look = getattr(self.cfg.control, "lookahead_m", 6.0)
        tgt = wps[-1]
        for wp in wps[1:]:
            if math.hypot(wp.x - p0.x, wp.y - p0.y) >= look:
                tgt = wp
                break
        dx, dy = tgt.x - p0.x, tgt.y - p0.y
        return (math.degrees(math.atan2(-dx, dy)) + 360.0) % 360.0

    # ---- главный цикл --------------------------------------------------------
    def run(self):
        self._running = True
        loop_dt = 1.0 / 30.0
        # выравниваем одометрию с маршрутом: считаем, что стартуем в первой точке
        # DEAD-RECKONING позиции: карта-одометрия на этом зуме слепая (сдвиг суб-пиксельный),
        # позицию ведём интегрированием курс×скорость (бот держит W → едет вперёд).
        self._dr_pos = [0.0, 0.0]
        self._dr_t = None
        if self.route.waypoints:
            wp0 = self.route.waypoints[0]
            self.tracker.set_position(wp0.x, wp0.y)
            self._dr_pos = [wp0.x, wp0.y]
            print(f"[run] старт-позиция = точка #1 ({wp0.x:.1f},{wp0.y:.1f}); "
                  f"встань в начале маршрута, ЛИЦОМ вдоль маршрута!")
        # Знак руля = ФИЗИЧЕСКАЯ КОНСТАНТА игры, не меняется между запусками. Он уже
        # верный из конфига (units_per_deg > 0, подтверждено рабочим запуском). Живую
        # пробу УБРАЛИ: она читала курс в момент смаза стрелки (резкий поворот камеры)
        # и выдавала случайный знак, ломая верную константу. auto-flip остался как
        # страховка на случай, если знак в конфиге однажды окажется неверным.
        print(f"[run] усиление руля {self.units_per_deg:+.1f} ед/° (знак из конфига)")

        # --- КАЛИБРОВКА СМЕЩЕНИЯ КУРСА (стоя, до движения) ---
        # Система стрелки (курс) и система пеленга/одометрии имеют ПОСТОЯННЫЙ сдвиг
        # (~140°, виден как стабильный стартовый err во всех прогонах). Считаем, что
        # на старте стоишь ЛИЦОМ вдоль маршрута → истинный курс = пеленг на первую цель.
        # Запоминаем разницу и вычитаем из всех курсов. Тогда прямой маршрут = err≈0.
        self._hdg_offset = 0.0
        brg0 = self._start_bearing()
        hs = self._avg_heading(0.8)     # стабильный курс стоя (медиана-фильтр уже внутри)
        if hs is not None and brg0 is not None:
            self._hdg_offset = angle_diff_deg(hs, brg0)
            print(f"[run] калибровка курса: курс стоя {hs:.0f}°, пеленг на цель {brg0:.0f}° "
                  f"→ смещение {self._hdg_offset:+.0f}° (предполагаю старт ЛИЦОМ вдоль маршрута!)")
        else:
            print("[run] калибровка курса не удалась (нет курса/пеленга) — смещение 0")

        self.inp.start_moving()         # теперь едем
        print("[run] начинаю вести")
        last_log = 0.0
        # покадровый CSV-трейс для диагностики качания (см. scripts/analyze_trace.py)
        import os as _os
        _trace_path = _os.path.join(_os.path.dirname(_os.path.dirname(
            _os.path.dirname(_os.path.abspath(__file__)))), "diag", "run_trace.csv")
        _os.makedirs(_os.path.dirname(_trace_path), exist_ok=True)
        trace = open(_trace_path, "w", encoding="utf-8")
        trace.write("t,px,py,hdg,brg,err,move,dx,action\n")
        t_start = time.monotonic()
        try:
            while self._running:
                t0 = time.monotonic()
                pose = self.tracker.poll()
                if pose is None:
                    time.sleep(0.001)
                    continue
                hdg = (pose.heading_deg - self._hdg_offset) % 360.0   # курс со стартовой калибровкой

                # DEAD-RECKONING позиции из курса×скорость (пока держим W). Курс hdg уже
                # в системе пеленга (восток=-x): смещение на курс θ = (-sinθ, cosθ),
                # его пеленг atan2(-dx,dy)=θ — самосогласовано с follower.
                if self._dr_t is not None and self.cfg.control.move_key in self.inp._held:
                    dt = min(0.2, max(0.0, t0 - self._dr_t))
                    spd = self.cfg.control.dead_reckon_speed
                    rad = math.radians(hdg)
                    self._dr_pos[0] += -math.sin(rad) * spd * dt
                    self._dr_pos[1] += math.cos(rad) * spd * dt
                self._dr_t = t0

                d = self.follower.step((self._dr_pos[0], self._dr_pos[1]), hdg)
                if d.done:
                    print("[run] МАРШРУТ ПРОЙДЕН")
                    break

                self.inp.start_moving()            # держим W всегда (кроме краткого пивота)

                # Рулим НЕПРЕРЫВНО: курс берём со СТРЕЛКИ игрока (надёжна всегда,
                # кругСКО 0.1° по дампу), поэтому ждать move=Y (детект сдвига карты,
                # срабатывает лишь ~8% кадров на этом зуме) больше НЕ нужно — раньше
                # именно это сводило руль к редким шумным правкам и давало качание.
                err = d.heading_err_deg
                self._last_dx = 0.0                # сбрасываем; _steer выставит если рулил
                self._auto_sign(err, t0)
                # ОДИН throttled пропорциональный регулятор на ходу. Отдельный «пивот»
                # на большой угол убрали: крупная разовая команда перелетает (доворот
                # персонажа отстаёт от камеры на кадры → курс проскакивает цель → знак
                # ошибки меняется → раскачка). Большой промах закрываем серией мелких
                # шагов с обратной связью между ними — медленнее, но без перелёта.
                act = "steer"
                self._steer(err, t0)

                trace.write(f"{t0 - t_start:.3f},{self._dr_pos[0]:.2f},"
                            f"{self._dr_pos[1]:.2f},{hdg:.1f},"
                            f"{d.bearing_deg:.1f},{d.heading_err_deg:+.1f},"
                            f"{1 if pose.moving else 0},{self._last_dx:+.0f},{act}\n")
                trace.flush()

                if t0 - last_log >= 0.4:
                    last_log = t0
                    print(f"[run] wp#{d.wp_idx} d={d.dist_m:5.1f}м hdg={pose.heading_deg:5.0f} "
                          f"brg={d.bearing_deg:5.0f} err={d.heading_err_deg:+5.0f} "
                          f"move={'Y' if pose.moving else 'n'}")

                slack = loop_dt - (time.monotonic() - t0)
                if slack > 0:
                    time.sleep(slack)
        finally:
            try:
                trace.close()
                print(f"[run] трейс записан: {_trace_path}")
            except Exception:
                pass
            self.stop()

    def stop(self):
        self._running = False
        try:
            self.inp.release_all()
        except Exception:
            pass
        try:
            self.tracker.close()
        except Exception:
            pass
