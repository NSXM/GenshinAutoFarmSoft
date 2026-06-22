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
import os
import time
from typing import Optional

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .pose_tracker import PoseTracker
from .route_follower import RouteFollower
from .input_sim import InputSimulator
from ..utils.geom import angle_diff_deg
from ..minimap.localizer import load_localizer, make_fingerprint


class RouteRunner:
    def __init__(self, cfg, route, hud: bool = False):
        self.cfg = cfg
        # route может быть одним Route или СПИСКОМ Route (цепочка сегментов —
        # продолжение после телепорта). Нормализуем в список.
        self.routes = list(route) if isinstance(route, (list, tuple)) else [route]
        self.route = self.routes[0]            # текущий сегмент (меняется в _setup_segment)
        self.tracker = PoseTracker(cfg)
        self.follower = RouteFollower(self.route, cfg.control)
        self.inp = InputSimulator(cfg.control, cfg.camera, dry_run=cfg.control.dry_run)
        # Визуальный HUD прохождения (окно поверх игры). Чисто отображение, на
        # вождение не влияет; включается флагом (run_route.py --hud).
        self.hud = None
        if hud:
            try:
                from .route_hud import RouteHUD
                self.hud = RouteHUD(cfg, self.route)
            except Exception as e:
                print(f"[run] HUD не запустился ({e}) — иду без него")

        # шаблоны для телепорта: подсказка «F», и элементы карты для фаст-тревела
        cc = cfg.control
        self._tp_template = (self._load_template(getattr(cc, "teleport_template_path", "Fteleport.png"))
                             if getattr(cc, "teleport_detect_template", False) else None)
        self._tpl_rombik = self._load_template(getattr(cc, "teleport_rombik_template", "Rombik.png"))
        self._tpl_point = self._load_template(getattr(cc, "teleport_point_template", "tockateleport.png"))
        self._tpl_confirm = self._load_template(getattr(cc, "teleport_confirm_template", "teleport.png"))
        self._localizer = None          # абсолютная локализация по миникарте (грузится в _setup_segment)
        self._loc_dbg = None

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

    def _drive(self):
        """Держать движение вперёд (W). Спринт (Shift) включается ОТДЕЛЬНО в цикле
        по величине ошибки курса (_set_sprint) — на крутых поворотах не спринтуем."""
        self.inp.start_moving()

    def _set_sprint(self, want: bool):
        """Включить/выключить спринт (Shift). При включении даётся свежий фронт
        нажатия (key_down только если не зажат) — это и есть «повторное нажатие
        Shift» при возврате к спринту после поворота/телепорта/нового сегмента."""
        if not getattr(self.cfg.control, "sprint", False):
            return
        k = self.cfg.control.sprint_key
        held = k in self.inp._held
        if want and not held:
            self.inp.key_down(k)
        elif not want and held:
            self.inp.key_up(k)

    def _load_template(self, path):
        """Загрузить шаблон (grayscale) от корня проекта. None при отсутствии cv2/файла."""
        if cv2 is None or not path:
            return None
        if not os.path.isabs(path):
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            path = os.path.join(root, path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[run] шаблон не загружен: {path}")
        else:
            print(f"[run] шаблон {os.path.basename(path)} загружен ({img.shape[1]}x{img.shape[0]})")
        return img

    def _capture_offset(self):
        """Смещение (left, top) региона захвата → перевод координат кадра в АБСОЛЮТНЫЕ
        экранные (для click_at). Если захват всего экрана — (0,0)."""
        reg = getattr(self.tracker.cap, "region", None)
        return (reg[0], reg[1]) if reg else (0, 0)

    def _tp_prompt_visible(self, frame) -> bool:
        """Видна ли на экране подсказка «F Точка телепортации» (template-match)."""
        if self._tp_template is None or frame is None or cv2 is None:
            return False
        try:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            th, tw = self._tp_template.shape[:2]
            if g.shape[0] < th or g.shape[1] < tw:
                return False
            res = cv2.matchTemplate(g, self._tp_template, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, _ = cv2.minMaxLoc(res)
            return maxv >= getattr(self.cfg.control, "teleport_template_threshold", 0.7)
        except Exception:
            return False

    def _find_and_click(self, template, threshold, click_offset=None,
                        retries=10, wait=0.25, label="") -> bool:
        """Найти шаблон на экране и кликнуть. click_offset=(dx,dy) от лев-верх угла
        совпадения (по умолч. центр шаблона). Координаты переводятся в абсолютные
        экранные (+смещение региона захвата). Несколько попыток (элемент появляется
        не сразу). True — кликнул."""
        if template is None or cv2 is None:
            print(f"[run] ТЕЛЕПОРТ: нет шаблона для '{label}'")
            return False
        th, tw = template.shape[:2]
        ox, oy = self._capture_offset()
        for _ in range(retries):
            frame = self.tracker.cap.grab()
            if frame is None:
                time.sleep(0.03)
                continue
            try:
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if g.shape[0] < th or g.shape[1] < tw:
                    return False
                res = cv2.matchTemplate(g, template, cv2.TM_CCOEFF_NORMED)
                _, maxv, _, loc = cv2.minMaxLoc(res)
                if maxv >= threshold:
                    if click_offset is None:
                        cx, cy = loc[0] + tw // 2, loc[1] + th // 2
                    else:
                        cx, cy = loc[0] + click_offset[0], loc[1] + click_offset[1]
                    self.inp.click_at(ox + cx, oy + cy)
                    print(f"[run] ТЕЛЕПОРТ: клик '{label}' ({ox+cx},{oy+cy}) score={maxv:.2f}")
                    return True
            except Exception:
                pass
            time.sleep(wait)
        print(f"[run] ТЕЛЕПОРТ: '{label}' не найден на экране (порог {threshold})")
        return False

    def _do_teleport(self):
        """Фаст-тревел через карту: стоп → F (активировать точку) → ждём F-меню →
        закрыть F-меню (клик teleport_dismiss_xy или Esc) → M (карта) →
        клик жёлтый ромбик → клик иконка «Точка телепортации» слева →
        клик кнопка «Телепорт» снизу → пауза → дальше."""
        c = self.cfg.control
        print("[run] ТЕЛЕПОРТ: стоп, активирую (F)")
        self.inp.stop_moving()
        self._set_sprint(False)
        time.sleep(0.3)
        self.inp.tap(getattr(c, "teleport_activate_key", "f"), 0.06)     # F — активировать точку
        time.sleep(getattr(c, "teleport_wait_menu_s", 1.5))              # ждём F-меню
        # Закрыть F-меню с ромбиком: клик по dismiss_xy (абсолютные координаты экрана)
        # или Esc — иначе M откроет не карту мира, а просто закроет всплывашку
        dismiss_xy = getattr(c, "teleport_dismiss_xy", None)
        if dismiss_xy:
            self.inp.click_at(int(dismiss_xy[0]), int(dismiss_xy[1]))
            print(f"[run] ТЕЛЕПОРТ: закрываю F-меню (клик {dismiss_xy})")
        else:
            self.inp.tap("escape", 0.05)
            print("[run] ТЕЛЕПОРТ: закрываю F-меню (Esc)")
        time.sleep(getattr(c, "teleport_dismiss_wait_s", 0.5))
        self._map_teleport()                                             # M → ромбик → иконка → «Телепорт»
        self._dr_t = None          # не интегрировать паузу телепорта в dead-reckon
        self._drive()
        print("[run] ТЕЛЕПОРТ: готово, продолжаю маршрут")

    def _do_climb(self, dur: float):
        """Карабканье на стену/скалу: высоту миникарта не видит, поэтому подъём —
        по времени. Прыжок (зацепиться за выступ) → держим W вверх dur секунд, с
        до-прыжками на выступах. Dead-reckon на время подъёма не интегрируем."""
        c = self.cfg.control
        do_jump = getattr(c, "climb_jump", True)
        print(f"[run] КАРАБКАНЬЕ: {'прыжок + ' if do_jump else ''}держу W {dur:.1f}с")
        self._set_sprint(False)
        if do_jump:
            self.inp.jump()                 # зацепиться за выступ
            time.sleep(0.15)
        self.inp.start_moving()             # держим W — лезем вверх
        t_end = time.monotonic() + dur
        next_jump = time.monotonic() + getattr(c, "climb_jump_interval_s", 1.5)
        while time.monotonic() < t_end and self._running:
            if do_jump and time.monotonic() >= next_jump:   # подпрыгнуть на выступе
                self.inp.jump()
                next_jump = time.monotonic() + getattr(c, "climb_jump_interval_s", 1.5)
            self.tracker.poll()             # держим захват живым
            time.sleep(0.02)
        self._dr_t = None                   # не интегрировать подъём в dead-reckon
        self._drive()
        print("[run] КАРАБКАНЬЕ: готово, продолжаю маршрут")

    def _map_teleport(self):
        """Только часть с картой: M → пауза → клик жёлтый ромбик → клик иконка
        «Точка телепортации» слева → клик кнопка «Телепорт» снизу. Вынесено отдельно,
        чтобы тестировать без F и без маршрута (scripts/test_teleport.py)."""
        c = self.cfg.control
        thr = getattr(c, "teleport_click_threshold", 0.7)
        step = getattr(c, "teleport_step_wait_s", 0.8)
        print("[run] ТЕЛЕПОРТ: открываю карту (M)")
        self.inp.tap(getattr(c, "teleport_open_map_key", "m"), 0.06)     # M — карта мира
        time.sleep(getattr(c, "teleport_map_wait_s", 1.0))
        # 1) жёлтый ромбик (точка на карте)
        self._find_and_click(self._tpl_rombik, thr, label="ромбик")
        time.sleep(step)
        # 2) иконка «Точка телепортации» слева в панели
        off = (getattr(c, "teleport_point_click_dx", 30), getattr(c, "teleport_point_click_dy", 42))
        self._find_and_click(self._tpl_point, thr, click_offset=off, label="иконка точки")
        time.sleep(step)
        # 3) кнопка «Телепорт» снизу
        self._find_and_click(self._tpl_confirm, thr, label="кнопка Телепорт")
        time.sleep(getattr(c, "teleport_after_s", 1.2))

    # ---- настройка одного сегмента -------------------------------------------
    def _setup_segment(self, route):
        """Подготовить прогон ОДНОГО маршрута-сегмента: follower, телепорты,
        привязка позиции к его 1-й точке, калибровка смещения курса из его
        start_heading. Вызывается для каждого сегмента в цепочке (после телепорта
        бот перепривязывается к новому сегменту — своя система координат)."""
        self.route = route
        self.follower = RouteFollower(route, self.cfg.control)
        self._dr_pos = [0.0, 0.0]
        self._dr_t = None
        self._tp_pending = {i for i, w in enumerate(route.waypoints)
                            if getattr(w, "action", None) == "teleport"}
        if self._tp_pending:
            print(f"[run] точек-телепортов: {len(self._tp_pending)} (индексы {sorted(self._tp_pending)})")
        # точки-КАРАБКАНЬЯ: idx -> длительность подъёма (сек). Срабатывают по достижении.
        self._climb_pending = {i: (getattr(w, "dur", None) or 2.0)
                               for i, w in enumerate(route.waypoints)
                               if getattr(w, "action", None) == "climb"}
        if self._climb_pending:
            print(f"[run] точек-карабканья: {len(self._climb_pending)} (индексы {sorted(self._climb_pending)})")
        if route.waypoints:
            wp0 = route.waypoints[0]
            self.tracker.set_position(wp0.x, wp0.y)
            self._dr_pos = [wp0.x, wp0.y]
            print(f"[run] старт-позиция сегмента = точка #1 ({wp0.x:.1f},{wp0.y:.1f})")

        # КАЛИБРОВКА СМЕЩЕНИЯ КУРСА. Надёжно — из start_heading маршрута (поза на
        # старте не важна, руль довернёт сам). Фолбэк (старый маршрут) — курс стоя.
        self._hdg_offset = 0.0
        brg0 = self._start_bearing()
        route_sh = getattr(route, "start_heading", None)
        if route_sh is not None and brg0 is not None:
            self._hdg_offset = angle_diff_deg(route_sh, brg0)
            print(f"[run] калибровка курса ИЗ МАРШРУТА: start_heading {route_sh:.0f}°, "
                  f"пеленг {brg0:.0f}° → смещение {self._hdg_offset:+.0f}° (поза не важна)")
        else:
            hs = self._avg_heading(0.8)
            if hs is not None and brg0 is not None:
                self._hdg_offset = angle_diff_deg(hs, brg0)
                print(f"[run] калибровка курса (фолбэк): смещение {self._hdg_offset:+.0f}° "
                      f"(СТАНЬ ЛИЦОМ вдоль маршрута)")
            else:
                print("[run] калибровка курса не удалась — смещение 0")

        # АБСОЛЮТНАЯ локализация по миникарте: грузим отпечатки этого сегмента
        self._localizer = None
        if getattr(self.cfg.control, "use_minimap_localize", False) and route.source_path:
            loc = load_localizer(route.source_path)
            if loc is not None and loc.n == len(route.waypoints):
                self._localizer = loc
                print(f"[run] локализация по миникарте: отпечатков {loc.n} — ВКЛ")
            elif loc is not None:
                print(f"[run] локализация ВЫКЛ: отпечатков {loc.n} ≠ точек {len(route.waypoints)} "
                      f"(перезапиши маршрут)")
            else:
                print("[run] локализация ВЫКЛ: нет файла отпечатков (.fp.npz) — перезапиши маршрут")

        if self.hud is not None and hasattr(self.hud, "set_route"):
            try:
                self.hud.set_route(route)
            except Exception:
                pass

    def _localize_correct(self, frame):
        """Коррекция позиции по миникарте: ищем, на какую записанную точку похож
        живой кадр (в окне вокруг текущей оценки), и подтягиваем dead-reckon к ней.
        Возврат True, если скорректировали (для лога)."""
        if self._localizer is None or frame is None:
            return False
        c = self.cfg.control
        l, t, w, h = self.cfg.minimap.region
        mm = frame[t:t + h, l:l + w]
        # кандидаты — точки в радиусе R (м) от ТЕКУЩЕЙ оценки (ловим дрейф, режем
        # ложные совпадения из других мест). Берём непрерывный диапазон индексов.
        R = getattr(c, "localize_radius_m", 40.0)
        px, py = self._dr_pos
        idxs = [i for i, wp in enumerate(self.route.waypoints)
                if (wp.x - px) ** 2 + (wp.y - py) ** 2 <= R * R]
        if not idxs:
            return False
        idx, sc = self._localizer.localize(mm, min(idxs), max(idxs) + 1)
        if idx < 0 or sc < getattr(c, "localize_threshold", 0.55):
            return False
        wp = self.route.waypoints[idx]
        g = getattr(c, "localize_correct_gain", 0.34)
        self._dr_pos[0] += g * (wp.x - self._dr_pos[0])      # мягко тянем к факт. месту
        self._dr_pos[1] += g * (wp.y - self._dr_pos[1])
        if idx > self.follower.wp_idx:                       # подтянуть прогресс вперёд
            self.follower.wp_idx = idx
        self._loc_dbg = (idx, sc)
        return True

    # ---- главный цикл --------------------------------------------------------
    def run(self):
        """Пройти ВСЕ сегменты (self.routes) по очереди. Между сегментами бот
        перепривязывается (своя система координат) — так работает продолжение после
        телепорта без перезаписи прежнего маршрута."""
        self._running = True
        print(f"[run] усиление руля {self.units_per_deg:+.1f} ед/° (знак из конфига); "
              f"сегментов: {len(self.routes)}")
        import os as _os
        _trace_path = _os.path.join(_os.path.dirname(_os.path.dirname(
            _os.path.dirname(_os.path.abspath(__file__)))), "diag", "run_trace.csv")
        _os.makedirs(_os.path.dirname(_trace_path), exist_ok=True)
        trace = open(_trace_path, "w", encoding="utf-8")
        trace.write("seg,t,px,py,hdg,brg,err,move,dx,action\n")
        t_start = time.monotonic()
        try:
            for seg_i, route in enumerate(self.routes):
                if not self._running:
                    break
                print(f"[run] ===== СЕГМЕНТ {seg_i + 1}/{len(self.routes)}: "
                      f"'{route.name}', точек {len(route.waypoints)} =====")
                self.inp.stop_moving()
                self._set_sprint(False)            # сброс спринта между сегментами (свежий фронт потом)
                if seg_i > 0:                      # пауза перед каждым НОВЫМ сегментом (кроме первого)
                    print("[run] пауза 1с перед новым сегментом")
                    time.sleep(1.0)
                self._setup_segment(route)
                self._drive()                      # едем (W); Shift включит цикл по ошибке курса
                res = self._drive_segment(seg_i, trace, t_start)
                if res in ("quit", "stopped"):
                    break
            else:
                print("[run] ВСЕ СЕГМЕНТЫ ПРОЙДЕНЫ")
        finally:
            try:
                trace.close()
                print(f"[run] трейс записан: {_trace_path}")
            except Exception:
                pass
            self.stop()

    def _drive_segment(self, seg_i, trace, t_start) -> str:
        """Вести текущий сегмент до конца. Возврат: 'done' | 'quit' | 'stopped'."""
        loop_dt = 1.0 / 30.0
        last_log = 0.0
        while self._running:
            t0 = time.monotonic()
            pose = self.tracker.poll()
            if pose is None:
                time.sleep(0.001)
                continue
            # курс: стартовая калибровка + ручная подстройка увода (heading_trim_deg)
            hdg = (pose.heading_deg - self._hdg_offset
                   + getattr(self.cfg.control, "heading_trim_deg", 0.0)) % 360.0

            # DEAD-RECKONING позиции из курса×скорость (пока держим W).
            if self._dr_t is not None and self.cfg.control.move_key in self.inp._held:
                dt = min(0.2, max(0.0, t0 - self._dr_t))
                spd = self.cfg.control.dead_reckon_speed
                rad = math.radians(hdg)
                self._dr_pos[0] += -math.sin(rad) * spd * dt
                self._dr_pos[1] += math.cos(rad) * spd * dt
            self._dr_t = t0

            # АБСОЛЮТНАЯ локализация по миникарте: подтянуть _dr_pos к фактическому
            # месту (если включена и есть отпечатки). Снимает дрейф/стамину.
            self._localize_correct(self.tracker.last_frame)

            # ПОЗИЦИЯ для follower. При включённой локализации — всегда _dr_pos
            # (её и корректирует локализатор). Иначе — по config position_source.
            if self._localizer is not None or \
                    getattr(self.cfg.control, "position_source", "odometry") == "dead_reckon":
                pos = (self._dr_pos[0], self._dr_pos[1])
            else:
                pos = (pose.player_xy[0], pose.player_xy[1])

            # ДЕЙСТВИЕ-ТЕЛЕПОРТ: по видимой подсказке (надёжно) или по расстоянию (фолбэк).
            if self._tp_pending:
                tol = getattr(self.cfg.control, "teleport_tolerance_m", 2.0)
                rad = getattr(self.cfg.control, "teleport_detect_radius_m", 8.0)
                for idx in sorted(self._tp_pending):
                    w = self.route.waypoints[idx]
                    dist = math.hypot(pos[0] - w.x, pos[1] - w.y)
                    fire = dist <= tol
                    if not fire and dist <= rad and self._tp_template is not None:
                        if self._tp_prompt_visible(self.tracker.last_frame):
                            print(f"[run] подсказка телепорта найдена на экране (d={dist:.1f}м)")
                            fire = True
                    if fire:
                        self._do_teleport()
                        self._tp_pending.discard(idx)
                        if self.follower.wp_idx <= idx:
                            self.follower.wp_idx = idx + 1
                        break

            # ДЕЙСТВИЕ-КАРАБКАНЬЕ: по достижению точки лезем вверх dur секунд.
            if self._climb_pending:
                tol = getattr(self.cfg.control, "climb_tolerance_m",
                              getattr(self.cfg.control, "teleport_tolerance_m", 2.0))
                for idx in sorted(self._climb_pending):
                    w = self.route.waypoints[idx]
                    if math.hypot(pos[0] - w.x, pos[1] - w.y) <= tol:
                        self._do_climb(self._climb_pending[idx])
                        del self._climb_pending[idx]
                        if self.follower.wp_idx <= idx:
                            self.follower.wp_idx = idx + 1
                        break

            d = self.follower.step(pos, hdg)
            if d.done:
                print("[run] СЕГМЕНТ ПРОЙДЕН")
                return "done"

            self._drive()                      # держим W (+Shift) всегда (кроме телепорта)

            err = d.heading_err_deg
            # спринт ТОЛЬКО на ~прямых: большой |err| (поворот/стартовый разворот) →
            # шаг (тугой разворот, малый занос); малый err → спринт (свежий фронт Shift)
            self._set_sprint(abs(err) <= getattr(self.cfg.control, "sprint_max_err_deg", 30.0))
            self._last_dx = 0.0
            self._auto_sign(err, t0)
            act = "steer"
            self._steer(err, t0)

            trace.write(f"{seg_i},{t0 - t_start:.3f},{pos[0]:.2f},{pos[1]:.2f},{hdg:.1f},"
                        f"{d.bearing_deg:.1f},{d.heading_err_deg:+.1f},"
                        f"{1 if pose.moving else 0},{self._last_dx:+.0f},{act}\n")
            trace.flush()

            if t0 - last_log >= 0.4:
                last_log = t0
                print(f"[run] сег{seg_i+1} wp#{d.wp_idx} d={d.dist_m:5.1f}м hdg={pose.heading_deg:5.0f} "
                      f"brg={d.bearing_deg:5.0f} err={d.heading_err_deg:+5.0f} "
                      f"move={'Y' if pose.moving else 'n'}")

            if self.hud is not None:
                cmd = self.hud.update(self.tracker.last_frame, (pos[0], pos[1]),
                                      pose.heading_deg, hdg, d, pose.moving)
                if cmd == "quit":
                    print("[run] HUD: q — остановка")
                    return "quit"

            slack = loop_dt - (time.monotonic() - t0)
            if slack > 0:
                time.sleep(slack)
        return "stopped"

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
        if self.hud is not None:
            try:
                self.hud.close()
            except Exception:
                pass
