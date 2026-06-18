"""
Шаг 1 ТЗ — миникарта как базовая точка позиционирования.

Из миникарты достаём:
  * heading (курс камеры/персонажа) — по ориентации стрелки игрока в центре;
  * глобальную позицию (X, Y) в метрах — template-matching кропа миникарты
    по заранее сшитому атласу мира (north-up). Если атласа нет, отдаём только
    относительное смещение между кадрами (по фазовой корреляции).

Точность встроенной миникарты Genshin высокая — её показания берём за основу,
которую далее уточняет CV/триангуляция (fusion.estimator).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class MinimapReading:
    heading_deg: Optional[float]          # ОСНОВНОЙ курс: из сдвига карты (None если стоим)
    # ВНИМАНИЕ: тут лежит направление КОНУСА камеры (_cone_heading), НЕ cyan-стрелки.
    # read_heading() (по стрелке игрока) сейчас НЕ вызывается — оставлен на будущее.
    arrow_heading_deg: Optional[float]    # направление конуса камеры (для отладки/фолбэка)
    world_xy_m: Optional[Tuple[float, float]]  # глобальная позиция, метры (если есть атлас)
    delta_xy_m: Optional[Tuple[float, float]]  # смещение от прошлого кадра, метры
    confidence: float                     # 0..1 — качество локализации


class MinimapReader:
    def __init__(self, cfg, atlas: Optional[np.ndarray] = None):
        self.cfg = cfg
        self.atlas = atlas               # сшитая карта мира (grayscale), north-up
        self._prev_crop_gray: Optional[np.ndarray] = None
        self._hann: Optional[np.ndarray] = None
        self._motion_win: Optional[np.ndarray] = None     # кольцевая маска * Hann
        self._prev_head_gray: Optional[np.ndarray] = None # пред. кадр для курса (маскир.)
        self._prev_head_t: float = 0.0                    # время этого кадра (для интервала)
        self._head_vec: Optional[np.ndarray] = None       # сглаженный единичный вектор курса
        self._head_last_update_t: float = -1e9            # когда курс по карте реально обновлялся
        self._last_move_heading: Optional[float] = None   # удержанный курс из движения
        self._moving: bool = False                        # «движемся ли» — гейт одометрии (delta)
        self._arrow_smooth: Optional[float] = None        # сглаженный курс по стрелке игрока
        self._arrow_bad: int = 0                          # счётчик подряд идущих «скачков» (выбросов)
        self._last_world: Optional[Tuple[float, float]] = None
        # КРУГОВАЯ МЕДИАНА курса по стрелке — главный фильтр выбросов. Сверено на дампе
        # «круг» (probe_circle.py, 742 кадра по всем направлениям): сырая стрелка верна
        # в ~88% (медиана скачка 0.5°), но 12% кадров — одиночные спайки до 147°. Медиана
        # медиана их гасит (p90 скачка 112°→4°), НЕ лагая как EMA. Окно 5 — баланс:
        # давит одиночные/двойные спайки, но БЕЗ лага (окно 9 отставало на поворотах,
        # «стрелка вела хуже»; редкие 4-кадровые глюки терпим — они не стоят лага).
        # Motion-курс на этом зуме мёртв (сдвиг суб-пиксельный, макс 0.5px) — поэтому стрелка.
        self._arrow_buf: deque = deque(maxlen=5)   # окно 5: давит спайки БЕЗ лага (9 лагало на поворотах)
        self._main_heading_smooth: Optional[float] = None   # (устар., EMA-фолбэк)
        self._main_heading_bad: int = 0
        self._heading_smooth_alpha = 0.15
        if atlas is None and cfg.world_atlas_path:
            self.load_atlas(cfg.world_atlas_path)

    def load_atlas(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[minimap] не удалось загрузить атлас: {path}")
            return
        self.atlas = img

    # ---- heading -----------------------------------------------------------
    def read_heading(self, minimap_bgr: np.ndarray) -> Optional[float]:
        """
        Курс по ГОЛУБОЙ (cyan) стрелке игрока в центре миникарты (миникарта
        север-залочена: север всегда вверху, стрелка вращается по курсу).

        Стрелка по hue близка к фону карты, поэтому: берём узкую центральную зону
        (стрелка всегда в центре), морфологией убираем шум и не даём слипнуться с
        посторонним cyan, берём крупнейший контур и вектор центроид->самая дальняя
        (носовая) точка.
        """
        hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
        lo = np.array(self.cfg.arrow_hsv_low, np.uint8)
        hi = np.array(self.cfg.arrow_hsv_high, np.uint8)
        mask = cv2.inRange(hsv, lo, hi)

        h, w = mask.shape
        cx, cy = w // 2, h // 2
        r = max(8, int(min(w, h) * self.cfg.arrow_roi_frac))
        roi = np.zeros_like(mask)
        cv2.circle(roi, (cx, cy), r, 255, -1)
        mask = cv2.bitwise_and(mask, roi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Стрелка игрока ВСЕГДА в центре миникарты. Пины/иконки — смещены. Поэтому
        # из cyan-блобов берём не самый крупный, а самый ЦЕНТРАЛЬНЫЙ (это и отсекает
        # белую «каплю»-пин и прочий мусор, дававший редкие срывы курса).
        best, best_d = None, 1e9
        for c in cnts:
            if cv2.contourArea(c) < 30:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            gx, gy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            d = np.hypot(gx - cx, gy - cy)
            if d < best_d:
                best, best_d = c, d
        if best is None or best_d > r:        # ничего центрального не нашли
            return None
        c = best
        M = cv2.moments(c)
        gx, gy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        pts = c.reshape(-1, 2).astype(np.float64)
        tip = pts[np.argmax(np.hypot(pts[:, 0] - gx, pts[:, 1] - gy))]
        dx, dy = tip[0] - gx, tip[1] - gy
        # экран: y вниз. Курс: 0=север(вверх), по часовой.
        heading = np.degrees(np.arctan2(dx, -dy))
        return float((heading + 360.0) % 360.0)

    # ---- глобальная локализация по атласу ----------------------------------
    def localize(self, minimap_bgr: np.ndarray,
                 heading_deg: Optional[float]) -> Optional[Tuple[float, float]]:
        """
        Глобальная позиция через сопоставление миникарты с атласом.
        Миникарта у нас СЕВЕР-ЗАЛОЧЕНА (подтверждено: курс из сдвига карты зависит от
        направления движения) — поэтому read() передаёт heading_deg=0.0 и разворот
        ниже становится no-op. Ветка warpAffine оставлена на случай camera-locked
        миникарты (другая настройка HUD); сейчас НЕ используется (атлас не задан).
        """
        if self.atlas is None or heading_deg is None or cv2 is None:
            return None
        crop = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
        h, w = crop.shape
        # маскируем круг миникарты, разворачиваем в north-up
        Mrot = cv2.getRotationMatrix2D((w / 2, h / 2), -heading_deg, 1.0)
        north = cv2.warpAffine(crop, Mrot, (w, h))
        # центральная вырезка без рамки/иконок по краям
        m = int(min(w, h) * 0.30)
        templ = north[m:h - m, m:w - m]
        if templ.size == 0:
            return None
        res = cv2.matchTemplate(self.atlas, templ, cv2.TM_CCOEFF_NORMED)
        _, maxval, _, maxloc = cv2.minMaxLoc(res)
        if maxval < 0.35:                       # слабый матч — не доверяем
            return None
        # центр найденного шаблона в пикселях атласа -> метры
        ax = maxloc[0] + templ.shape[1] / 2.0
        ay = maxloc[1] + templ.shape[0] / 2.0
        mpp = self.cfg.atlas_meters_per_px
        world = (ax * mpp, ay * mpp)
        self._last_world = world
        return world

    # ---- относительное смещение (фазовая корреляция) -----------------------
    def relative_shift(self, minimap_bgr: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Сдвиг ТЕКСТУРЫ карты в метрах между текущим и прошлым кадром.
        Миникарта север-залочена -> НЕ разворачиваем кроп. Маскируем тем же
        КОЛЬЦЕВЫМ окном (_motion_window), что и курс: вырезаем центральный маркер
        игрока, иначе вращающаяся cyan-стрелка тянет phaseCorrelate к себе и
        одометрия недооценивает путь + получает направление-зависимый сдвиг (увод
        трека). Окно уже содержит Hann, поэтому отдельное Hann-окно не нужно.
        """
        gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = gray * self._motion_window(gray.shape)
        out = None
        if self._prev_crop_gray is not None and self._prev_crop_gray.shape == g.shape:
            (sx, sy), _ = cv2.phaseCorrelate(self._prev_crop_gray, g)
            mpp = self.cfg.minimap_meters_per_px
            out = (sx * mpp, sy * mpp)
        self._prev_crop_gray = g
        return out

    def _motion_window(self, shape: Tuple[int, int]) -> np.ndarray:
        """Кольцевая маска (без центра и углов) * Hann — кэшируется по форме кропа."""
        if self._motion_win is not None and self._motion_win.shape == shape:
            return self._motion_win
        h, w = shape
        cx, cy = w // 2, h // 2
        rad = min(h, w)
        ring = np.zeros(shape, np.float32)
        cv2.circle(ring, (cx, cy), int(rad * self.cfg.motion_ring_outer_frac), 1.0, -1)
        cv2.circle(ring, (cx, cy), int(rad * self.cfg.motion_ring_inner_frac), 0.0, -1)
        hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
        self._motion_win = ring * hann
        return self._motion_win

    def _heading_from_motion(self, minimap_bgr: np.ndarray,
                             now: Optional[float] = None) -> Optional[float]:
        """
        Курс из движения: сравниваем кадры с интервалом heading_sample_dt (на
        высокой частоте сдвиг карты за кадр суб-пиксельный и тонет в шуме). Каждый
        сэмпл с хорошим качеством (response) и достаточным сдвигом уже даёт надёжное
        направление — берём его сразу и сглаживаем круговым EMA (без долгого
        накопления, чтобы стрелка появлялась через ~0.15с и плавно обновлялась).
        Кольцевая маска убирает статичный маркер игрока в центре и углы.
        Стоим (сдвиг ниже порога) — держим последний курс. 0=север, по часовой.

        now — время в секундах (для тестов); по умолчанию time.monotonic().
        """
        import math
        import time as _time
        if now is None:
            now = _time.monotonic()
        gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = gray * self._motion_window(gray.shape)
        if self._prev_head_gray is None or self._prev_head_gray.shape != g.shape:
            self._prev_head_gray = g
            self._prev_head_t = now
            return self._last_move_heading
        # слишком рано — кадры почти одинаковы, сдвиг суб-пиксельный (шум). Ждём.
        if now - self._prev_head_t < self.cfg.heading_sample_dt:
            return self._last_move_heading
        (sx, sy), resp = cv2.phaseCorrelate(self._prev_head_gray, g)
        self._prev_head_gray = g
        self._prev_head_t = now
        # шумный или почти нулевой сэмпл (стоим/мусор) — держим последний курс
        if resp < self.cfg.heading_min_response or \
           math.hypot(sx, sy) < self.cfg.heading_move_floor_px:
            self._moving = False          # стоим → одометрия должна молчать (без дрейфа)
            return self._last_move_heading
        self._moving = True               # реальное движение по текстуре карты
        s = self.cfg.move_sign
        # единичный вектор мгновенного курса (экран: y вниз, север=вверх)
        ang = math.atan2(s * sx, -(s * sy))
        v = np.array([math.cos(ang), math.sin(ang)])
        if self._head_vec is None:
            self._head_vec = v
        else:
            a = self.cfg.heading_smooth_alpha
            self._head_vec = (1 - a) * self._head_vec + a * v   # круговое EMA
        hv = self._head_vec
        self._last_move_heading = float((math.degrees(math.atan2(hv[1], hv[0])) + 360.0) % 360.0)
        self._head_last_update_t = now            # курс по карте реально обновился
        return self._last_move_heading

    def _cone_heading(self, minimap_bgr: np.ndarray) -> Optional[float]:
        """
        Направление голубого КОНУСА камеры (текстуро-независимо). Узкая центральная
        зона, морфология, крупнейший контур, нос = самая дальняя точка от центроида.
        0=север(вверх), по часовой. Конус = направление камеры (куда пойдёт W).
        """
        import math
        if not self.cfg.use_cone_fallback:
            return None
        hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
        lo = np.array(self.cfg.cone_hsv_low, np.uint8)
        hi = np.array(self.cfg.cone_hsv_high, np.uint8)
        m = cv2.inRange(hsv, lo, hi)
        h, w = m.shape
        cx, cy = w // 2, h // 2
        roi = np.zeros_like(m)
        cv2.circle(roi, (cx, cy), max(8, int(min(h, w) * self.cfg.cone_roi_frac)), 255, -1)
        m = cv2.bitwise_and(m, roi)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < self.cfg.cone_min_area:
            return None
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        gx, gy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        pts = c.reshape(-1, 2).astype(np.float64)
        tip = pts[np.argmax(np.hypot(pts[:, 0] - gx, pts[:, 1] - gy))]
        return float((math.degrees(math.atan2(tip[0] - gx, -(tip[1] - gy))) + 360.0) % 360.0)

    # ---- фильтр выбросов курса по стрелке -----------------------------------
    @staticmethod
    def _ang_diff(a: float, b: float) -> float:
        """Кратчайшая разница углов a-b в диапазоне (-180, 180]."""
        d = (a - b + 180.0) % 360.0 - 180.0
        return d

    def _filter_arrow(self, raw: Optional[float]) -> Optional[float]:
        """
        Отсекает редкие срывы детектора стрелки. Персонаж не разворачивается на
        150° за один кадр (~33 мс): одиночный скачок >70° считаем ошибкой и держим
        старый курс. Но если новое направление подтверждается JUMP_CONFIRM кадров
        подряд — это реальный быстрый поворот, принимаем.
        """
        JUMP_DEG = 70.0
        JUMP_CONFIRM = 3
        if raw is None:
            return self._arrow_smooth                    # потеря детекции — держим старое
        if self._arrow_smooth is None:
            self._arrow_smooth = raw                     # первая инициализация
            self._arrow_bad = 0
            return raw
        if abs(self._ang_diff(raw, self._arrow_smooth)) > JUMP_DEG:
            self._arrow_bad += 1
            if self._arrow_bad < JUMP_CONFIRM:
                return self._arrow_smooth                # выброс — игнорируем
            # подтверждённый реальный поворот — принимаем
        self._arrow_bad = 0
        self._arrow_smooth = raw
        return raw

    def _median_arrow(self, raw: Optional[float]) -> Optional[float]:
        """Круговая медиана последних N курсов по стрелке — гасит одиночные спайки
        детектора (конус/хвост/пин), НЕ внося лага EMA. Медиана = элемент буфера с
        минимальной суммой |угловых разниц| до остальных (устойчива к выбросам)."""
        if raw is not None:
            self._arrow_buf.append(raw)
        if not self._arrow_buf:
            return None
        buf = list(self._arrow_buf)
        best, best_s = buf[0], 1e18
        for x in buf:
            s = sum(abs(self._ang_diff(x, y)) for y in buf)
            if s < best_s:
                best_s, best = s, x
        return best

    @staticmethod
    def _smooth_angle(new_angle: float, old_angle: float, alpha: float) -> float:
        """Плавное усреднение углов с учётом цикличности (0..360)."""
        diff = (new_angle - old_angle + 180.0) % 360.0 - 180.0
        return float((old_angle + alpha * diff) % 360.0)

    def _filter_heading(self, raw: Optional[float]) -> Optional[float]:
        """
        Фильтр выбросов + EMA для ОСНОВНОГО курса. Скачок >50° держим старое (с
        подтверждением 2 кадра), иначе сглаживаем. ВНИМАНИЕ: это смягчает прыжки
        arrow-primary, но не лечит корень — глюк стрелки на 2+ кадра проходит, а
        alpha=0.15 добавляет лаг. Настоящий фундамент — heading_detector.py (motion).
        """
        JUMP_DEG = 50.0
        JUMP_CONFIRM = 2
        if raw is None:
            return self._main_heading_smooth
        if self._main_heading_smooth is None:
            self._main_heading_smooth = raw
            self._main_heading_bad = 0
            return raw
        if abs(self._ang_diff(raw, self._main_heading_smooth)) > JUMP_DEG:
            self._main_heading_bad += 1
            if self._main_heading_bad < JUMP_CONFIRM:
                return self._main_heading_smooth          # выброс — держим старое
        else:
            self._main_heading_bad = 0
        self._main_heading_smooth = self._smooth_angle(
            raw, self._main_heading_smooth, self._heading_smooth_alpha)
        return self._main_heading_smooth

    # ---- основной вызов ----------------------------------------------------
    def read(self, frame_bgr: np.ndarray) -> MinimapReading:
        """frame_bgr — полный кадр игры; регион миникарты берём из cfg."""
        import time as _time
        now = _time.monotonic()
        l, t, w, h = self.cfg.region
        mm = frame_bgr[t:t + h, l:l + w]
        arrow_h = self._median_arrow(self.read_heading(mm))  # СТРЕЛКА игрока + круговая медиана (гасит спайки)
        map_h = self._heading_from_motion(mm, now)       # курс по сдвигу карты (ставит self._moving)
        cone_h = self._cone_heading(mm)                  # курс по конусу камеры (резерв)
        delta = self.relative_shift(mm)                  # сдвиг карты за кадр (для скорости)
        if delta is not None and not self._moving:       # стоим → глушим, иначе fused_pos дрейфует
            delta = (0.0, 0.0)

        # ОСНОВНОЙ курс — по СТРЕЛКЕ игрока. Подтверждено живым дампом
        # (diag_fast_dump.py): 180/180 кадров, круговой СКО 0.1° — идеально стабильна
        # после ужесточения HSV. И она СОГЛАСОВАНА с системой follower'а: при
        # move_sign=-1 стрелка совпадает с курсом-из-сдвига-карты в пределах ~7°.
        # Курс-из-сдвига-карты на этом зуме миникарты НЕЧИТАЕМ (сдвиг при ходьбе
        # медиана 0.01px, выше floor лишь 8% кадров) → годится только как фолбэк,
        # когда стрелка не нашлась. Конус — последний резерв.
        fresh = (now - self._head_last_update_t) < self.cfg.heading_stale_after_s
        if arrow_h is not None:
            heading = arrow_h                            # ОСНОВНОЙ: стрелка (стабильна, согласована)
        elif fresh and map_h is not None:
            heading = map_h                              # стрелка пропала → свежий сдвиг карты
        elif cone_h is not None:
            heading = cone_h
        else:
            heading = map_h                              # последний удержанный

        # медиана уже применена к стрелке выше; доп. EMA не нужна (вносила лаг)
        world = self.localize(mm, 0.0)                   # атлас north-up (если задан)
        conf = 0.9 if world is not None else (0.6 if heading is not None else 0.2)
        return MinimapReading(heading, cone_h, world, delta, conf)
