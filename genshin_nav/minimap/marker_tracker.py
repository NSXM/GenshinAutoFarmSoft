#!/usr/bin/env python3
"""
marker_tracker.py — порт внешнего детектора курса (genshin_minimap_tracker.py).

Альтернатива arrow-median курсу из minimap_reader.MinimapReader: вместо
«центроид → самая дальняя точка» ищет ОСЬ маркера через минимум поперечной
дисперсии дальнего конца (устойчивее к шуму на кончике), а неоднозначность
180° разрешает по дисперсии среды в конусе FOV + continuity между кадрами.

Класс MinimapTracker оставлен ДОСЛОВНО как во внешнем файле (чтобы рядом
лежащий live_debug.py продолжал работать). Добавлен лишь from_cfg() —
конструктор из MinimapCfg проекта (формат региона (left,top,w,h) и HSV
стрелки берутся из config.yaml).

Использование в проекте:
    from genshin_nav.minimap.marker_tracker import MinimapTracker
    tracker = MinimapTracker.from_cfg(cfg.minimap)
    angle, conf = tracker.detect(full_frame_bgr)   # ПОЛНЫЙ кадр игры
"""

import cv2
import numpy as np
import math


class MinimapTracker:
    def __init__(self,
                 minimap_region=(0, 0, 200, 200),   # x1,y1,x2,y2 в пикселях кадра
                 hsv_lower=(85, 180, 180),
                 hsv_upper=(105, 255, 255),
                 min_marker_area=50,
                 max_marker_area=500,
                 cone_r_near=12,
                 cone_r_far=35):
        """
        minimap_region: область миникарты в кадре (x1,y1,x2,y2)
        hsv_lower/upper: HSV диапазон маркера (голубой, BGR=(255,220,0) ≈ H=90°)
        min/max_marker_area: допустимый диапазон площади маркера (пикс²)
        cone_r_near/far: кольцо для анализа конуса FOV
        """
        self.region = minimap_region
        self.hsv_lower = np.array(hsv_lower)
        self.hsv_upper = np.array(hsv_upper)
        self.min_area = min_marker_area
        self.max_area = max_marker_area
        self.cone_r_near = cone_r_near
        self.cone_r_far = cone_r_far

        # Состояние для frame-to-frame трекинга
        self._prev_angle = None
        self._prev_confidence = 0.0
        self._polarity = None      # +1 = phi_a есть нос, -1 = phi_b есть нос
        self._stable_frames = 0

    # ------------------------------------------------------------------ from_cfg
    @classmethod
    def from_cfg(cls, mcfg, **overrides):
        """
        Построить трекер из MinimapCfg проекта.

        mcfg.region хранится как (left, top, w, h) — переводим в (x1,y1,x2,y2).
        HSV стрелки берём из mcfg.arrow_hsv_low/high (та же cyan-стрелка, что
        ищет MinimapReader). Площадь/кольцо конуса можно переопределить через
        **overrides (min_marker_area, max_marker_area, cone_r_near, cone_r_far).
        """
        l, t, w, h = mcfg.region
        region = (int(l), int(t), int(l + w), int(t + h))
        params = dict(
            minimap_region=region,
            hsv_lower=tuple(mcfg.arrow_hsv_low),
            hsv_upper=tuple(mcfg.arrow_hsv_high),
            # маркер на 200px-миникарте мелкий — расширяем дефолтный диапазон
            min_marker_area=30,
            max_marker_area=1500,
        )
        params.update(overrides)
        return cls(**params)

    def reset_polarity(self):
        """Сбросить трекинг (вызвать после телепорта или долгой паузы)"""
        self._prev_angle = None
        self._polarity = None
        self._stable_frames = 0

    def detect(self, frame_bgr):
        """
        Определить угол взгляда персонажа.

        Args:
            frame_bgr: numpy array BGR (OpenCV формат), полный кадр игры

        Returns:
            (angle_degrees, confidence):
                angle_degrees: 0-359, компасный азимут (0=север, 90=восток)
                confidence: 0.0-1.0 (выше = надёжнее)
                (None, 0.0) если маркер не найден
        """
        x1, y1, x2, y2 = self.region
        mm = frame_bgr[y1:y2, x1:x2]

        # 1. Найти маркер
        hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0

        valid = [(cv2.contourArea(c), c) for c in contours
                 if self.min_area < cv2.contourArea(c) < self.max_area]
        if not valid:
            return None, 0.0

        _, main_cnt = max(valid, key=lambda x: x[0])
        mask_c = np.zeros_like(mask)
        cv2.drawContours(mask_c, [main_cnt], -1, 255, -1)

        ys, xs = np.where(mask_c > 0)
        if len(xs) < 10:
            return None, 0.0

        pts = np.column_stack([xs.astype(float), ys.astype(float)])
        centroid = pts.mean(axis=0)
        pts_c = pts - centroid
        n_top = max(3, len(pts) // 5)

        # 2. Найти ось маркера (направление с min поперечной дисперсией дальнего конца)
        scores = np.zeros(360)
        for phi_deg in range(360):
            phi = math.radians(phi_deg)
            dx, dy = math.cos(phi), math.sin(phi)
            proj = pts_c[:,0]*dx + pts_c[:,1]*dy
            perp = pts_c[:,0]*(-dy) + pts_c[:,1]*dx
            top_idx = np.argsort(proj)[-n_top:]
            scores[phi_deg] = perp[top_idx].std() if n_top > 1 else 999.0

        phi_a = int(np.argmin(scores))
        phi_b = (phi_a + 180) % 360

        # 3. Определить полярность (нос = phi_a или phi_b)
        #    через дисперсию конуса FOV в кольце вокруг маркера
        V = hsv[:,:,2].astype(float)
        h_img, w_img = V.shape
        gy, gx = np.mgrid[0:h_img, 0:w_img]
        r = np.sqrt((gx - centroid[0])**2 + (gy - centroid[1])**2)
        px = (gx - centroid[0]).astype(float)
        py = (gy - centroid[1]).astype(float)
        half_cone = math.radians(50)

        def cone_dispersion(phi_deg):
            phi = math.radians(phi_deg)
            dx, dy = math.cos(phi), math.sin(phi)
            ang = np.arctan2(py, px)
            phi_pix = math.atan2(dy, dx)
            diff = np.abs(np.angle(np.exp(1j*(ang - phi_pix))))
            sector = ((r >= self.cone_r_near) & (r <= self.cone_r_far)
                      & (diff <= half_cone))
            if sector.sum() < 5:
                return 999.0
            return float(V[sector].std())

        var_a = cone_dispersion(phi_a)
        var_b = cone_dispersion(phi_b)

        # Нос = конец с меньшей дисперсией среды (конус более однороден)
        polarity_score = var_b - var_a  # > 0 → phi_a = нос

        # Если есть предыдущий угол — используем continuity для стабилизации
        if self._prev_angle is not None and self._polarity is not None:
            # Вычислить угол по обоим кандидатам
            def phi_to_compass(phi_deg):
                phi = math.radians(phi_deg)
                return math.degrees(math.atan2(math.cos(phi), -math.sin(phi))) % 360

            angle_if_a = phi_to_compass(phi_a)
            angle_if_b = phi_to_compass(phi_b)

            diff_a = abs(((angle_if_a - self._prev_angle) + 180) % 360 - 180)
            diff_b = abs(((angle_if_b - self._prev_angle) + 180) % 360 - 180)

            # Если один кандидат явно ближе к предыдущему — выбираем его
            if diff_a < diff_b - 20:
                chosen_phi = phi_a
                self._polarity = 1
            elif diff_b < diff_a - 20:
                chosen_phi = phi_b
                self._polarity = -1
            else:
                # Нет явного предпочтения — доверяем конусу
                chosen_phi = phi_a if polarity_score >= 0 else phi_b
                self._polarity = 1 if chosen_phi == phi_a else -1
        else:
            # Первый кадр или после сброса
            chosen_phi = phi_a if polarity_score >= 0 else phi_b
            self._polarity = 1 if chosen_phi == phi_a else -1

        # 4. Вычислить итоговый угол
        phi = math.radians(chosen_phi)
        compass = math.degrees(math.atan2(math.cos(phi), -math.sin(phi))) % 360

        # 5. Уверенность: насколько чёток минимум + насколько чёток конус
        score_gap = np.sort(scores)[1] - np.sort(scores)[0]  # разница 1-го и 2-го минимума
        cone_gap = abs(polarity_score)

        confidence = min(1.0, (score_gap / 0.5) * 0.5 + (cone_gap / 15.0) * 0.5)
        confidence = max(0.0, confidence)

        self._prev_angle = compass
        self._prev_confidence = confidence

        return compass, confidence

    def detect_raw(self, frame_bgr):
        """То же что detect(), но без frame-to-frame сглаживания."""
        saved_prev = self._prev_angle
        self._prev_angle = None
        angle, conf = self.detect(frame_bgr)
        self._prev_angle = saved_prev
        return angle, conf

    def draw_overlay(self, frame_bgr, angle=None, confidence=None):
        """Нарисовать отладочный оверлей на кадре (in-place)."""
        if angle is None:
            angle, confidence = self.detect_raw(frame_bgr)
        if angle is None:
            return frame_bgr

        x1, y1, x2, y2 = self.region
        mm = frame_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [(cv2.contourArea(c), c) for c in contours
                 if self.min_area < cv2.contourArea(c) < self.max_area]
        if not valid:
            return frame_bgr
        _, main_cnt = max(valid, key=lambda x: x[0])
        mask_c = np.zeros_like(mask)
        cv2.drawContours(mask_c, [main_cnt], -1, 255, -1)
        ys, xs = np.where(mask_c > 0)
        cx, cy = int(xs.mean()) + x1, int(ys.mean()) + y1

        # Стрелка в направлении угла
        phi_rad = math.radians(angle)
        # angle=0=север=вверх, angle=90=восток=вправо
        dx_pix = math.sin(phi_rad)   # восток → вправо (+x)
        dy_pix = -math.cos(phi_rad)  # север → вверх (-y)
        length = 25
        ex, ey = int(cx + dx_pix*length), int(cy + dy_pix*length)

        cv2.arrowedLine(frame_bgr, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.3)
        cv2.putText(frame_bgr, f"{angle:.0f}deg ({confidence:.2f})",
                    (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 0), 1)
        return frame_bgr
