"""
HUD-оверлей прохождения маршрута — окно поверх игры, чтобы ГЛАЗАМИ видеть, как
бот идёт по точкам, не сворачивая Genshin.

Показывает:
  * top-down карту маршрута: все точки, пройденный путь, ЦЕЛЕВАЯ точка подсвечена,
    текущая позиция бота (зелёная точка) + вектор курса;
  * вырезанную миникарту со стрелкой курса (зелёная = продакшн heading — тот, что
    «работает хорошо»);
  * цифры: точка #/всего, дистанция, пеленг, курс, ошибка курса, move, action.

Чисто визуальный слой: НЕ трогает логику вождения, только читает состояние.
Закрепляется поверх окна игры (WND_PROP_TOPMOST / WinAPI-фолбэк), как scripts/
compare_heading.py --live. Окно держим в правом верхнем углу.

Конвенция координат (как в route_follower): СЕВЕР=+y, ВОСТОК=-x (ось x зеркальна).
Для естественного вида (север вверх, восток вправо) точку мира (wx,wy) кладём на
экран как P=(-wx, -wy): -wx → вправо(восток), -wy → вверх(север, т.к. ось экрана
вниз). Вектор курса θ в мире = (-sinθ, cosθ) → на экране (sinθ, -cosθ).
"""
from __future__ import annotations

import math
import sys
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _set_topmost(window_name: str) -> bool:
    """Прижать окно поверх всего: родное свойство OpenCV, затем WinAPI-фолбэк."""
    if cv2 is None:
        return False
    try:
        if hasattr(cv2, "WND_PROP_TOPMOST"):
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1.0)
            return True
    except Exception:
        pass
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
        if not hwnd:
            return False
        ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0002 | 0x0001 | 0x0010)
        return True
    except Exception:
        return False


def _screen_size():
    if sys.platform == "win32":
        try:
            import ctypes
            sw = ctypes.windll.user32.GetSystemMetrics(0)
            sh = ctypes.windll.user32.GetSystemMetrics(1)
            if sw > 0 and sh > 0:
                return sw, sh
        except Exception:
            pass
    return 1920, 1080


class RouteHUD:
    def __init__(self, cfg, route, topmost: bool = True, corner: str = "top-right",
                 plot_size: int = 360, minimap_scale: int = 2):
        self.cfg = cfg
        self.route = route
        self.topmost = topmost
        self.corner = corner
        self.ps = plot_size
        self.mm_scale = minimap_scale
        self.win = "route HUD (genshin_auto_nav)"
        self._frames = 0
        self._positioned = False
        self.enabled = cv2 is not None and len(route.waypoints) > 0
        self._trail = []                  # след пройденной позиции (для наглядности)

        # габариты маршрута в экранной системе P=(-x,-y) — считаем один раз
        self._sx = [-w.x for w in route.waypoints]
        self._sy = [-w.y for w in route.waypoints]
        self._minx, self._maxx = min(self._sx), max(self._sx)
        self._miny, self._maxy = min(self._sy), max(self._sy)

    # ---- мир -> пиксели плана -------------------------------------------------
    def _to_px(self, wx: float, wy: float, pad: int = 24):
        sx, sy = -wx, -wy
        span_x = max(1e-6, self._maxx - self._minx)
        span_y = max(1e-6, self._maxy - self._miny)
        span = max(span_x, span_y)        # единый масштаб → без искажения пропорций
        usable = self.ps - 2 * pad
        scale = usable / span
        # центрируем маршрут в окне
        cx_world = (self._minx + self._maxx) / 2.0
        cy_world = (self._miny + self._maxy) / 2.0
        px = self.ps / 2.0 + (sx - cx_world) * scale
        py = self.ps / 2.0 + (sy - cy_world) * scale
        return int(round(px)), int(round(py))

    # ---- отрисовка панели плана ----------------------------------------------
    def _draw_plot(self, dr_pos, hdg_cal, tgt_idx, cur_idx):
        img = np.full((self.ps, self.ps, 3), 30, np.uint8)
        wps = self.route.waypoints

        # линия маршрута
        for i in range(len(wps) - 1):
            p1 = self._to_px(wps[i].x, wps[i].y)
            p2 = self._to_px(wps[i + 1].x, wps[i + 1].y)
            cv2.line(img, p1, p2, (90, 90, 90), 1, cv2.LINE_AA)

        # точки: пройденные тускло, будущие ярче, цель — крупный круг
        for i, w in enumerate(wps):
            p = self._to_px(w.x, w.y)
            if i < cur_idx:
                col, rad = (70, 110, 70), 2          # пройдено
            else:
                col, rad = (160, 160, 160), 3        # впереди
            cv2.circle(img, p, rad, col, -1, cv2.LINE_AA)
        if 0 <= tgt_idx < len(wps):
            pt = self._to_px(wps[tgt_idx].x, wps[tgt_idx].y)
            cv2.circle(img, pt, 7, (0, 200, 255), 2, cv2.LINE_AA)   # ЦЕЛЬ (оранж)

        # след бота
        self._trail.append(self._to_px(dr_pos[0], dr_pos[1]))
        self._trail[:] = self._trail[-400:]
        for i in range(1, len(self._trail)):
            cv2.line(img, self._trail[i - 1], self._trail[i], (60, 160, 60), 1, cv2.LINE_AA)

        # текущая позиция + вектор курса (зелёный)
        pc = self._to_px(dr_pos[0], dr_pos[1])
        cv2.circle(img, pc, 4, (0, 255, 0), -1, cv2.LINE_AA)
        if hdg_cal is not None:
            th = math.radians(hdg_cal)
            vx, vy = math.sin(th), -math.cos(th)     # курс в экранной системе
            ex, ey = int(pc[0] + vx * 22), int(pc[1] + vy * 22)
            cv2.arrowedLine(img, pc, (ex, ey), (0, 255, 0), 2, tipLength=0.3)

        # компас (С вверху)
        cv2.putText(img, "N", (self.ps - 16, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return img

    # ---- миникарта со стрелкой ------------------------------------------------
    def _draw_minimap(self, frame):
        l, t, w, h = self.cfg.minimap.region
        if frame is None:
            return None
        crop = frame[t:t + h, l:l + w]
        if crop.size == 0:
            return None
        s = self.mm_scale
        mm = cv2.resize(crop, (w * s, h * s), interpolation=cv2.INTER_NEAREST)
        return mm

    # ---- основной вызов из цикла раннера -------------------------------------
    def update(self, frame, dr_pos, hdg_raw, hdg_cal, decision, moving: bool) -> str:
        """
        Нарисовать кадр HUD. Возвращает строку команды: "" | "quit".
        frame      — последний кадр игры (для миникарты), может быть None;
        dr_pos     — позиция бота (dead-reckon), метры;
        hdg_raw    — курс по стрелке (зелёный, как видел пользователь);
        hdg_cal    — курс со стартовой калибровкой (в системе пеленга follower);
        decision   — FollowDecision из RouteFollower;
        moving     — флаг движения миникарты.
        """
        if not self.enabled:
            return ""
        try:
            plot = self._draw_plot(dr_pos, hdg_cal, getattr(decision, "wp_idx", 0),
                                   getattr(decision, "wp_idx", 0))

            # правая колонка: миникарта (фикс. ширина) + блок текста. Размеры
            # считаем ЗАРАНЕЕ, чтобы высота холста вместила всё (иначе текст
            # обрезался под миникартой).
            pad = 8
            right_w = 200
            mm = self._draw_minimap(frame)
            mm_h = 0
            if mm is not None:
                mw = min(mm.shape[1], right_w)
                mm_h = int(mm.shape[0] * mw / mm.shape[1])
                mm = cv2.resize(mm, (mw, mm_h))
                cxy = (mw // 2, mm_h // 2)
                if hdg_raw is not None:
                    th = math.radians(hdg_raw)
                    ex = int(cxy[0] + math.sin(th) * mw * 0.4)
                    ey = int(cxy[1] - math.cos(th) * mm_h * 0.4)
                    cv2.arrowedLine(mm, cxy, (ex, ey), (0, 255, 0), 2, tipLength=0.3)

            done = getattr(decision, "done", False)
            lines = [
                f"point  {getattr(decision,'wp_idx',0)+1}/{len(self.route.waypoints)}",
                f"dist   {getattr(decision,'dist_m',0.0):5.1f} m",
                f"bearing{getattr(decision,'bearing_deg',0.0):6.0f}",
                f"hdg    {hdg_raw:6.0f}" if hdg_raw is not None else "hdg      --",
                f"err    {getattr(decision,'heading_err_deg',0.0):+6.0f}",
                f"move   {'Y' if moving else 'n'}   {getattr(decision,'action','')}",
            ]
            if done:
                lines.append("ROUTE DONE")
            text_block_h = 12 + mm_h + 12 + len(lines) * 22 + 24
            H = max(self.ps, text_block_h)
            canvas = np.full((H, self.ps + right_w + pad * 3, 3), 20, np.uint8)
            canvas[0:self.ps, pad:pad + self.ps] = plot

            x0 = self.ps + pad * 2
            y = 12
            if mm is not None:
                canvas[y:y + mm_h, x0:x0 + mm.shape[1]] = mm
                y += mm_h + 12
            for ln in lines:
                cv2.putText(canvas, ln, (x0, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (235, 235, 235), 1, cv2.LINE_AA)
                y += 22
            cv2.putText(canvas, "q-quit", (x0, H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (140, 140, 140), 1, cv2.LINE_AA)

            if not self._positioned:
                cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
                self._place(canvas.shape[1], canvas.shape[0])
                self._positioned = True

            cv2.imshow(self.win, canvas)
            self._frames += 1
            if self.topmost and (self._frames <= 3 or self._frames % 60 == 0):
                _set_topmost(self.win)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                return "quit"
        except Exception as e:  # HUD не должен ронять вождение
            print(f"[hud] ошибка отрисовки (пропускаю кадр): {e}")
        return ""

    def _place(self, win_w, win_h, margin=10):
        sw, sh = _screen_size()
        if self.corner == "top-left":
            x, y = margin, margin
        elif self.corner == "bottom-right":
            x, y = sw - win_w - margin, sh - win_h - margin
        elif self.corner == "bottom-left":
            x, y = margin, sh - win_h - margin
        else:  # top-right
            x, y = sw - win_w - margin, margin
        try:
            cv2.moveWindow(self.win, max(0, x), max(0, y))
        except Exception:
            pass

    def close(self):
        if cv2 is None:
            return
        try:
            cv2.destroyWindow(self.win)
        except Exception:
            pass
