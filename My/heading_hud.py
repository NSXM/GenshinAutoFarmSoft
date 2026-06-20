"""
heading_hud.py — полупрозрачный HUD-оверлей поверх игры.

Показывает:
  • Большую стрелку компаса (вращается в реальном времени)
  • Текущий курс в градусах
  • Мини-компас с метками С/В/Ю/З
  • Статус детектора (OK / NO SIGNAL)
  • Сырой vs сглаженный угол (для отладки)

Запуск:
  .venv\\Scripts\\python.exe scripts\\heading_hud.py

Клавиши (фокус на HUD-окне):
  Q — выход
  F — сбросить HeadingFilter (при застревании)
  H — скрыть/показать подробности

Зависимости: те же, что у calibrate_arrow.py + stable_heading.py в PATH/рядом.
"""
from __future__ import annotations
import math
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Подключаем наши модули
from genshin_nav.config import Config
from genshin_nav.capture.screen_capture import ScreenCapture

# stable_heading лежит рядом со скриптом или в scripts/
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from stable_heading import detect_angle, HeadingFilter  # noqa: E402

# ── Настройки HUD ──────────────────────────────────────────────
WIN_W, WIN_H   = 260, 300      # размер HUD-окна
WIN_TITLE      = "Heading HUD"
HUD_POS        = (20, 80)      # позиция окна (x, y) на экране
BG_COLOR       = (18, 18, 18)  # почти чёрный фон
ALPHA          = 0.88          # непрозрачность (только визуально через цвет)

# Цвета (BGR)
C_ARROW   = (0, 230, 255)   # циан — стрелка
C_NORTH   = (60, 100, 255)  # красноватый — метка севера
C_GRID    = (55, 55, 55)    # тёмно-серый — сетка
C_TEXT    = (210, 210, 210) # светло-серый — основной текст
C_DIM     = (90, 90, 90)    # тёмный текст (вторичный)
C_OK      = (60, 200, 90)   # зелёный — статус OK
C_ERR     = (60, 60, 220)   # красный — NO SIGNAL
C_SMOOTH  = (200, 170, 60)  # голубой — сглаженный угол
C_RAW     = (80, 80, 150)   # серый — сырой угол

ROI_FRAC = 0.22   # совпадает с calibrate_arrow / stable_heading


def draw_compass_rose(canvas: np.ndarray, cx: int, cy: int, r: int,
                      heading_deg: float | None, raw_deg: float | None) -> None:
    """Рисует компасную розу с вращающейся стрелкой."""

    # Внешнее кольцо
    cv2.circle(canvas, (cx, cy), r, C_GRID, 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r + 4, C_GRID, 1, cv2.LINE_AA)

    # Тиковые метки (каждые 30°)
    for deg in range(0, 360, 30):
        rad = math.radians(deg)
        sin_r, cos_r = math.sin(rad), math.cos(rad)
        is_cardinal = deg % 90 == 0
        r_in  = r - (8 if is_cardinal else 4)
        r_out = r
        x1 = int(cx + sin_r * r_in)
        y1 = int(cy - cos_r * r_in)
        x2 = int(cx + sin_r * r_out)
        y2 = int(cy - cos_r * r_out)
        color = C_NORTH if deg == 0 else (C_TEXT if is_cardinal else C_GRID)
        cv2.line(canvas, (x1, y1), (x2, y2), color, 2 if is_cardinal else 1, cv2.LINE_AA)

    # Метки сторон света
    labels = {0: "С", 90: "В", 180: "Ю", 270: "З"}
    for deg, label in labels.items():
        rad = math.radians(deg)
        lx = int(cx + math.sin(rad) * (r + 16))
        ly = int(cy - math.cos(rad) * (r + 16))
        color = C_NORTH if deg == 0 else C_TEXT
        cv2.putText(canvas, label, (lx - 6, ly + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Сырой угол (тонкая серая линия)
    if raw_deg is not None:
        rad_raw = math.radians(raw_deg)
        rx2 = int(cx + math.sin(rad_raw) * (r - 6))
        ry2 = int(cy - math.cos(rad_raw) * (r - 6))
        cv2.line(canvas, (cx, cy), (rx2, ry2), C_RAW, 1, cv2.LINE_AA)

    # Главная стрелка (сглаженный угол)
    if heading_deg is not None:
        rad = math.radians(heading_deg)
        sin_r, cos_r = math.sin(rad), math.cos(rad)

        # Хвост
        tail_x = int(cx - sin_r * (r - 18))
        tail_y = int(cy + cos_r * (r - 18))

        # Нос
        tip_x = int(cx + sin_r * (r - 8))
        tip_y = int(cy - cos_r * (r - 8))

        # Тело стрелки
        cv2.line(canvas, (tail_x, tail_y), (tip_x, tip_y), C_ARROW, 3, cv2.LINE_AA)

        # Наконечник (треугольник)
        perp_x = -cos_r
        perp_y = -sin_r
        wing = 9
        tip_back_x = int(cx + sin_r * (r - 22))
        tip_back_y = int(cy - cos_r * (r - 22))
        w1 = (int(tip_back_x + perp_x * wing), int(tip_back_y + perp_y * wing))
        w2 = (int(tip_back_x - perp_x * wing), int(tip_back_y - perp_y * wing))
        pts = np.array([[tip_x, tip_y], w1, w2], np.int32)
        cv2.fillPoly(canvas, [pts], C_ARROW)
        cv2.polylines(canvas, [pts], True, C_ARROW, 1, cv2.LINE_AA)

        # Хвостовой маркер (короткая поперечная черта)
        hx1 = int(tail_x + perp_x * 5)
        hy1 = int(tail_y + perp_y * 5)
        hx2 = int(tail_x - perp_x * 5)
        hy2 = int(tail_y - perp_y * 5)
        cv2.line(canvas, (hx1, hy1), (hx2, hy2), C_ARROW, 2, cv2.LINE_AA)

    # Центральная точка
    cv2.circle(canvas, (cx, cy), 4, C_ARROW, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 4, BG_COLOR, 1, cv2.LINE_AA)


def make_hud_frame(heading: float | None, raw: float | None,
                   fps: float, show_details: bool) -> np.ndarray:
    """Собрать один кадр HUD."""
    canvas = np.full((WIN_H, WIN_W, 3), BG_COLOR, dtype=np.uint8)

    # Заголовок
    cv2.putText(canvas, "HEADING HUD", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_DIM, 1, cv2.LINE_AA)
    cv2.line(canvas, (10, 28), (WIN_W - 10, 28), C_GRID, 1)

    # Компасная роза
    rose_cx, rose_cy, rose_r = WIN_W // 2, 130, 75
    draw_compass_rose(canvas, rose_cx, rose_cy, rose_r, heading, raw)

    # Большое число курса
    if heading is not None:
        deg_str  = f"{heading:05.1f}°"
        cv2.putText(canvas, deg_str, (WIN_W // 2 - 52, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_ARROW, 2, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "---.-°", (WIN_W // 2 - 48, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_DIM, 2, cv2.LINE_AA)

    # Статус
    if heading is not None:
        cv2.putText(canvas, "● OK", (10, WIN_H - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_OK, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "● NO SIGNAL", (10, WIN_H - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_ERR, 1, cv2.LINE_AA)

    # FPS
    cv2.putText(canvas, f"{fps:.0f} fps", (WIN_W - 58, WIN_H - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_DIM, 1, cv2.LINE_AA)

    # Детали (raw vs smooth)
    if show_details and raw is not None:
        cv2.putText(canvas, f"raw:    {raw:06.1f}°", (10, WIN_H - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_RAW, 1, cv2.LINE_AA)
        if heading is not None:
            cv2.putText(canvas, f"smooth: {heading:06.1f}°", (10, WIN_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_SMOOTH, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "H — детали  Q — выход  F — сброс",
                    (10, WIN_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_DIM, 1, cv2.LINE_AA)

    # Рамка окна
    cv2.rectangle(canvas, (0, 0), (WIN_W - 1, WIN_H - 1), C_GRID, 1)

    return canvas


def main() -> None:
    cfg = Config.load(os.path.join(ROOT, "config.yaml"))
    l, t, w, h = cfg.minimap.region
    lo = list(cfg.minimap.arrow_hsv_low)
    hi = list(cfg.minimap.arrow_hsv_high)

    cap = ScreenCapture(cfg.capture.region, cfg.capture.backend,
                        cfg.capture.target_fps, cfg.capture.window_title)

    hf           = HeadingFilter(ema_alpha=0.40, spike_threshold=60.0)
    show_details = False
    last_raw: float | None     = None
    last_smooth: float | None  = None

    # FPS-счётчик
    fps_buf:  list[float] = []
    t_last    = time.perf_counter()
    fps_val   = 0.0

    cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_TITLE, WIN_W, WIN_H)
    try:
        cv2.setWindowProperty(WIN_TITLE, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    cv2.moveWindow(WIN_TITLE, *HUD_POS)

    print("[HUD] запущен. Q — выход, F — сброс фильтра, H — детали.")

    positioned = False
    try:
        while True:
            frame = cap.grab()
            if frame is None:
                cv2.waitKey(1)
                continue

            mm = frame[t:t + h, l:l + w]
            raw, mask, _ = detect_angle(mm, lo, hi)
            smooth = hf.update(raw)

            last_raw    = raw
            last_smooth = smooth

            # FPS
            now = time.perf_counter()
            fps_buf.append(1.0 / max(now - t_last, 1e-6))
            t_last = now
            if len(fps_buf) > 20:
                fps_buf.pop(0)
            fps_val = sum(fps_buf) / len(fps_buf)

            # Рендер HUD
            hud = make_hud_frame(last_smooth, last_raw, fps_val, show_details)
            cv2.imshow(WIN_TITLE, hud)

            if not positioned:
                cv2.moveWindow(WIN_TITLE, *HUD_POS)
                positioned = True

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), ord('Q')):
                break
            elif k in (ord('f'), ord('F')):
                hf.reset()
                print("[HUD] HeadingFilter сброшен.")
            elif k in (ord('h'), ord('H')):
                show_details = not show_details

    finally:
        cap.close()
        cv2.destroyAllWindows()
        print("[HUD] завершён.")


if __name__ == "__main__":
    main()
