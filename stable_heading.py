"""
stable_heading.py — замена detect_angle для calibrate_arrow.py и MinimapReader.

Три слоя защиты против срывов:
  1. ROI-круг + морфология — те же, что в оригинале
  2. PCA-ось вместо "дальней точки" — стабильная ось треугольника стрелки
  3. Disambiguation (нос vs хвост) по ШИРИНЕ маски поперёк оси:
       нос треугольника уже → меньше пикселей маски на 0.55r впереди
  4. HeadingFilter — outlier rejection (спайки > spike_threshold°) + EMA

Замена 1-в-1:
  calibrate_arrow.py   → заменить блок detect_angle
  MinimapReader        → self._hf = HeadingFilter(); return self._hf.update(raw)
"""
from __future__ import annotations
import math
import cv2
import numpy as np

ROI_FRAC = 0.22   # совпадает с calibrate_arrow.py


# ──────────────────────────────────────────────────────────────
# Вспомогательная: ширина маски поперёк оси
# ──────────────────────────────────────────────────────────────

def _mask_width(direction: np.ndarray,
                mask: np.ndarray,
                gx: float, gy: float,
                r: int,
                frac: float = 0.55,
                half_span: int = 10) -> int:
    """Число ненулевых пикселей маски в поперечной полосе на frac*r вдоль direction."""
    perp = np.array([-direction[1], direction[0]])
    px0 = gx + direction[0] * r * frac
    py0 = gy + direction[1] * r * frac
    h_, w_ = mask.shape
    total = 0
    for s in range(-half_span, half_span + 1):
        px = int(round(px0 + perp[0] * s))
        py = int(round(py0 + perp[1] * s))
        if 0 <= px < w_ and 0 <= py < h_:
            total += int(mask[py, px]) // 255
    return total


# ──────────────────────────────────────────────────────────────
# Основная функция детектирования угла
# ──────────────────────────────────────────────────────────────

def detect_angle(mm_bgr: np.ndarray,
                 lo: list[int],
                 hi: list[int]) -> tuple[float | None, np.ndarray, np.ndarray | None]:
    """
    Определить курс стрелки через PCA-ось + disambiguation по ширине.

    Параметры
    ---------
    mm_bgr : BGR-кроп миникарты
    lo, hi : HSV-пороги [H, S, V]  (H: 0-179, S/V: 0-255)

    Возвращает
    ----------
    (angle_deg, mask, best_contour)
      angle_deg — курс 0-360 (север=0, по часовой), None если не найдено
      mask      — бинарная маска (для отладки в calibrate_arrow)
      best_contour — numpy-контур или None
    """
    hsv  = cv2.cvtColor(mm_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

    h, w = mask.shape
    cx, cy = w // 2, h // 2
    r = max(8, int(min(w, h) * ROI_FRAC))

    # ROI-круг
    roi = np.zeros_like(mask)
    cv2.circle(roi, (cx, cy), r, 255, -1)
    mask = cv2.bitwise_and(mask, roi)

    # Морфология
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ближайший к центру контур достаточного размера
    best, best_d = None, 1e9
    for c in cnts:
        if cv2.contourArea(c) < 30:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        gx = M["m10"] / M["m00"]
        gy = M["m01"] / M["m00"]
        d  = math.hypot(gx - cx, gy - cy)
        if d < best_d:
            best, best_d = c, d

    if best is None or best_d > r:
        return None, mask, None

    # PCA-ось
    M   = cv2.moments(best)
    gx  = M["m10"] / M["m00"]
    gy  = M["m01"] / M["m00"]
    pts = best.reshape(-1, 2).astype(np.float64)
    centered = pts - np.array([gx, gy])
    cov      = centered.T @ centered / max(len(centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]

    # Disambiguation: нос — сторона с меньшей шириной маски (острее)
    w_pos = _mask_width( axis, mask, gx, gy, r)
    w_neg = _mask_width(-axis, mask, gx, gy, r)
    nose_dir = axis if w_pos <= w_neg else -axis

    # Компасный угол: север=0, по часовой
    ang = (math.degrees(math.atan2(nose_dir[0], -nose_dir[1])) + 360) % 360

    return ang, mask, best


# ──────────────────────────────────────────────────────────────
# Фильтр курса с outlier rejection + EMA
# ──────────────────────────────────────────────────────────────

class HeadingFilter:
    """
    Подавляет спайки курса в два этапа:
      1. Outlier rejection: если новый угол отличается от текущей EMA
         больше чем на spike_threshold°, кадр отбрасывается.
      2. EMA: экспоненциальное скользящее среднее с угловой арифметикой.

    Параметры
    ---------
    ema_alpha       : скорость реакции EMA (0-1). 0.4 = умеренно быстро.
    spike_threshold : максимально допустимый прыжок за кадр, градусы.
                      60° — подходит для пешего хода; уменьшите до 40° если
                      игрок не может резко поворачиваться между кадрами.

    Пример в MinimapReader.read_heading:
        if not hasattr(self, '_hf'):
            self._hf = HeadingFilter()
        raw, mask, cnt = detect_angle(mm, lo, hi)
        return self._hf.update(raw)
    """

    def __init__(self, ema_alpha: float = 0.40, spike_threshold: float = 60.0):
        self._ema:    float | None = None
        self._alpha   = ema_alpha
        self._thresh  = spike_threshold

    @staticmethod
    def _adiff(a: float, b: float) -> float:
        """Кратчайшая угловая разница в (-180, 180]."""
        return (a - b + 180) % 360 - 180

    def update(self, raw: float | None) -> float | None:
        """
        raw=None  → вернуть последнее EMA (стрелка не найдена в кадре).
        raw=угол  → применить outlier rejection + EMA, вернуть результат.
        """
        if raw is None:
            return self._ema

        raw = raw % 360

        # Первый кадр — инициализация без фильтрации
        if self._ema is None:
            self._ema = raw
            return self._ema

        # Outlier rejection
        diff = self._adiff(raw, self._ema)
        if abs(diff) > self._thresh:
            return self._ema   # спайк — игнорируем, держим старое

        # EMA
        self._ema = (self._ema + self._alpha * diff + 360) % 360
        return self._ema

    def reset(self):
        self._ema = None


# ──────────────────────────────────────────────────────────────
# Самотест:  python stable_heading.py
# ──────────────────────────────────────────────────────────────

def _ang_close(a: float, b: float, tol: float = 10.0) -> bool:
    return min(abs(a - b), 360 - abs(a - b)) < tol


if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("Тест 1: стрелка на север (ожидается ~0°)")
    size = 80
    img  = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    cv2.fillPoly(img, [np.array([(cx, cy-22), (cx-9, cy+12), (cx+9, cy+12)])],
                 color=(255, 255, 0))
    lo, hi = [80, 180, 180], [100, 255, 255]
    ang, _, _ = detect_angle(img, lo, hi)
    ok1 = ang is not None and _ang_close(ang, 0.0)
    print(f"  {'✓' if ok1 else '✗'} angle={ang:.1f}°  ({'OK' if ok1 else 'ОШИБКА'})")

    print("Тест 2: стрелка на восток (ожидается ~90°)")
    img2 = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.fillPoly(img2, [np.array([(cx+22, cy), (cx-12, cy-9), (cx-12, cy+9)])],
                 color=(255, 255, 0))
    ang2, _, _ = detect_angle(img2, lo, hi)
    ok2 = ang2 is not None and _ang_close(ang2, 90.0)
    print(f"  {'✓' if ok2 else '✗'} angle={ang2:.1f}°  ({'OK' if ok2 else 'ОШИБКА'})")

    print("Тест 3: стрелка на юг (ожидается ~180°)")
    img3 = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.fillPoly(img3, [np.array([(cx, cy+22), (cx-9, cy-12), (cx+9, cy-12)])],
                 color=(255, 255, 0))
    ang3, _, _ = detect_angle(img3, lo, hi)
    ok3 = ang3 is not None and _ang_close(ang3, 180.0)
    print(f"  {'✓' if ok3 else '✗'} angle={ang3:.1f}°  ({'OK' if ok3 else 'ОШИБКА'})")

    print("=" * 55)
    print("Тест 4: HeadingFilter подавляет спайк 147°")
    hf = HeadingFilter(ema_alpha=0.40, spike_threshold=60.0)
    series = [355.0, 2.0, 358.0, 147.0, 1.0, 3.0]
    prev = None
    for r_val in series:
        filtered = hf.update(r_val)
        spike = "  ← СПАЙК (ожидается: остаться ~358°)" if abs(r_val - 147) < 5 else ""
        print(f"  raw={r_val:>6.1f}°  →  {filtered:>6.1f}°{spike}")
        prev = filtered
    # После спайка фильтр должен продолжить с ~358-5° (EMA из 355,2,358 + 1,3)
    expected_final = hf.update(None)
    final_ok = expected_final is not None and _ang_close(expected_final, 358.0, tol=20.0)
    print(f"  {'✓' if final_ok else '✗'} Финальный (без нового кадра): {expected_final:.1f}°")

    all_ok = ok1 and ok2 and ok3 and final_ok
    print("=" * 55)
    print(f"Итог: {'ВСЕ ТЕСТЫ ПРОШЛИ ✓' if all_ok else 'ЕСТЬ ОШИБКИ — проверь HSV параметры'}")
    sys.exit(0 if all_ok else 1)
