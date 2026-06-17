"""
Офлайн-подбор детектора стрелки игрока по дампу diag/mm_*.png.
Все кадры сняты при ходьбе ПРЯМО -> направление должно быть ПОСТОЯННЫМ.
Сравниваем методы и смотрим, какой даёт стабильный угол (малый разброс).
"""
from __future__ import annotations
import glob, math, os, sys
import cv2
import numpy as np

DIAG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diag")


def cyan_mask(hsv, lo, hi):
    return cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))


def biggest_blob(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 8:
        return None
    return c


def ang_farthest(c):
    """Текущий метод: центроид -> самая дальняя точка."""
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    gx, gy = M["m10"]/M["m00"], M["m01"]/M["m00"]
    pts = c.reshape(-1, 2).astype(np.float64)
    tip = pts[np.argmax(np.hypot(pts[:, 0]-gx, pts[:, 1]-gy))]
    return (math.degrees(math.atan2(tip[0]-gx, -(tip[1]-gy)))+360) % 360


def ang_pca(c):
    """PCA: главная ось + сторона вершины (дальше от центроида вдоль оси)."""
    pts = c.reshape(-1, 2).astype(np.float64)
    mean = pts.mean(axis=0)
    d = pts - mean
    cov = np.cov(d.T)
    w, v = np.linalg.eigh(cov)
    axis = v[:, np.argmax(w)]                 # главная ось (вектор)
    proj = d @ axis
    # вершина — экстремум проекции с большим |значением| (у треугольника вершина
    # дальше от центроида, чем база)
    tip = pts[np.argmax(np.abs(proj))]
    return (math.degrees(math.atan2(tip[0]-mean[0], -(tip[1]-mean[1])))+360) % 360


def run(lo, hi, roi_frac, label):
    files = sorted(glob.glob(os.path.join(DIAG, "mm_*.png")))
    far, pca, areas = [], [], []
    for f in files:
        img = cv2.imread(f)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = cyan_mask(hsv, lo, hi)
        h, w = m.shape
        roi = np.zeros_like(m)
        cv2.circle(roi, (w//2, h//2), int(min(w, h)*roi_frac), 255, -1)
        m = cv2.bitwise_and(m, roi)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        c = biggest_blob(m)
        if c is None:
            far.append(None); pca.append(None); areas.append(0); continue
        areas.append(int(cv2.contourArea(c)))
        far.append(ang_farthest(c)); pca.append(ang_pca(c))

    def stats(vals):
        v = [x for x in vals if x is not None]
        if len(v) < 2:
            return f"n={len(v)} —"
        # круговой разброс
        s = sum(math.sin(math.radians(x)) for x in v)/len(v)
        cc = sum(math.cos(math.radians(x)) for x in v)/len(v)
        R = math.hypot(s, cc)
        spread = math.degrees(math.sqrt(max(0, -2*math.log(max(1e-9, R)))))  # круговое стд
        mean = (math.degrees(math.atan2(s, cc))+360) % 360
        return f"n={len(v)} mean={mean:5.1f} spread={spread:5.1f}"

    print(f"[{label}] HSV{lo}-{hi} roi={roi_frac}")
    print(f"   area: min={min(areas)} max={max(areas)} | farthest: {stats(far)} | PCA: {stats(pca)}")
    return far, pca


if __name__ == "__main__":
    # текущие настройки
    run((85, 90, 140), (110, 255, 255), 0.10, "current")
    # шире зона + ярче порог (только сама стрелка, не конус)
    run((85, 90, 140), (110, 255, 255), 0.22, "wide-roi")
    run((80, 120, 180), (110, 255, 255), 0.22, "bright")
    run((80, 100, 200), (105, 255, 255), 0.25, "brighter")
    run((85, 140, 160), (105, 255, 255), 0.22, "sat+")
