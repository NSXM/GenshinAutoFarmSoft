#!/usr/bin/env python3
"""
Шаг 1 ТЗ — АБСОЛЮТНАЯ позиция по миникарте (teach-and-repeat).

Миникарта Genshin СЕВЕР-ЗАЛОЧЕНА: один и тот же участок мира выглядит на ней
одинаково независимо от того, куда смотрит персонаж. Значит «отпечаток» миникарты
(кольцо карты без центрального маркера) однозначно привязан к МЕСТУ.

  * при ЗАПИСИ маршрута сохраняем отпечаток в каждой точке (record_route → .fp.npz);
  * при ПРОГОНЕ сравниваем живой отпечаток с отпечатками точек РЯДОМ и понимаем,
    где бот НА САМОМ ДЕЛЕ — без интегрирования скорости. Скорость/стамина/дрейф
    перестают влиять: позиция берётся из СОВПАДЕНИЯ картинки, а не из время×скорость.

Дёшево (CPU, мелкие 56×56 кропы), без GPU/атласа/сшивки.
"""
from __future__ import annotations

import os

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _ring_mask(size: int, inner: float, outer: float) -> np.ndarray:
    """Кольцевая маска: без центра (стрелка/конус) и без углов за кругом миникарты."""
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / (size / 2.0)
    return (r >= inner) & (r <= outer)


def make_fingerprint(minimap_bgr, size: int = 56, inner: float = 0.25,
                     outer: float = 0.92):
    """Нормированный отпечаток кропа миникарты (кольцо карты). float32 [size,size]
    (вне кольца — 0), нулевое среднее/единичное СКО по кольцу — для инвариантности
    к яркости. None, если cv2 нет или кроп пустой."""
    if cv2 is None or minimap_bgr is None or getattr(minimap_bgr, "size", 0) == 0:
        return None
    g = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.resize(g, (size, size))
    mask = _ring_mask(size, inner, outer)
    vals = g[mask]
    if vals.size < 10:
        return None
    g = (g - float(vals.mean())) / (float(vals.std()) + 1e-6)
    g[~mask] = 0.0
    return g.astype(np.float32)


class MinimapLocalizer:
    """Сопоставление живой миникарты с отпечатками точек маршрута."""

    def __init__(self, fingerprints, size: int = 56, inner: float = 0.25,
                 outer: float = 0.92):
        self.fps = fingerprints                       # float32 [N,size,size]
        self.size = size
        self.inner = inner
        self.outer = outer
        self._mask = _ring_mask(size, inner, outer)
        self._nmask = float(self._mask.sum())

    @property
    def n(self) -> int:
        return 0 if self.fps is None else int(len(self.fps))

    def score(self, fp_live, idx: int) -> float:
        """Корреляция (−1..1) живого отпечатка с отпечатком точки idx."""
        if fp_live is None or self.fps is None or not (0 <= idx < self.n):
            return -2.0
        fp = self.fps[idx]
        return float((fp_live * fp).sum() / self._nmask)

    def localize(self, minimap_bgr, lo: int, hi: int):
        """Лучшее совпадение среди точек [lo,hi). Возврат (idx, score); (-1,-2) если
        нечего сравнивать. Окно [lo,hi) ограничивает поиск точками РЯДОМ с текущей
        оценкой — чтобы похожие далёкие места не давали ложных скачков."""
        fp_live = make_fingerprint(minimap_bgr, self.size, self.inner, self.outer)
        if fp_live is None or self.fps is None:
            return -1, -2.0
        best, bests = -1, -2.0
        for i in range(max(0, lo), min(self.n, hi)):
            s = float((fp_live * self.fps[i]).sum() / self._nmask)
            if s > bests:
                bests, best = s, i
        return best, bests


# ------------------------------------------------------------------ файлы (.fp.npz)
def fp_path_for(route_path: str) -> str:
    """routes/x.json → routes/x.fp.npz (отпечатки лежат рядом с маршрутом)."""
    base, _ = os.path.splitext(route_path)
    return base + ".fp.npz"


def save_fingerprints(path: str, fps, size: int = 56, inner: float = 0.25,
                      outer: float = 0.92) -> int:
    """Сохранить список отпечатков (None → нулевой кадр). Возврат: сколько валидных."""
    if not fps:
        return 0
    arr = np.stack([f if f is not None else np.zeros((size, size), np.float32)
                    for f in fps]).astype(np.float32)
    valid = np.array([f is not None for f in fps], dtype=bool)
    np.savez_compressed(path, fps=arr, valid=valid,
                        size=size, inner=inner, outer=outer)
    return int(valid.sum())


def load_localizer(route_path: str):
    """Загрузить локализатор по пути МАРШРУТА (ищет рядом .fp.npz). None, если нет."""
    if cv2 is None:
        return None
    p = fp_path_for(route_path)
    if not os.path.exists(p):
        return None
    try:
        d = np.load(p)
        fps = d["fps"].astype(np.float32)
        return MinimapLocalizer(fps, int(d["size"]), float(d["inner"]), float(d["outer"]))
    except Exception as e:  # pragma: no cover
        print(f"[localize] не удалось загрузить {p}: {e}")
        return None
