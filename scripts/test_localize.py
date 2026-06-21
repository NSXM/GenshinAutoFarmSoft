"""
test_localize.py — офлайн-проверка отпечатков миникарты (БЕЗ игры).

Берёт последовательность кропов миникарты (diag/mm_*.png — это кадры движения) и
считает, насколько отпечатки РАЗЛИЧАЮТ позицию: корреляция кадра с самим собой
(=1), с соседями (высокая) и с далёкими (должна падать). Если падает с расстоянием —
teach-and-repeat локализация рабочая; насколько быстро падает — это и есть её
разрешение на твоём зуме миникарты.

    .venv\\Scripts\\python.exe scripts\\test_localize.py
    .venv\\Scripts\\python.exe scripts\\test_localize.py --glob "diag/mm_*.png"
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2  # noqa: E402
from genshin_nav.minimap.localizer import make_fingerprint  # noqa: E402


def imread_any(path):
    img = cv2.imread(path)
    if img is not None:
        return img
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def corr(a, b):
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="diag/mm_*.png")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(ROOT, args.glob)))
    fps = []
    for f in files:
        im = imread_any(f)
        fp = make_fingerprint(im) if im is not None else None
        if fp is not None:
            fps.append(fp)
    n = len(fps)
    if n < 3:
        print(f"мало кадров ({n}) по {args.glob}")
        return
    print(f"кадров: {n}\n")
    print(f"{'#':>3} {'self':>5} {'±1':>6} {'best_other':>12}")
    print("-" * 32)
    for i in range(n):
        s = [corr(fps[i], fps[j]) for j in range(n)]
        neigh = [s[j] for j in (i - 1, i + 1) if 0 <= j < n]
        navg = sum(neigh) / len(neigh)
        bo = max((v, j) for j, v in enumerate(s) if j != i)
        print(f"{i:>3} {s[i]:5.2f} {navg:6.2f}   #{bo[1]:<3}({bo[0]:.2f})")

    # средняя корреляция по расстоянию между кадрами
    print("\nсредняя корреляция vs расстояние между кадрами:")
    for d in (1, 2, 3, 5, 8, 12):
        vals = [corr(fps[i], fps[i + d]) for i in range(n - d)]
        if vals:
            print(f"  d={d:2d}: {sum(vals)/len(vals):+.2f}")
    print("\nХорошо, если: self≈1.00, ±1 высокая (>0.7), и корреляция ПАДАЕТ с d.")
    print("Если падает медленно — разрешение грубое (соседние точки почти одинаковы);")
    print("локализация поймает крупный дрейф, но не метровую точность — это норм для шага 1.")


if __name__ == "__main__":
    main()
