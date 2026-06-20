"""
ЖИВОЙ пробник детектора стрелки — ловит и сохраняет кадры СРЫВА курса.

Зачем: на чистых дампах стрелка детектится идеально (spread 0.2°), но в поле курс
изредка прыгает на 100°+ (детектор путает стрелку с КОНУСОМ камеры / пином / синим
фоном-водой). Чтобы калибровать не вслепую, нужно собрать кадры именно тех моментов.

Что делает: на полной частоте читает миникарту и считает курс ТРЕМЯ способами —
farthest (как в проде), PCA, конус — плюс число голубых блобов и их площади. Когда
методы РАСХОДЯТСЯ (>thr) или курс ПРЫГАЕТ между кадрами (>jump), сохраняет
аннотированный кроп миникарты в diag/arrow_fail/ и пишет строку в CSV.

Просто походи/побегай по РАЗНЫМ местам (вода, трава, рядом с метками квестов, повороты
камеры). Потом разберём собранные срывы и подправим детектор.

Запуск:  .venv\\Scripts\\python.exe scripts\\probe_arrow.py
Выход:   F9  или Ctrl+C
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from genshin_nav.config import Config            # noqa: E402
from genshin_nav.capture.screen_capture import ScreenCapture  # noqa: E402


def ang_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def cyan_blobs(mm_bgr, lo, hi, roi_frac):
    """Все голубые блобы в центральной зоне: список (contour, area, centroid, dist_to_center)."""
    hsv = cv2.cvtColor(mm_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    h, w = mask.shape
    cx, cy = w // 2, h // 2
    r = max(8, int(min(w, h) * roi_frac))
    roi = np.zeros_like(mask)
    cv2.circle(roi, (cx, cy), r, 255, -1)
    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 30:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        gx, gy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        out.append((c, a, (gx, gy), math.hypot(gx - cx, gy - cy)))
    return out, mask


def head_farthest(c, gx, gy):
    pts = c.reshape(-1, 2).astype(np.float64)
    tip = pts[np.argmax(np.hypot(pts[:, 0] - gx, pts[:, 1] - gy))]
    return (math.degrees(math.atan2(tip[0] - gx, -(tip[1] - gy))) + 360) % 360


def head_pca(c):
    pts = c.reshape(-1, 2).astype(np.float64)
    mean = pts.mean(axis=0)
    d = pts - mean
    cov = np.cov(d.T)
    w, v = np.linalg.eigh(cov)
    axis = v[:, np.argmax(w)]
    proj = d @ axis
    tip = pts[np.argmax(np.abs(proj))]
    return (math.degrees(math.atan2(tip[0] - mean[0], -(tip[1] - mean[1])) ) + 360) % 360


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--disagree", type=float, default=20.0, help="порог расхождения методов, °")
    ap.add_argument("--jump", type=float, default=40.0, help="порог скачка курса между кадрами, °")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    m = cfg.minimap
    lo, hi, roi_frac = m.arrow_hsv_low, m.arrow_hsv_high, m.arrow_roi_frac
    l, t, w, h = m.region

    out_dir = os.path.join(ROOT, "diag", "arrow_fail")
    os.makedirs(out_dir, exist_ok=True)
    csv = open(os.path.join(ROOT, "diag", "arrow_probe.csv"), "w", encoding="utf-8")
    csv.write("t,far,pca,disagree,jump,nblobs,areas,saved\n")

    cap = ScreenCapture(cfg.capture.region, cfg.capture.backend,
                        cfg.capture.target_fps, cfg.capture.window_title)
    try:
        import keyboard
        have_kb = True
    except Exception:
        have_kb = False

    print(f"[probe] HSV {lo}-{hi} roi={roi_frac} регион миникарты=({l},{t},{w},{h})")
    print(f"[probe] срыв = методы расходятся >{args.disagree}° ИЛИ скачок >{args.jump}°")
    print("[probe] 3с на переключение в игру... ходи по РАЗНЫМ местам. F9 — стоп.")
    time.sleep(3)

    last_far = None
    n_saved = 0
    n_frames = 0
    t0 = time.monotonic()
    try:
        while True:
            frame = cap.grab()
            if frame is None:
                time.sleep(0.001)
                continue
            now = time.monotonic() - t0
            mm = frame[t:t + h, l:l + w]
            blobs, mask = cyan_blobs(mm, lo, hi, roi_frac)
            n_frames += 1

            far = pca = None
            if blobs:
                # как в проде: самый ЦЕНТРАЛЬНЫЙ блоб
                c, a, (gx, gy), dist = min(blobs, key=lambda b: b[3])
                far = head_farthest(c, gx, gy)
                pca = head_pca(c)

            dis = abs(ang_diff(far, pca)) if (far is not None and pca is not None) else 0.0
            jmp = abs(ang_diff(far, last_far)) if (far is not None and last_far is not None) else 0.0
            areas = "|".join(str(int(b[1])) for b in sorted(blobs, key=lambda b: -b[1]))

            saved = ""
            bad = (far is None) or (dis > args.disagree) or (jmp > args.jump)
            if bad and far is not None or (bad and len(blobs) >= 1):
                # аннотируем кроп и сохраняем
                vis = mm.copy()
                cx, cy = w // 2, h // 2
                for (c, a, (gx, gy), dist) in blobs:
                    cv2.drawContours(vis, [c], -1, (0, 255, 0), 1)
                    cv2.circle(vis, (int(gx), int(gy)), 2, (0, 0, 255), -1)
                if far is not None:
                    fa = math.radians(far)
                    cv2.arrowedLine(vis, (cx, cy),
                                    (int(cx + 40 * math.sin(fa)), int(cy - 40 * math.cos(fa))),
                                    (0, 255, 255), 2, tipLength=0.3)
                fname = f"fail_{n_saved:03d}_t{now:05.1f}_far{(-1 if far is None else int(far)):04d}_dis{int(dis)}_jmp{int(jmp)}.png"
                cv2.imwrite(os.path.join(out_dir, fname), vis)
                cv2.imwrite(os.path.join(out_dir, fname.replace('.png', '_mask.png')), mask)
                saved = fname
                n_saved += 1

            csv.write(f"{now:.2f},{'' if far is None else f'{far:.1f}'},"
                      f"{'' if pca is None else f'{pca:.1f}'},{dis:.1f},{jmp:.1f},"
                      f"{len(blobs)},{areas},{saved}\n")
            csv.flush()
            if far is not None:
                last_far = far

            if n_frames % 30 == 0:
                print(f"[probe] t={now:5.1f}s кадров={n_frames} срывов_сохранено={n_saved} "
                      f"| far={('--' if far is None else f'{far:5.1f}')} "
                      f"pca={('--' if pca is None else f'{pca:5.1f}')} "
                      f"dis={dis:4.1f} blobs={len(blobs)}({areas})")

            if have_kb and keyboard.is_pressed("f9"):
                print("[probe] F9 — стоп")
                break
    except KeyboardInterrupt:
        print("\n[probe] прервано")
    finally:
        cap.close()
        csv.close()
        print(f"\n[probe] кадров={n_frames}, срывов сохранено={n_saved} -> {out_dir}")
        print("[probe] пришли несколько fail_*.png — посмотрим, на что срывается детектор")


if __name__ == "__main__":
    main()
