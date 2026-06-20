#!/usr/bin/env python3
"""
compare_heading.py — сравнение двух детекторов курса по миникарте:

  A) ПРОДАКШН  — MinimapReader.read().heading_deg
                 (cyan-стрелка «центроид→дальняя точка» + круговая медиана окно 5)
  B) ТРЕКЕР    — MinimapTracker.detect()  (ось через минимум поперечной дисперсии
                 + разрешение 180° по дисперсии конуса FOV + continuity)

Цель — увидеть, который СТАБИЛЬНЕЕ и ПРАВИЛЬНЕЕ ловит курс, без доступа к
внутрянке игры. Скрипт НЕ трогает рабочий пайплайн — только читает.

Режимы:
  (по умолчанию) ОФЛАЙН по дампам:
      python scripts/compare_heading.py
      python scripts/compare_heading.py --glob "diag/mm_*.png" --cropped
      python scripts/compare_heading.py --glob "shots/*.png"          # полные кадры
  ЖИВОЙ оверлей на запущенной игре:
      python scripts/compare_heading.py --live
        q — выход, r — reset_polarity трекера, SPACE — пауза
        окно по умолчанию закрепляется сверху (always-on-top, Windows),
        чтобы не пропадало за игрой при переключении фокуса; --no-topmost
        отключает это поведение.

--cropped  : картинки уже вырезаны до миникарты (region = весь файл). Так лежат
             diag/mm_*.png (200x200). Для полных кадров (shots/) НЕ указывать —
             регион берётся из config.yaml.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Windows-консоль часто cp1251 — не ломаемся на utf-8 символах (°, кириллица).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from genshin_nav.config import Config                       # noqa: E402
from genshin_nav.minimap.minimap_reader import MinimapReader  # noqa: E402
from genshin_nav.minimap.marker_tracker import MinimapTracker  # noqa: E402


# ------------------------------------------------------------------- always-on-top
def set_window_topmost(window_name: str) -> bool:
    """
    Прижать окно OpenCV поверх всех остальных (в т.ч. поверх окна игры).

    Основной путь — РОДНОЕ свойство OpenCV WND_PROP_TOPMOST (есть в 4.5.2+):
    оно ставит окно поверх через собственный хэндл бэкенда HighGUI и работает
    надёжно, в отличие от поиска по заголовку. WinAPI-путь (FindWindowW +
    SetWindowPos) оставлен ФОЛБЭКОМ на старые сборки cv2 без этого свойства.
    Возвращает True, если хоть один путь сработал.
    """
    # 1) Родное свойство OpenCV — самый надёжный способ.
    try:
        if hasattr(cv2, "WND_PROP_TOPMOST"):
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1.0)
            return True
    except Exception:
        pass
    # 2) Фолбэк: WinAPI по заголовку окна (Windows only).
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
        if not hwnd:
            return False
        HWND_TOPMOST = -1
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_NOACTIVATE = 0x0010  # не отбираем фокус у игры
        ctypes.windll.user32.SetWindowPos(
            hwnd, HWND_TOPMOST, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        )
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- io
def imread_any(path: str):
    """Читать картинку, в т.ч. с кириллицей в пути (cv2.imread там возвращает None)."""
    img = cv2.imread(path)
    if img is not None:
        return img
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def ang_diff(a, b):
    """Кратчайшая разница углов a-b в (-180, 180]. None-безопасно."""
    if a is None or b is None:
        return None
    return (a - b + 180.0) % 360.0 - 180.0


def circ_median(vals):
    """Круговая медиана списка углов (элемент с мин суммой |разниц|)."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    best, best_s = vals[0], 1e18
    for x in vals:
        s = sum(abs(ang_diff(x, y)) for y in vals)
        if s < best_s:
            best_s, best = s, x
    return best


# --------------------------------------------------------------- build detectors
def build_detectors(cfg, region):
    """MinimapReader и MinimapTracker, настроенные на заданный region (l,t,w,h)."""
    mcfg = dataclasses.replace(cfg.minimap, region=tuple(region))
    reader = MinimapReader(mcfg)
    tracker = MinimapTracker.from_cfg(mcfg)
    return reader, tracker


# ------------------------------------------------------------------ offline mode
def run_offline(cfg, pattern, cropped):
    files = sorted(glob.glob(os.path.join(ROOT, pattern)))
    if not files:
        print(f"нет файлов по шаблону: {pattern}")
        return
    first = imread_any(files[0])
    if first is None:
        print(f"не читается: {files[0]}")
        return

    if cropped:
        h, w = first.shape[:2]
        region = (0, 0, w, h)
    else:
        region = cfg.minimap.region
    reader, tracker = build_detectors(cfg, region)

    print(f"файлов: {len(files)}   режим: {'CROPPED (файл=миникарта)' if cropped else 'полный кадр'}")
    print(f"регион миникарты: {region}")
    print(f"arrow HSV: {cfg.minimap.arrow_hsv_low}..{cfg.minimap.arrow_hsv_high}\n")
    print(f"{'#':>3}  {'файл':28s} {'ПРОД':>7} {'ТРЕКЕР':>7} {'conf':>5} "
          f"{'d(п-т)':>7} {'скачП':>6} {'скачТ':>6}")
    print("-" * 80)

    prod_seq, trk_seq, offsets = [], [], []
    prod_prev = trk_prev = None
    prod_spikes = trk_spikes = prod_none = trk_none = 0

    for i, f in enumerate(files):
        full = imread_any(f)
        if full is None:
            continue
        r = reader.read(full)
        prod = r.heading_deg
        trk, conf = tracker.detect(full)

        jp = ang_diff(prod, prod_prev)
        jt = ang_diff(trk, trk_prev)
        d = ang_diff(prod, trk)
        if d is not None:
            offsets.append(d)
        if prod is None:
            prod_none += 1
        elif jp is not None and abs(jp) > 70:
            prod_spikes += 1
        if trk is None:
            trk_none += 1
        elif jt is not None and abs(jt) > 70:
            trk_spikes += 1

        prod_seq.append(prod)
        trk_seq.append(trk)
        prod_prev = prod if prod is not None else prod_prev
        trk_prev = trk if trk is not None else trk_prev

        ps = f"{prod:7.1f}" if prod is not None else "   None"
        ts = f"{trk:7.1f}" if trk is not None else "   None"
        ds = f"{d:+7.1f}" if d is not None else "      -"
        jps = f"{jp:+6.1f}" if jp is not None else "     -"
        jts = f"{jt:+6.1f}" if jt is not None else "     -"
        print(f"{i:>3}  {os.path.basename(f):28.28s} {ps} {ts} {conf:5.2f} "
              f"{ds} {jps} {jts}")

    # ----- сводка
    def mean_abs_jump(seq):
        js = [abs(ang_diff(seq[k], seq[k-1]))
              for k in range(1, len(seq))
              if seq[k] is not None and seq[k-1] is not None]
        return (sum(js) / len(js)) if js else float("nan")

    off = circ_median(offsets)
    print("\n" + "=" * 80)
    print("СВОДКА")
    print(f"  кадров:                 {len(files)}")
    print(f"  потерь детекции:        ПРОД={prod_none}   ТРЕКЕР={trk_none}")
    print(f"  спайков >70°/кадр:      ПРОД={prod_spikes}   ТРЕКЕР={trk_spikes}"
          "   (меньше = стабильнее)")
    print(f"  ср.|скачок| межкадровый: ПРОД={mean_abs_jump(prod_seq):5.2f}°   "
          f"ТРЕКЕР={mean_abs_jump(trk_seq):5.2f}°   (меньше = глаже)")
    if off is not None:
        spread = circ_median([abs(ang_diff(o, off)) for o in offsets])
        print(f"  офсет конвенций (ПРОД-ТРЕКЕР): медиана={off:+.1f} deg  разброс~{spread:.1f} deg")
        print(f"    -> если разброс мал, источники в одной системе с поправкой {off:+.1f} deg.")
        print(f"    -> для продакшна-на-трекере поставь minimap.tracker_heading_offset_deg: {off:.1f}")
    print("=" * 80)
    print("Как читать: «стабильнее» = меньше спайков и меньше ср.|скачок|. Какой курс")
    print("ВЕРНЫЙ по существу — смотри глазами в --live (стрелка должна совпадать с")
    print("реальным направлением взгляда персонажа).")


# --------------------------------------------------------------------- live mode
def draw_compass_arrow(img, center, heading_deg, color, length, label=None):
    if heading_deg is None:
        return
    cx, cy = center
    dx = math.sin(math.radians(heading_deg))   # восток → +x
    dy = -math.cos(math.radians(heading_deg))  # север → -y
    ex, ey = int(cx + dx * length), int(cy + dy * length)
    cv2.arrowedLine(img, (int(cx), int(cy)), (ex, ey), color, 2, tipLength=0.25)
    if label:
        cv2.putText(img, label, (ex + 4, ey), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)


def get_screen_size():
    """Размер основного экрана (px). На Windows — через GetSystemMetrics,
    иначе разумный фолбэк 1920x1080 (просто чтобы было от чего отталкиваться)."""
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


def position_window(window_name: str, win_w: int, win_h: int, corner: str = "top-right",
                     margin: int = 10):
    """Передвинуть окно в угол экрана (по умолч. правый верх), без сворачивания игры."""
    screen_w, screen_h = get_screen_size()
    if corner == "top-right":
        x, y = screen_w - win_w - margin, margin
    elif corner == "top-left":
        x, y = margin, margin
    elif corner == "bottom-right":
        x, y = screen_w - win_w - margin, screen_h - win_h - margin
    elif corner == "bottom-left":
        x, y = margin, screen_h - win_h - margin
    else:
        x, y = margin, margin
    x, y = max(0, x), max(0, y)
    cv2.moveWindow(window_name, x, y)

def run_live(cfg, topmost=True, corner="top-right"):
    from genshin_nav.capture.screen_capture import ScreenCapture
    region = cfg.minimap.region
    reader, tracker = build_detectors(cfg, region)
    cap = ScreenCapture(cfg.capture.region, cfg.capture.backend,
                        cfg.capture.target_fps, cfg.capture.window_title)
    l, t, w, h = region
    SCALE = 3
    offsets = []
    paused = False
    print("LIVE: q — выход, r — reset трекера, SPACE — пауза")
    print("зелёная стрелка = ПРОДАКШН (стрелка+медиана), жёлтая = ТРЕКЕР")
    print("ВАЖНО: чтобы оверлей был ПОВЕРХ игры, Genshin должен идти в режиме")
    print("       «Оконный без рамки» (Borderless). Поверх ЭКСКЛЮЗИВНОГО полноэкранного")
    print("       никакое окно не показывается — это ограничение Windows, не скрипта.")

    WIN_NAME = "compare heading (PROD=green TRK=yellow)"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
    if corner:
        position_window(WIN_NAME, w * SCALE, h * SCALE, corner=corner)
    if topmost:
        ok = set_window_topmost(WIN_NAME)
        if ok:
            print("[topmost] окно закреплено сверху")
        elif sys.platform == "win32":
            print("[topmost] не удалось закрепить окно (FindWindowW не нашёл его —"
                  " попробуй ещё раз через --live, иногда окно появляется не сразу)")
        else:
            print("[topmost] поддерживается только на Windows, пропускаю")

    last = None
    frame_idx = 0
    while True:
        if not paused:
            frame = cap.grab()
            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            last = frame
        frame = last
        if frame is None:
            continue
        r = reader.read(frame)
        prod = r.heading_deg
        trk, conf = tracker.detect(frame)
        d = ang_diff(prod, trk)
        if d is not None:
            offsets.append(d)
            offsets[:] = offsets[-120:]

        mm = frame[t:t + h, l:l + w].copy()
        mm = cv2.resize(mm, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
        center = (w * SCALE // 2, h * SCALE // 2)
        draw_compass_arrow(mm, center, prod, (0, 255, 0), w * SCALE * 0.42, "PROD")
        draw_compass_arrow(mm, center, trk, (0, 255, 255), w * SCALE * 0.34, "TRK")

        off = circ_median(offsets)
        lines = [
            f"PROD={prod:6.1f}" if prod is not None else "PROD=  None",
            f"TRK ={trk:6.1f} ({conf:.2f})" if trk is not None else "TRK =  None",
            f"d(p-t)={d:+6.1f}" if d is not None else "d(p-t)=   -",
            f"offset~={off:+5.1f}" if off is not None else "offset~=  -",
        ]
        y = 18
        for ln in lines:
            cv2.putText(mm, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += 20

        cv2.imshow(WIN_NAME, mm)
        frame_idx += 1
        # WND_PROP_TOPMOST надёжно цепляется ТОЛЬКО после первого imshow (окно
        # реально создано бэкендом). Поэтому давим topmost на первых кадрах, а
        # дальше переустанавливаем периодически — игры/DWM при alt-tab/смене
        # фокуса сбрасывают флаг. Стоит копейки.
        if topmost and (frame_idx <= 3 or frame_idx % 60 == 0):
            set_window_topmost(WIN_NAME)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('r'):
            tracker.reset_polarity()
            print("[trk] reset_polarity")
        elif k == ord(' '):
            paused = not paused

    cap.close()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Сравнение детекторов курса миникарты")
    ap.add_argument("--live", action="store_true", help="живой оверлей на игре")
    ap.add_argument("--glob", default="diag/mm_*.png",
                    help="шаблон файлов для офлайн-режима (отн. корня проекта)")
    ap.add_argument("--cropped", action="store_true",
                    help="файлы уже вырезаны до миникарты (по умолч. для diag/mm_*)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-topmost", action="store_true",
                    help="не закреплять окно --live сверху (по умолч. закрепляется, Windows only)")
    ap.add_argument("--corner", default="top-right",
                    choices=["top-right", "top-left", "bottom-right", "bottom-left", "none"],
                    help="куда поставить окно --live (по умолч. top-right); none — не двигать")
    args = ap.parse_args()

    cfg = Config.load(os.path.join(ROOT, args.config))
    if args.live:
        corner = None if args.corner == "none" else args.corner
        run_live(cfg, topmost=not args.no_topmost, corner=corner)
    else:
        # diag/mm_* — это уже вырезанные миникарты: включаем cropped автоматически
        cropped = args.cropped or ("mm_" in args.glob and "diag" in args.glob)
        run_offline(cfg, args.glob, cropped)


if __name__ == "__main__":
    main()
