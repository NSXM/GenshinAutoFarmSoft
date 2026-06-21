"""
build_atlas.py — сборка СШИТОГО АТЛАСА локации из живых кадров миникарты.

ЗАЧЕМ: MinimapReader.localize() и FusionEstimator (estimator.py) уже умеют
принимать атлас и давать АБСОЛЮТНУЮ позицию (world_xy_m) через matchTemplate —
но самого файла атласа в проекте никогда не было (world_atlas_path: null).
Без него позиция = чистая одометрия (сумма delta_xy_m), которая на длинных
дистанциях/нестабильных условиях (стамина, рельеф, спайки детектора) НАКАПЛИВАЕТ
дрейф. Атлас даёт боту внешний, не дрейфующий ориентир «где я на самом деле».

КАК ЭТО РАБОТАЕТ:
  1. Ты бегаешь (или ведёшь бота dry-run) по всей площади будущих маршрутов,
     стараясь покрыть её с запасом по краям (зигзагом, не только по линии
     маршрута — иначе при отклонении бота край атласа окажется обрезан).
  2. Скрипт на каждом кадре читает кроп миникарты, копит ОТНОСИТЕЛЬНУЮ позицию
     через тот же relative_shift()/phaseCorrelate, что и обычная одометрия
     (MinimapReader) — на КОРОТКОЙ дистанции одного прохода она точна, дрейф
     успевает накопиться только на ДЛИННЫХ маршрутах (наша исходная проблема),
     а сбор атласа — это один компактный проход.
  3. Каждый кадр кладём на большой холст ПО ЭТОЙ позиции (просто vanilla
     максимум/последний-выигрывает, без сложной feature-сшивки — нам не нужна
     идеальная склейка пиксель-в-пиксель, matchTemplate потом ищет лучшее
     совпадение локально и переживёт небольшие швы).
  4. В конце сохраняем результат в PNG (grayscale, как ждёт MinimapReader.load_atlas)
     и печатаем, что вписать в config.yaml (world_atlas_path, atlas_meters_per_px).

ТУМАН: миникарта без открытой статуи Архонта показывает только крупные формы
рельефа (берег/обрывы) без мелкой детализации — это OK для сборки атласа прямо
сейчас (см. обсуждение). Если позже откроешь статую — текстура изменится и
атлас придётся пересобрать этим же скриптом.

ВАЖНО — динамические элементы миникарты (квест-маркеры „!“, ромбики предметов,
NPC, конус камеры): они МОГУТ портить как сшивку, так и последующий matchTemplate,
если случайно оказались на снятом кадре, а потом исчезли/сместились в реальности.
Сама cyan-стрелка и конус камеры уже вырезаны через ту же кольцевую маску, что в
MinimapReader (центр + углы квадрата). Прочие маркеры (квесты/предметы) НЕ
маскируются автоматически — старайся собирать атлас в месте/время, где их не
видно в этой области, либо переснимай при появлении.

ЗАПУСК (открытый мир, проход всей площади будущих маршрутов, зигзагом):
    .venv\\Scripts\\python.exe scripts\\build_atlas.py --out atlas_garden.png
    (остановка: F9 — РАБОТАЕТ ВСЕГДА, независимо от того, какое окно в фокусе;
     Q в окне предпросмотра и Ctrl+C в консоли тоже работают, но только если
     фокус именно на этом окне — если игру свернул, фокус может оказаться нигде)

Дальше впиши в config.yaml (секция minimap):
    world_atlas_path: atlas_garden.png
    atlas_meters_per_px: <число, которое скрипт распечатает>
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

from genshin_nav.config import Config                          # noqa: E402
from genshin_nav.capture.screen_capture import ScreenCapture    # noqa: E402
from genshin_nav.minimap.minimap_reader import MinimapReader    # noqa: E402


# ------------------------------------------------------------------- always-on-top
def set_window_topmost(window_name: str) -> bool:
    """Прижать окно OpenCV сверху всех остальных (Windows only, через WinAPI),
    без отбора фокуса у игры. Та же утилита, что в compare_heading.py."""
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
        SWP_NOACTIVATE = 0x0010
        ctypes.windll.user32.SetWindowPos(
            hwnd, HWND_TOPMOST, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        )
        return True
    except Exception:
        return False


def get_screen_size():
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
    """Передвинуть окно в угол экрана (по умолч. правый верх) — не перекрывает
    миникарту игры и не заставляет сворачивать игру, чтобы добраться до окна."""
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


def main():
    ap = argparse.ArgumentParser(description="Сборка атласа локации из живых кадров миникарты")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="atlas.png", help="куда сохранить готовый атлас (PNG, grayscale)")
    ap.add_argument("--canvas-margin-m", type=float, default=40.0,
                    help="запас холста за пределами пройденного пути, метры (на случай "
                         "недооценки границ при первом проходе)")
    ap.add_argument("--min-move-px", type=float, default=0.04,
                    help="игнорировать сравнения с микро-сдвигом меньше этого (px) — должен "
                         "отсекать шум СТОЯНИЯ (~0px), а НЕ реальное движение. Раньше стоял "
                         "0.3, что на ходьбе ШАГОМ (без спринта) при sample_dt=0.15с давало "
                         "сдвиг ~0.08px — порог 0.3 резал почти весь шаговый путь, оставляя "
                         "только спринт. 0.04 — компромисс, проверь сам по итоговому % "
                         "в отчёте (должно быть >70-80%, а не 20%)")
    ap.add_argument("--sample-dt", type=float, default=None,
                    help="мин. интервал между кадрами, которые СРАВНИВАЕМ для сдвига (сек). "
                         "По умолч. = cfg.minimap.heading_sample_dt (та же логика разреживания, "
                         "что в остальном проекте — на 60 FPS сдвиг между СОСЕДНИМИ кадрами "
                         "субпиксельный и тонет в шуме, нужно копить интервал побольше)")
    ap.add_argument("--no-topmost", action="store_true",
                    help="не закреплять окно предпросмотра сверху (по умолч. закрепляется, Windows only)")
    ap.add_argument("--corner", default="top-right",
                    choices=["top-right", "top-left", "bottom-right", "bottom-left", "none"],
                    help="куда поставить окно предпросмотра (по умолч. top-right); "
                         "none — не двигать (тогда придётся сворачивать игру самому)")
    args = ap.parse_args()

    cfg = Config.load(os.path.join(ROOT, args.config))
    mm_cfg = cfg.minimap
    l, t, w, h = mm_cfg.region
    mpp = mm_cfg.minimap_meters_per_px   # тот же масштаб, что у обычной одометрии
    sample_dt = args.sample_dt if args.sample_dt is not None else mm_cfg.heading_sample_dt

    cap = ScreenCapture(cfg.capture.region, cfg.capture.backend,
                        cfg.capture.target_fps, cfg.capture.window_title)
    # Свой MinimapReader без атласа — нам нужен ТОЛЬКО relative_shift(), не localize().
    reader = MinimapReader(mm_cfg, atlas=None)

    # Холст в пикселях атласа: 1px атласа = 1px миникарты (mpp одинаков для обоих
    # по построению — берём то же minimap_meters_per_px). Размер — с запасом,
    # позиция персонажа стартует в ЦЕНТРЕ холста (т.к. в обе стороны можем уйти).
    margin_px = int(args.canvas_margin_m / mpp)
    canvas_size = 2 * margin_px + max(w, h) * 20   # грубая верхняя оценка площади прохода
    canvas_size = min(canvas_size, 8000)            # не раздуваем память безмерно
    canvas = np.zeros((canvas_size, canvas_size), np.uint8)
    covered = np.zeros((canvas_size, canvas_size), np.uint8)  # где уже что-то легло (для отчёта)
    cx0, cy0 = canvas_size // 2, canvas_size // 2   # старт = центр холста (в пикселях)

    print(f"[atlas] регион миникарты={mm_cfg.region}, mpp={mpp:.4f}, sample_dt={sample_dt:.3f}с")
    print(f"[atlas] холст {canvas_size}x{canvas_size}px (запас {args.canvas_margin_m:.0f}м)")
    print("[atlas] 3 секунды на переключение в игру. Затем ходи ЗИГЗАГОМ по всей площади "
          "будущих маршрутов, с запасом по краям. F9 — закончить (работает всегда, из "
          "любого окна); Q в окне предпросмотра — тоже работает, если фокус на нём.")
    time.sleep(3)

    pos_px = [float(cx0), float(cy0)]   # накопленная позиция в пикселях ХОЛСТА
    n_frames = 0
    n_used = 0
    n_compared = 0           # сколько раз реально вызвали relative_shift (раз в sample_dt)
    last_sample_t = 0.0
    win = "build_atlas (Q=finish)"
    PREVIEW_SIZE = 600
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, PREVIEW_SIZE, PREVIEW_SIZE)
    corner = None if args.corner == "none" else args.corner
    if corner:
        position_window(win, PREVIEW_SIZE, PREVIEW_SIZE, corner=corner)
    if not args.no_topmost:
        ok = set_window_topmost(win)
        if ok:
            print("[atlas] окно предпросмотра закреплено сверху — игру сворачивать не нужно")
        elif sys.platform == "win32":
            print("[atlas] не удалось закрепить окно сверху (попробуй перезапустить)")

    # ГЛОБАЛЬНАЯ остановка F9 — работает НЕЗАВИСИМО от того, какое окно сейчас в
    # фокусе (игра/предпросмотр/рабочий стол). Q в окне предпросмотра ловится
    # только когда фокус именно на нём — если свернул игру и не кликнул по окну
    # предпросмотра, фокус может оказаться нигде, и ни Q, ни Ctrl+C не дойдут.
    # F9 (тот же хоткей, что в run_route.py) решает это глобально.
    _stop_flag = {"stop": False}
    try:
        import keyboard as _kb
        def _on_f9():
            print("\n[atlas] >>> F9 НАЖАТ, ставлю флаг остановки <<<", flush=True)
            _stop_flag["stop"] = True
        _kb.add_hotkey("f9", _on_f9)
        print("[atlas] F9 зарегистрирован успешно — глобальная остановка активна", flush=True)
    except Exception as e:
        print(f"[atlas] keyboard НЕ подключился: {e!r} — F9 не сработает, "
              "используй Q (кликнув по окну предпросмотра) либо Ctrl+C "
              "(кликнув по окну консоли)", flush=True)

    interrupted = False
    try:
        while True:
            if _stop_flag["stop"]:
                break
            frame = cap.grab()
            if frame is None:
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')) or _stop_flag["stop"]:
                    break
                continue
            mm = frame[t:t + h, l:l + w]
            n_frames += 1
            gray = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY)

            # Throttling: сравниваем для сдвига не чаще раза в sample_dt секунд — на
            # 60 FPS сдвиг между СОСЕДНИМИ кадрами субпиксельный (доли px) и тонет в
            # шуме phaseCorrelate, поэтому копим интервал побольше (как в остальном
            # проекте: heading_sample_dt). Кадр на холст всё равно кладём каждый раз —
            # по уже накопленной на тот момент pos_px (не дожидаясь следующего сэмпла).
            now = time.monotonic()
            if now - last_sample_t >= sample_dt:
                last_sample_t = now
                n_compared += 1
                delta = reader.relative_shift(mm)   # (dx_m, dy_m) или None на первом сэмпле
                if delta is not None:
                    dx_px, dy_px = delta[0] / mpp, delta[1] / mpp
                    step = math.hypot(dx_px, dy_px)
                    if step >= args.min_move_px:
                        pos_px[0] += dx_px
                        pos_px[1] += dy_px
                        n_used += 1

            # Кладём текущий кадр на холст с центром в pos_px. Простой "последний
            # выигрывает" — matchTemplate потом ищет лучшее совпадение локально,
            # небольшие швы между кадрами не критичны.
            cx, cy = int(round(pos_px[0])), int(round(pos_px[1]))
            x0, y0 = cx - w // 2, cy - h // 2
            x1, y1 = x0 + w, y0 + h
            if 0 <= x0 and 0 <= y0 and x1 <= canvas_size and y1 <= canvas_size:
                canvas[y0:y1, x0:x1] = gray
                covered[y0:y1, x0:x1] = 255
            else:
                print(f"[atlas] ВНИМАНИЕ: вышли за пределы холста (px={cx},{cy}) — "
                      f"увеличь --canvas-margin-m и начни заново")

            if n_frames % 5 == 0:   # не дёргаем imshow каждый кадр — дорого
                prev = cv2.resize(covered, (500, 500), interpolation=cv2.INTER_NEAREST)
                cv2.putText(prev, f"frames={n_frames} used={n_used} pos_px=({cx},{cy})",
                            (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,), 1)
                cv2.imshow(win, prev)

            if not args.no_topmost and n_frames % 150 == 0:
                # некоторые игры/DWM сбрасывают topmost при alt-tab — подкрепляем
                set_window_topmost(win)

            if (cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'))) or _stop_flag["stop"]:
                break
    except KeyboardInterrupt:
        # Ctrl+C (например, из консоли, если окно предпросмотра всё же не в фокусе) —
        # ТАК ЖЕ штатно сохраняем уже собранное, а не теряем весь проход.
        interrupted = True
        print("\n[atlas] Ctrl+C — завершаю и сохраняю то, что уже собрано...")
    finally:
        cap.close()
        cv2.destroyAllWindows()

    ys, xs = np.where(covered > 0)
    if len(xs) == 0:
        print("[atlas] ничего не записано — атлас не сохранён")
        return
    pad = 10
    x0, x1 = max(0, xs.min() - pad), min(canvas_size, xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(canvas_size, ys.max() + pad)
    cropped = canvas[y0:y1, x0:x1]

    out_path = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    cv2.imwrite(out_path, cropped)

    pct_used = 100 * n_used / max(1, n_compared)
    print(f"\n[atlas] кадров захвата: {n_frames}, сравнений сдвига: {n_compared}, "
          f"учтено в одометрии: {n_used} ({pct_used:.0f}% "
          f"от сравнений прошли порог --min-move-px){' (завершено по Ctrl+C)' if interrupted else ''}")
    if pct_used < 50:
        print(f"[atlas] ВНИМАНИЕ: только {pct_used:.0f}% сравнений прошли порог движения — "
              f"скорее всего часть пути шла ШАГОМ (без спринта) и сдвиг тонул в --min-move-px. "
              f"Атлас может содержать пробелы/повторы. Перезапусти с меньшим --min-move-px "
              f"(например {args.min_move_px / 3:.3f}) либо иди только спринтом при следующей сборке.")
    print(f"[atlas] итоговый атлас: {cropped.shape[1]}x{cropped.shape[0]}px -> {out_path}")
    print("\n=== Впиши в config.yaml -> minimap: ===")
    print(f"  world_atlas_path: {os.path.basename(out_path)}")
    print(f"  atlas_meters_per_px: {mpp:.4f}")
    print("========================================")
    print("[atlas] совет: прогони scripts/run_route.py --hud на этой локации и проверь, "
          "что world_xy_m из MinimapReader.localize() не None (confidence в трейсе ~0.9) — "
          "если None/низкий confidence почти всюду, в первую очередь подними допуск "
          "совпадения (TM_CCOEFF_NORMED порог 0.35 в minimap_reader.localize) либо пересобери "
          "атлас плотнее (помедленнее, с меньшими промежутками между кадрами).")


if __name__ == "__main__":
    main()
