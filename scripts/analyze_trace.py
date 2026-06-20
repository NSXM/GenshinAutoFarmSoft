"""
Анализ покадрового трейса живого прохождения (diag/run_trace.csv) — ставит диагноз
качанию руля. Запуск:  .venv\\Scripts\\python.exe scripts\\analyze_trace.py

Различает три причины колебаний на (около)прямом участке:
  1) ШУМ КУРСА     — hdg дёргается сам, даже когда руля нет (dx=0) и err мал.
  2) ПЕРЕЛЁТ/ЛАГ    — err меняет знак с заметным периодом, hdg гоняется за brg
                      с запаздыванием (контур перерегулирует: gain/throttle велики).
  3) РАССОГЛАСОВАНИЕ — hdg и brg держат устойчивую РАЗНИЦУ (систематический сдвиг/
                      зеркало системы координат): средняя |err| большая, знак стабилен.
"""
from __future__ import annotations
import csv
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ang_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def circ_mean(degs):
    s = sum(math.sin(math.radians(d)) for d in degs)
    c = sum(math.cos(math.radians(d)) for d in degs)
    return math.degrees(math.atan2(s, c))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "diag", "run_trace.csv")
    if not os.path.exists(path):
        print(f"нет трейса: {path}\nСначала прогони scripts/run_route.py (он пишет CSV).")
        return
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "t": float(r["t"]), "px": float(r["px"]), "py": float(r["py"]),
                    "hdg": float(r["hdg"]), "brg": float(r["brg"]),
                    "err": float(r["err"]), "move": int(r["move"]),
                    "dx": float(r["dx"]), "action": r["action"],
                })
            except (ValueError, KeyError):
                continue
    if len(rows) < 10:
        print(f"слишком мало строк ({len(rows)}) — прогони подольше (10-20с прямого хода)")
        return

    mv = [r for r in rows if r["move"] == 1]
    dur = rows[-1]["t"] - rows[0]["t"]
    print(f"строк: {len(rows)}  длительность: {dur:.1f}с  движение(move=Y): "
          f"{len(mv)} ({100*len(mv)//max(1,len(rows))}%)")
    if len(mv) < 10:
        print("!! бот почти не «движется» по миникарте (move=Y редко) — это и есть корень:")
        print("   курс из сдвига карты не считается -> рулёжка идёт по мусору/последнему курсу.")
        print("   Проверь: реально ли едет персонаж, регион миникарты, heading_move_floor_px.")
        return

    errs = [r["err"] for r in mv]
    abs_errs = [abs(e) for e in errs]
    mean_abs = sum(abs_errs) / len(abs_errs)
    mean_signed = sum(errs) / len(errs)               # систематический сдвиг?
    # смены знака ошибки (частота колебаний)
    flips = sum(1 for a, b in zip(errs, errs[1:]) if a * b < 0)
    flip_rate = flips / max(1e-3, dur)

    # шум курса: разброс hdg на соседних кадрах ПРИ ОТСУТСТВИИ руля (dx=0) и малом err
    quiet = [(a, b) for a, b in zip(mv, mv[1:])
             if a["dx"] == 0 and abs(a["err"]) < 10 and (b["t"] - a["t"]) < 0.1]
    if quiet:
        jit = [abs(ang_diff(b["hdg"], a["hdg"])) for a, b in quiet]
        jit_med = sorted(jit)[len(jit) // 2]
        jit_max = max(jit)
    else:
        jit_med = jit_max = float("nan")

    print(f"\n|err| средн = {mean_abs:5.1f}°   err средн (со знаком) = {mean_signed:+5.1f}°")
    print(f"смен знака err: {flips}  ({flip_rate:.2f}/с)")
    print(f"джиттер курса в покое (dx=0,|err|<10): медиана={jit_med:.1f}° макс={jit_max:.1f}°")

    print("\n--- ДИАГНОЗ ---")
    verdict = []
    if not math.isnan(jit_med) and jit_med > 4.0:
        verdict.append(
            f"1) ШУМ КУРСА: курс скачет на {jit_med:.0f}° даже без руля. Контур гоняется\n"
            f"   за шумом. Лечить в источнике курса (сильнее EMA: heading_smooth_alpha\n"
            f"   0.4->0.25; выше heading_min_response/heading_move_floor_px) и увеличить\n"
            f"   turn_deadzone_deg выше джиттера.")
    if abs(mean_signed) > 12.0 and abs(mean_signed) > 0.6 * mean_abs:
        verdict.append(
            f"3) РАССОГЛАСОВАНИЕ СИСТЕМ: устойчивый сдвиг err≈{mean_signed:+.0f}° (не вокруг 0).\n"
            f"   hdg и brg в разных системах координат -> проверь знак/зеркало (move_sign,\n"
            f"   bearing=atan2(-dx,dy)). Если |err|≈180 — курс развёрнут; если ≈±90 —\n"
            f"   перепутаны оси.")
    if flip_rate > 0.8 and mean_abs < 35 and (math.isnan(jit_med) or jit_med <= 4.0):
        verdict.append(
            f"2) ПЕРЕЛЁТ/ЛАГ: err ритмично меняет знак ({flip_rate:.1f}/с) при чистом курсе.\n"
            f"   Контур перерегулирует. Лечить: turn_gain 0.5->0.3, turn_deadzone_deg\n"
            f"   5->8, turn_throttle_s 0.2->0.3, turn_max_units 40->25.")
    if not verdict:
        verdict.append("Явной патологии по метрикам нет — пришли сам CSV/лог, гляну вручную.")
    print("\n".join(verdict))

    print("\n--- первые 30 строк движения (t hdg brg err dx act) ---")
    for r in mv[:30]:
        print(f"  t={r['t']:6.2f} hdg={r['hdg']:5.0f} brg={r['brg']:5.0f} "
              f"err={r['err']:+5.0f} dx={r['dx']:+5.0f} {r['action']}")


if __name__ == "__main__":
    main()
