"""
genshin_calibrate.py — калибровщик HSV стрелки поверх игры (tkinter+PIL вариант).
Адаптирован под проект: регион миникарты и стартовые пороги берутся из config.yaml,
вывод — готовые строки для config.yaml (секция minimap).

ТРЕБУЕТ Pillow:  pip install Pillow  (кэш гнать на G:, диск C: переполнен)
Альтернатива БЕЗ Pillow: scripts/calibrate_arrow.py (на cv2-трекбарах, уже работает).

Запуск:  .venv\\Scripts\\python.exe scripts\\genshin_calibrate.py
Двигай ползунки H/S/V пока в окне маски не останется ТОЛЬКО синяя стрелка → «Сохранить».
"""
import math
import os
import sys
import threading
import time
import tkinter as tk

import numpy as np
import cv2
import mss
from PIL import Image, ImageTk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from genshin_nav.config import Config  # noqa: E402

_cfg = Config.load(os.path.join(ROOT, "config.yaml")).minimap
_l, _t, _w, _h = _cfg.region
MINIMAP = {"left": _l, "top": _t, "width": _w, "height": _h}
SCALE = 2
ROI_FRAC = getattr(_cfg, "arrow_roi_frac", 0.22)


class CalibrateApp:
    def __init__(self):
        self.sct = mss.mss()
        self.running = True
        self.root = tk.Tk()
        self.root.title("Genshin Calibrate (-> config.yaml)")
        self.root.wm_attributes("-topmost", True)
        self.root.configure(bg="#111")
        self.root.resizable(False, False)
        W = MINIMAP["width"] * SCALE
        self.root.geometry(f"+{1920 - W - 40}+40")
        self._build_ui()
        self._start_loop()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.mainloop()

    def _build_ui(self):
        W = MINIMAP["width"] * SCALE
        H = MINIMAP["height"] * SCALE
        self.canvas = tk.Canvas(self.root, width=W, height=H, bg="#000", highlightthickness=0)
        self.canvas.pack(padx=4, pady=4)
        self.canvas2 = tk.Canvas(self.root, width=W, height=H, bg="#000", highlightthickness=0)
        self.canvas2.pack(padx=4, pady=(0, 4))
        self.angle_var = tk.StringVar(value="Угол: —")
        tk.Label(self.root, textvariable=self.angle_var, fg="#00e5ff", bg="#111",
                 font=("Consolas", 12, "bold")).pack(pady=2)
        frame = tk.Frame(self.root, bg="#111"); frame.pack(padx=8, pady=4, fill="x")
        self.sliders = {}
        lo = list(_cfg.arrow_hsv_low); hi = list(_cfg.arrow_hsv_high)
        params = [("H Low", lo[0], 179), ("H High", hi[0], 179),
                  ("S Low", lo[1], 255), ("S High", hi[1], 255),
                  ("V Low", lo[2], 255), ("V High", hi[2], 255)]
        for name, default, mx in params:
            row = tk.Frame(frame, bg="#111"); row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{name:7}", fg="#aaa", bg="#111",
                     font=("Consolas", 9), width=7, anchor="w").pack(side="left")
            var = tk.IntVar(value=default)
            tk.Scale(row, from_=0, to=mx, orient="horizontal", variable=var,
                     bg="#1a1a2e", fg="#00e5ff", highlightthickness=0,
                     troughcolor="#003344", length=240).pack(side="left")
            self.sliders[name] = var
        bf = tk.Frame(self.root, bg="#111"); bf.pack(pady=6)
        tk.Button(bf, text="Сохранить -> config.yaml", command=self._save, bg="#004466",
                  fg="#00e5ff", font=("Consolas", 10, "bold"), relief="flat", padx=10).pack(side="left", padx=4)
        tk.Button(bf, text="Выход", command=self._quit, bg="#330000", fg="#ff6666",
                  font=("Consolas", 10), relief="flat", padx=10).pack(side="left", padx=4)
        self._img_ref = None; self._img_ref2 = None

    def _start_loop(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            try:
                self._tick()
            except Exception:
                pass
            time.sleep(1 / 20)

    def _tick(self):
        raw = self.sct.grab(MINIMAP)
        frame = np.array(raw, dtype=np.uint8)[:, :, :3]
        hl = self.sliders["H Low"].get(); hh = self.sliders["H High"].get()
        sl = self.sliders["S Low"].get(); sh = self.sliders["S High"].get()
        vl = self.sliders["V Low"].get(); vh = self.sliders["V High"].get()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hh, sh, vh]))
        h2, w2 = frame.shape[:2]
        cx, cy = w2 / 2, h2 / 2
        # ROI как в проде: круг радиуса arrow_roi_frac (а не весь круг миникарты)
        cmask = np.zeros((h2, w2), np.uint8)
        cv2.circle(cmask, (w2 // 2, h2 // 2), max(8, int(min(w2, h2) * ROI_FRAC)), 255, -1)
        mask = cv2.bitwise_and(mask, cmask)
        debug = frame.copy(); angle_str = "не найдено"
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_d = None, 1e9
        for c in cnts:
            if cv2.contourArea(c) < 30:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            gx, gy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            d = math.hypot(gx - cx, gy - cy)
            if d < best_d:
                best_d, best = d, c
        for c in cnts:
            if c is not best and cv2.contourArea(c) >= 30:
                cv2.drawContours(debug, [c], -1, (80, 80, 80), 1)
        if best is not None:
            M = cv2.moments(best)
            gx, gy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            pts = best[:, 0, :]
            tip = pts[np.argmax(np.linalg.norm(pts - [gx, gy], axis=1))]
            ang = math.degrees(math.atan2(tip[0] - gx, -(tip[1] - gy))) % 360
            angle_str = f"{ang:.1f}°"
            cv2.drawContours(debug, [best], -1, (0, 255, 0), 1)
            cv2.line(debug, (int(gx), int(gy)), tuple(tip.astype(int)), (0, 200, 255), 1)
        W, H = MINIMAP["width"] * SCALE, MINIMAP["height"] * SCALE
        big = cv2.resize(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_NEAREST)
        bigmask = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), (W, H), interpolation=cv2.INTER_NEAREST)
        img1 = ImageTk.PhotoImage(Image.fromarray(big))
        img2 = ImageTk.PhotoImage(Image.fromarray(bigmask))
        self.root.after(0, self._update_canvas, img1, img2, angle_str)

    def _update_canvas(self, img1, img2, angle_str):
        self.canvas.create_image(0, 0, anchor="nw", image=img1)
        self.canvas2.create_image(0, 0, anchor="nw", image=img2)
        self._img_ref = img1; self._img_ref2 = img2
        self.angle_var.set(f"Угол стрелки: {angle_str}")

    def _save(self):
        hl = self.sliders["H Low"].get(); hh = self.sliders["H High"].get()
        sl = self.sliders["S Low"].get(); sh = self.sliders["S High"].get()
        vl = self.sliders["V Low"].get(); vh = self.sliders["V High"].get()
        print("\n=== В config.yaml -> minimap: ===")
        print(f"  arrow_hsv_low:  [{hl}, {sl}, {vl}]")
        print(f"  arrow_hsv_high: [{hh}, {sh}, {vh}]")
        print("=================================\n")
        self.angle_var.set(f"Сохранено в консоль: [{hl},{sl},{vl}]-[{hh},{sh},{vh}]")

    def _quit(self):
        self.running = False
        self.root.destroy()


if __name__ == "__main__":
    CalibrateApp()
