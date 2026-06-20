"""
genshin_calibrate.py
====================
Калибровщик поверх игры — не нужно сворачивать Genshin.

Запуск:
    python genshin_calibrate.py

Что делать:
    1. Запустите скрипт — окно появится ПОВЕРХ игры.
    2. Ползунки H/S/V двигайте мышью пока в окне
       не останется подсвечена ТОЛЬКО синяя стрелка.
    3. Нажмите кнопку "Сохранить" — значения выведутся
       в консоль, вставьте их в genshin_minimap.py.
    4. Закройте крестиком или кнопкой "Выход".
"""

import math
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
import mss
from PIL import Image, ImageTk

# ── Координаты миникарты (левый верхний угол 1920×1080) ──────────────────────
MINIMAP = {
    "left":    20,
    "top":     20,
    "width":  180,
    "height": 180,
}
SCALE = 3   # увеличение превью (180 → 540 px)

# ─────────────────────────────────────────────────────────────────────────────

class CalibrateApp:
    def __init__(self):
        self.sct     = mss.mss()
        self.running = True

        self.root = tk.Tk()
        self.root.title("Genshin Calibrate")
        self.root.wm_attributes("-topmost", True)
        self.root.configure(bg="#111")
        self.root.resizable(False, False)
        # Правый верхний угол экрана (не мешает миникарте слева)
        W = MINIMAP["width"] * SCALE
        self.root.geometry(f"+{1920 - W - 20}+20")

        self._build_ui()
        self._start_loop()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.mainloop()

    def _build_ui(self):
        W = MINIMAP["width"]  * SCALE
        H = MINIMAP["height"] * SCALE

        # Превью миникарты
        self.canvas = tk.Canvas(self.root, width=W, height=H,
                                bg="#000", highlightthickness=0)
        self.canvas.pack(padx=4, pady=4)

        # Превью маски
        self.canvas2 = tk.Canvas(self.root, width=W, height=H,
                                 bg="#000", highlightthickness=0)
        self.canvas2.pack(padx=4, pady=(0, 4))

        # Метка угла
        self.angle_var = tk.StringVar(value="Угол: —")
        tk.Label(self.root, textvariable=self.angle_var,
                 fg="#00e5ff", bg="#111",
                 font=("Consolas", 12, "bold")).pack(pady=2)

        # Ползунки
        sliders_frame = tk.Frame(self.root, bg="#111")
        sliders_frame.pack(padx=8, pady=4, fill="x")

        self.sliders = {}
        params = [
            ("H Low",   85, 0,   179),
            ("H High", 105, 0,   179),
            ("S Low",  120, 0,   255),
            ("S High", 255, 0,   255),
            ("V Low",  120, 0,   255),
            ("V High", 255, 0,   255),
        ]
        for name, default, lo, hi in params:
            row = tk.Frame(sliders_frame, bg="#111")
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{name:7}", fg="#aaa", bg="#111",
                     font=("Consolas", 9), width=7, anchor="w").pack(side="left")
            var = tk.IntVar(value=default)
            sl  = tk.Scale(row, from_=lo, to=hi, orient="horizontal",
                           variable=var, bg="#1a1a2e", fg="#00e5ff",
                           highlightthickness=0, troughcolor="#003344",
                           length=260, showvalue=True)
            sl.pack(side="left")
            self.sliders[name] = var

        # Кнопки
        btn_frame = tk.Frame(self.root, bg="#111")
        btn_frame.pack(pady=6)
        tk.Button(btn_frame, text="💾 Сохранить значения",
                  command=self._save,
                  bg="#004466", fg="#00e5ff",
                  font=("Consolas", 10, "bold"),
                  relief="flat", padx=10).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Выход",
                  command=self._quit,
                  bg="#330000", fg="#ff6666",
                  font=("Consolas", 10),
                  relief="flat", padx=10).pack(side="left", padx=4)

        self._img_ref  = None
        self._img_ref2 = None

    # ── Цикл захвата ──────────────────────────────────────────────────────────

    def _start_loop(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            try:
                self._tick()
            except Exception:
                pass
            import time; time.sleep(1/20)

    def _tick(self):
        # Захват
        raw   = self.sct.grab(MINIMAP)
        frame = np.array(raw, dtype=np.uint8)[:, :, :3]

        # HSV-маска
        hl = self.sliders["H Low"].get();  hh = self.sliders["H High"].get()
        sl = self.sliders["S Low"].get();  sh = self.sliders["S High"].get()
        vl = self.sliders["V Low"].get();  vh = self.sliders["V High"].get()

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array([hl, sl, vl]),
                           np.array([hh, sh, vh]))

        # Детекция угла — берём контур БЛИЖАЙШИЙ к центру миникарты
        debug     = frame.copy()
        angle_str = "не найдено"
        h2, w2    = frame.shape[:2]
        img_cx, img_cy = w2 / 2, h2 / 2

        # Обнулить пиксели вне круга миникарты
        cmask = np.zeros((h2, w2), dtype=np.uint8)
        cv2.circle(cmask, (w2 // 2, h2 // 2), min(w2, h2) // 2 - 4, 255, -1)
        mask = cv2.bitwise_and(mask, cmask)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        best_c, best_dist = None, float("inf")
        for c in cnts:
            if cv2.contourArea(c) < 20:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            ccx = M["m10"] / M["m00"]
            ccy = M["m01"] / M["m00"]
            d   = math.hypot(ccx - img_cx, ccy - img_cy)
            if d < best_dist:
                best_dist, best_c = d, c

        # Отброшенные контуры — серым (видно что именно отбросили)
        for c in cnts:
            if c is not best_c and cv2.contourArea(c) >= 20:
                cv2.drawContours(debug, [c], -1, (80, 80, 80), 1)

        max_dist = min(w2, h2) / 2 * 0.35
        if best_c is not None and best_dist <= max_dist:
            M   = cv2.moments(best_c)
            cx_ = M["m10"] / M["m00"]
            cy_ = M["m01"] / M["m00"]
            pts   = best_c[:, 0, :]
            dists = np.linalg.norm(pts - [cx_, cy_], axis=1)
            tip   = pts[np.argmax(dists)]
            dx, dy = tip[0] - cx_, tip[1] - cy_
            angle  = math.degrees(math.atan2(dx, -dy)) % 360
            angle_str = f"{angle:.1f}°"
            cv2.drawContours(debug, [best_c], -1, (0, 255, 0), 1)
            cv2.circle(debug, (int(cx_), int(cy_)), 3, (255, 50, 50), -1)
            cv2.circle(debug, tuple(tip.astype(int)), 3, (0, 0, 255), -1)
            cv2.line(debug, (int(cx_), int(cy_)),
                     tuple(tip.astype(int)), (0, 200, 255), 1)

        # Масштаб x3
        W, H = MINIMAP["width"] * SCALE, MINIMAP["height"] * SCALE
        big     = cv2.resize(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB),
                             (W, H), interpolation=cv2.INTER_NEAREST)
        bigmask = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB),
                             (W, H), interpolation=cv2.INTER_NEAREST)

        # Текст на превью
        cv2.putText(big, f"Angle: {angle_str}", (5, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 1)

        img1 = ImageTk.PhotoImage(Image.fromarray(big))
        img2 = ImageTk.PhotoImage(Image.fromarray(bigmask))

        self._angle_str = angle_str
        self._img1, self._img2 = img1, img2

        self.root.after(0, self._update_canvas, img1, img2, angle_str)

    def _update_canvas(self, img1, img2, angle_str):
        self.canvas.create_image(0, 0, anchor="nw", image=img1)
        self.canvas2.create_image(0, 0, anchor="nw", image=img2)
        self._img_ref  = img1   # держим ссылку (GC)
        self._img_ref2 = img2
        self.angle_var.set(f"Угол стрелки: {angle_str}")

    # ── Действия ──────────────────────────────────────────────────────────────

    def _save(self):
        hl = self.sliders["H Low"].get();  hh = self.sliders["H High"].get()
        sl = self.sliders["S Low"].get();  sh = self.sliders["S High"].get()
        vl = self.sliders["V Low"].get();  vh = self.sliders["V High"].get()
        print("\n=== Вставьте в genshin_minimap.py ===")
        print(f"  ARROW_LOW  = np.array([{hl}, {sl}, {vl}])")
        print(f"  ARROW_HIGH = np.array([{hh}, {sh}, {vh}])")
        print("=====================================\n")
        # Показать также в окне
        self.angle_var.set(f"Сохранено в консоль! [{hl},{sl},{vl}] — [{hh},{sh},{vh}]")

    def _quit(self):
        self.running = False
        self.root.destroy()


if __name__ == "__main__":
    CalibrateApp()
