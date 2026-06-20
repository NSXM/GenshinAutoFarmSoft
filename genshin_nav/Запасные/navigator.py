"""
Шаг 7 ТЗ — общий control loop / state machine.

Объединяет все источники сигналов:
  * позицию по миникарте (fusion),
  * объекты от SAM + триангуляция (perception + geometry),
  * флаг застревания (stuck_detector),
и ведёт персонажа по маршруту, симулируя ввод и обходя препятствия.

Архитектура потоков:
  - Главный поток (высокая частота ~30-60 Гц): захват, миникарта, optical-flow,
    Kalman, принятие решений, ввод. Всё на CPU — не конкурирует с игрой за GPU.
  - Поток перцепции (низкая частота, по событиям): SAM-кейфреймы + семантика.
    Запускается ТОЛЬКО когда KeyframeTrigger разрешил (покачивание камеры /
    падение уверенности / форс-интервал). Результат асинхронно вливается в
    fusion. Так тяжёлый GPU-инференс не блокирует управление.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from ..capture.screen_capture import ScreenCapture
from ..minimap.minimap_reader import MinimapReader
from ..perception.optical_flow import OpticalFlowTracker
from ..perception.keyframe import KeyframeTrigger
from ..perception.sam_segmenter import SamSegmenter
from ..perception.semantic import SemanticLabeler
from ..geometry.triangulation import Triangulator, CameraPose
from ..fusion.estimator import FusionEstimator
from ..utils.geom import angle_diff_deg
from .input_sim import InputSimulator
from .stuck_detector import StuckDetector


class State(Enum):
    IDLE = auto()
    ALIGN = auto()         # доворот камеры/курса на следующую точку
    MOVE = auto()
    WOBBLE = auto()        # покачивание камеры для триангуляции (= SAM-кейфрейм)
    AVOID = auto()         # обход препятствия
    STUCK = auto()
    ARRIVED = auto()


@dataclass
class Waypoint:
    x: float
    y: float


class Navigator:
    def __init__(self, cfg, route: List[Waypoint]):
        self.cfg = cfg
        self.route = route
        self.wp_idx = 0

        self.cap = ScreenCapture(cfg.capture.region, cfg.capture.backend,
                                 cfg.capture.target_fps, cfg.capture.window_title)
        self.minimap = MinimapReader(cfg.minimap)
        self.flow = OpticalFlowTracker(cfg.keyframe)
        self.sam = SamSegmenter(cfg.sam)
        self.semantic = SemanticLabeler(cfg.semantic)
        self.trigger = KeyframeTrigger(cfg.keyframe, cfg.sam)
        self.triangulator = Triangulator(cfg.camera)
        self.fusion = FusionEstimator(cfg)
        self.inp = InputSimulator(cfg.control, cfg.camera, dry_run=cfg.control.dry_run)
        self.stuck = StuckDetector(cfg.control)
        self.dry_run = cfg.control.dry_run

        self.state = State.IDLE
        self._running = False
        self._last_t = time.monotonic()

        # обмен с потоком перцепции
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._sam_busy = False
        self._pending_roi: Optional[Tuple[int, int, int, int]] = None

    # ------------------------------------------------------------------ #
    #  Поток перцепции (SAM + семантика), запускается по событию          #
    # ------------------------------------------------------------------ #
    def _perception_worker(self, frame, roi, yaw_before, yaw_after,
                           player_xy):
        try:
            res = self.sam.segment(frame, roi=roi)
            label_map = self.semantic.label_map(frame) if self.cfg.semantic.enabled else None
            cats = self.semantic.classify_masks(res.objects, label_map)

            # пересеять точки optical-flow по свежим маскам (ресинк дрейфа)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            masks = {o.obj_id: (o.mask > 0).astype("uint8") * 255 for o in res.objects}
            with self._lock:
                self.flow.seed_from_masks(gray, masks)

            # триангуляция каждого объекта по покачиванию камеры
            for o in res.objects:
                u = o.centroid[0]
                tri = self.triangulator.from_camera_wobble(
                    player_xy, yaw_before, u, yaw_after, u)
                if tri.world_xy is not None:
                    cat = cats.get(o.obj_id, "obstacle")
                    var = max(0.5, 0.02 * (tri.depth_m or 5.0) ** 2)
                    with self._lock:
                        self.fusion.add_object_world(tri.world_xy, cat, var, tri.moving)
            with self._lock:
                self.fusion.decay()
        finally:
            self._sam_busy = False

    def _maybe_keyframe(self, frame, flow_res, player_xy, did_wobble,
                        yaw_before, yaw_after):
        if not self.cfg.sam.enabled or self._sam_busy:
            return
        if did_wobble:
            self.trigger.notify_camera_wobble()
        dec = self.trigger.update(flow_res.confident, flow_res.good_features,
                                  flow_res.residual_px)
        if not dec.fire:
            return
        # ROI вокруг ожидаемой зоны препятствий (центр-низ кадра по ходу движения)
        H, W = frame.shape[:2]
        roi = (int(W * 0.25), int(H * 0.35), int(W * 0.5), int(H * 0.5))
        self._sam_busy = True
        threading.Thread(
            target=self._perception_worker,
            args=(frame.copy(), roi, yaw_before, yaw_after, player_xy),
            daemon=True).start()

    # ------------------------------------------------------------------ #
    #  Геометрия маршрута                                                 #
    # ------------------------------------------------------------------ #
    def _bearing_to_wp(self, player_xy) -> Optional[float]:
        if self.wp_idx >= len(self.route):
            return None
        wp = self.route[self.wp_idx]
        dx, dy = wp.x - player_xy[0], wp.y - player_xy[1]
        return math.degrees(math.atan2(dx, dy))   # 0=север, по часовой

    def _dist_to_wp(self, player_xy) -> float:
        wp = self.route[self.wp_idx]
        return math.hypot(wp.x - player_xy[0], wp.y - player_xy[1])

    def _obstacle_ahead(self, st) -> Optional[Tuple[float, float]]:
        """Есть ли объект в коридоре перед персонажем? Вернуть его (lateral, fwd)."""
        yaw = math.radians(st.heading_deg)
        fwd = np.array([math.sin(yaw), math.cos(yaw)])
        right = np.array([math.cos(yaw), -math.sin(yaw)])
        px, py = st.player_xy
        for o in st.objects:
            rel = np.array(o.xy) - np.array([px, py])
            f = float(rel @ fwd)
            l = float(rel @ right)
            if 0.3 < f < self.cfg.control.corridor_length_m and \
               abs(l) < self.cfg.control.corridor_half_width_m:
                return (l, f)
        return None

    # ------------------------------------------------------------------ #
    #  Покачивание камеры (триангуляция) — короткий свинг влево-вправо     #
    # ------------------------------------------------------------------ #
    def _do_wobble(self, st) -> Tuple[float, float]:
        """Качнуть камеру на известный угол и вернуть (yaw_before, yaw_after)."""
        yaw_before = st.heading_deg
        swing = 12.0
        self.inp.rotate_camera(+swing)
        time.sleep(0.04)
        self.inp.rotate_camera(-swing)        # вернуть назад
        yaw_after = yaw_before + swing
        return yaw_before, yaw_after

    # ------------------------------------------------------------------ #
    #  Главный цикл                                                       #
    # ------------------------------------------------------------------ #
    def run(self):
        self._running = True
        self.state = State.ALIGN
        frame_i = 0
        try:
            while self._running and self.wp_idx < len(self.route):
                frame = self.cap.grab()
                if frame is None:
                    time.sleep(0.001)
                    continue
                now = time.monotonic()
                dt = max(1e-3, now - self._last_t)
                self._last_t = now
                frame_i += 1

                # --- восприятие высокой частоты (CPU) ---
                mm = self.minimap.read(frame)
                with self._lock:
                    flow_res = self.flow.track(frame)
                # пиксельную скорость flow -> метры/с (грубо, через миникарту-скейл)
                flow_vxy = None
                if mm.delta_xy_m is not None:
                    flow_vxy = (mm.delta_xy_m[0] / dt, mm.delta_xy_m[1] / dt)

                with self._lock:
                    self.fusion.step_player(dt, mm.world_xy_m, flow_vxy, mm.heading_deg)
                    st = self.fusion.snapshot()

                # --- застревание ---
                moving_cmd = self.cfg.control.move_key in self.inp._held
                sv = self.stuck.update(now, st.player_xy, moving_cmd)
                if sv.stuck:
                    self._handle_stuck(sv)
                    continue

                # --- прибытие к точке ---
                if self._dist_to_wp(st.player_xy) < self.cfg.control.waypoint_tolerance_m:
                    self.wp_idx += 1
                    self.inp.stop_moving()
                    self.state = State.ALIGN
                    if self.wp_idx >= len(self.route):
                        break
                    continue

                # --- доворот на цель ---
                target = self._bearing_to_wp(st.player_xy)
                if target is not None:
                    err = angle_diff_deg(target, st.heading_deg)
                    if abs(err) > 12.0:
                        self.inp.rotate_camera(max(-25, min(25, err)))
                        self.state = State.ALIGN

                # --- обход препятствий ---
                obs = self._obstacle_ahead(st)
                did_wobble = False
                yaw_b = yaw_a = st.heading_deg
                if obs is not None:
                    lateral, _ = obs
                    self.state = State.AVOID
                    # шаг в сторону, противоположную препятствию
                    self.inp.rotate_camera(-18.0 if lateral > 0 else 18.0)
                    if obs[1] < 1.5:           # совсем близко — прыжок через кочку
                        self.inp.jump()
                else:
                    # периодически качаем камеру для триангуляции (= SAM-кейфрейм)
                    if frame_i % 20 == 0:
                        yaw_b, yaw_a = self._do_wobble(st)
                        did_wobble = True
                        self.state = State.WOBBLE
                    else:
                        self.state = State.MOVE
                    self.inp.start_moving()

                # --- событийный SAM-кейфрейм ---
                self._maybe_keyframe(frame, flow_res, st.player_xy,
                                     did_wobble, yaw_b, yaw_a)

                # --- структурный лог (особенно полезен в dry-run) ---
                if frame_i % 15 == 0:
                    self._log_state(st, flow_res, obs)

                # ограничить частоту, чтобы не жечь CPU зря
                budget = 1.0 / max(1, self.cfg.capture.target_fps)
                slack = budget - (time.monotonic() - now)
                if slack > 0:
                    time.sleep(slack)

            self.state = State.ARRIVED
            print("[nav] маршрут пройден" if self.wp_idx >= len(self.route)
                  else "[nav] остановлено")
        finally:
            self.stop()

    def _log_state(self, st, flow_res, obs):
        wp = self.route[self.wp_idx] if self.wp_idx < len(self.route) else None
        dist = self._dist_to_wp(st.player_xy) if wp else -1.0
        tgt = self._bearing_to_wp(st.player_xy)
        tag = "[dry]" if self.dry_run else "[nav]"
        print(
            f"{tag} st={self.state.name:7s} "
            f"pos=({st.player_xy[0]:7.2f},{st.player_xy[1]:7.2f}) "
            f"hdg={st.heading_deg:6.1f} "
            f"wp#{self.wp_idx} d={dist:6.2f} "
            f"tgt={('%.1f' % tgt) if tgt is not None else '  -- '} "
            f"flow(gf={flow_res.good_features:3d},res={flow_res.residual_px:4.1f}) "
            f"obj={len(st.objects)} "
            f"obstacle={'yes' if obs else 'no'}"
        )

    def _handle_stuck(self, sv):
        self.state = State.STUCK
        print(f"[nav] STUCK ({sv.displacement_m:.2f} м) -> {sv.suggestion}")
        self.inp.stop_moving()
        if sv.suggestion == "jump":
            self.inp.jump()
        elif sv.suggestion == "turn":
            self.inp.rotate_camera(35.0)
            self.inp.jump()
        elif sv.suggestion == "reroute":
            self.inp.rotate_camera(70.0)
            time.sleep(0.1)
            self.inp.start_moving()
            time.sleep(0.4)
        self.inp.start_moving()

    def stop(self):
        self._running = False
        try:
            self.inp.release_all()
            self.cap.close()
        except Exception:
            pass