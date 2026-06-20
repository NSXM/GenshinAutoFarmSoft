"""
CPU-параллакс — высокочастотный слой между тяжёлыми SAM-кейфреймами.

Зачем на CPU: SAM/CUDA конкурируют с DirectX-игрой за дискретный GPU, а WDDM
не умеет application-aware приоритизацию (только тайм-слайсинг). Оптический
поток Lucas-Kanade гоняется на CPU и снимает нагрузку с GPU между кейфреймами —
это единственный вариант, который реально убирает контеншн, а не смягчает его.

Дополнительно модуль отдаёт МЕТРИКУ УВЕРЕННОСТИ (число удержанных точек и
residual ожид-vs-факт). По ней KeyframeTrigger решает, что дрейф накопился и
пора ресинкать SAM, — независимо от того, качали ли мы камеру.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

_LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(3, 30, 0.01))  # (TERM_COUNT|EPS, iters, eps)
_FEATURE_PARAMS = dict(maxCorners=300, qualityLevel=0.01,
                       minDistance=7, blockSize=7)


@dataclass
class FlowResult:
    # На каждый трекаемый объект: текущий центроид (пиксели) и скорость (px/кадр).
    centroids: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    velocities: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    good_features: int = 0          # сколько точек удержано суммарно
    residual_px: float = 0.0        # ср. расхождение факт-vs-предсказание
    confident: bool = True          # быстрый флаг качества трекинга


class OpticalFlowTracker:
    """
    Спарс-трекер: на каждый объект (по id) держим набор feature-точек внутри
    его маски от SAM. Между кейфреймами тащим их LK-потоком, центроид облака —
    это и есть актуальная позиция объекта без обращения к GPU.
    """

    def __init__(self, cfg_keyframe):
        self.kf = cfg_keyframe
        self._prev_gray: Optional[np.ndarray] = None
        # id -> (N,1,2) float32 точки
        self._pts: Dict[int, np.ndarray] = {}

    def reset_object(self, obj_id: int, gray: np.ndarray, mask: np.ndarray):
        """Засеять/пересеять точки объекта внутри маски (вызывается на кейфрейме SAM)."""
        feats = cv2.goodFeaturesToTrack(gray, mask=mask, **_FEATURE_PARAMS)
        if feats is not None:
            self._pts[obj_id] = feats.astype(np.float32)
        else:
            self._pts.pop(obj_id, None)

    def seed_from_masks(self, gray: np.ndarray, masks: Dict[int, np.ndarray]):
        for oid, m in masks.items():
            self.reset_object(oid, gray, (m > 0).astype(np.uint8) * 255)
        self._prev_gray = gray

    def track(self, frame_bgr: np.ndarray,
              predicted: Optional[Dict[int, Tuple[float, float]]] = None) -> FlowResult:
        """
        Протащить все точки на новый кадр.
        predicted — ожидаемые центроиды объектов (из триангуляции/модели движения)
        для расчёта residual; опционально.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        res = FlowResult()
        if self._prev_gray is None:
            self._prev_gray = gray
            self._autoseed(gray)          # засеяться сразу, не дожидаясь SAM
            return res

        total_good = 0
        residuals: List[float] = []
        for oid, p0 in list(self._pts.items()):
            if p0 is None or len(p0) < 3:
                self._pts.pop(oid, None)
                continue
            p1, stt, err = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, p0, None, **_LK_PARAMS)
            if p1 is None:
                self._pts.pop(oid, None)
                continue
            mask = stt.reshape(-1).astype(bool)
            good_new = p1[mask]
            good_old = p0[mask]
            if len(good_new) < 3:
                self._pts.pop(oid, None)
                continue
            self._pts[oid] = good_new.reshape(-1, 1, 2)
            total_good += len(good_new)

            prev_c = good_old.reshape(-1, 2).mean(axis=0)
            c = good_new.reshape(-1, 2).mean(axis=0)
            res.centroids[oid] = (float(c[0]), float(c[1]))
            res.velocities[oid] = (float(c[0] - prev_c[0]), float(c[1] - prev_c[1]))
            if predicted and oid in predicted:
                px, py = predicted[oid]
                residuals.append(float(np.hypot(c[0] - px, c[1] - py)))

        # САМОЗАСЕВ независимо от SAM: если точек мало (или SAM выключен) —
        # набрать новые фоновые точки по всему кадру. Без этого gf=0.
        if total_good < self.kf.min_good_features:
            self._autoseed(gray)
            total_good = sum(len(p) for p in self._pts.values() if p is not None)

        res.good_features = total_good
        res.residual_px = float(np.mean(residuals)) if residuals else 0.0
        res.confident = (total_good >= self.kf.min_good_features and
                         res.residual_px <= self.kf.max_flow_residual_px)
        self._prev_gray = gray
        return res

    _BG_ID = 0   # зарезервированный id фонового облака точек (SAM-объекты идут с 1)

    def _autoseed(self, gray: np.ndarray):
        """Набрать фоновые точки по всему кадру (goodFeaturesToTrack), без SAM."""
        feats = cv2.goodFeaturesToTrack(gray, **_FEATURE_PARAMS)
        if feats is not None and len(feats):
            self._pts[self._BG_ID] = feats.astype(np.float32)

    def global_motion(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Глобальный сдвиг кадра (для оценки движения камеры/персонажа в пикселях)
        через фазовую корреляцию по всему кадру. Дёшево, на CPU.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        out = None
        if hasattr(self, "_pg") and self._pg.shape == gray.shape:
            (sx, sy), _ = cv2.phaseCorrelate(self._pg, gray)
            out = (sx, sy)
        self._pg = gray
        return out
