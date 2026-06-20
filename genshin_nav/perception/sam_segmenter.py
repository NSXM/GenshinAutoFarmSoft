"""
Шаг 2 ТЗ — уточнение позиции через SAM3 (маски + метки объектов).

Ключевые решения (по обсуждению контеншна за GPU):
  * ROI-режим: сегментируем только предсказанную область (кроп вокруг ожидаемой
    позиции объекта по углу поворота камеры и пройденному пути), а не весь кадр.
    Резко уменьшает compute и длительность GPU-стоппера в момент кейфрейма.
  * backend='openvino': увести инференс на CPU/iGPU/NPU и физически освободить
    дискретную видеокарту под игру — корневое решение контеншна, т.к. WDDM не
    приоритизирует игру и CUDA-процесс друг относительно друга.
  * Sam3Tracker (use_tracker) — лёгкий промптовый трекер SAM3 между кейфреймами.
    Включать только если чистого Lucas-Kanade не хватает по точности, и держать
    его тоже на openvino/CPU/iGPU, иначе контеншн вернётся.

Вызов происходит НЕ по таймеру, а по событиям (см. keyframe.KeyframeTrigger).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class SamObject:
    obj_id: int
    mask: np.ndarray                       # bool/uint8, в координатах ПОЛНОГО кадра
    centroid: Tuple[float, float]          # (u, v) пиксели полного кадра
    bbox: Tuple[int, int, int, int]        # (x, y, w, h)
    area: int
    label: Optional[str] = None            # заполняется semantic-слоем


@dataclass
class SamResult:
    objects: List[SamObject] = field(default_factory=list)
    roi: Optional[Tuple[int, int, int, int]] = None   # где реально считали
    backend: str = ""
    infer_ms: float = 0.0


class SamSegmenter:
    def __init__(self, cfg_sam):
        self.cfg = cfg_sam
        self._impl = None
        self._next_id = 1
        if cfg_sam.enabled:
            self._init_backend()

    # ---- инициализация бэкендов -------------------------------------------
    def _init_backend(self):
        if self.cfg.backend == "openvino":
            self._init_openvino()
        else:
            self._init_torch()

    def _init_torch(self):
        """SAM3 через transformers (decoupled detector–tracker design)."""
        try:
            import torch
            from transformers import Sam3Processor, Sam3Model  # SAM3 API
            self._torch = torch
            self._processor = Sam3Processor.from_pretrained(self.cfg.model_id)
            self._model = Sam3Model.from_pretrained(self.cfg.model_id)
            dev = self.cfg.device if torch.cuda.is_available() or self.cfg.device == "cpu" else "cpu"
            self._model.to(dev).eval()
            self._device = dev
            self._impl = "torch"
            print(f"[sam] torch backend on {dev}")
        except Exception as e:
            print(f"[sam] не удалось поднять SAM3 (transformers): {e}\n"
                  f"      перцепция SAM отключена, остаётся optical-flow слой.")
            self._impl = None

    def _init_openvino(self):
        """
        SAM/сегментация на CPU/iGPU/NPU через OpenVINO — освобождает дискретку.
        Ожидается заранее сконвертированная IR-модель (openvino_model.xml).
        """
        try:
            import openvino as ov
            core = ov.Core()
            # device: 'CPU' | 'GPU.0'(iGPU) | 'GPU.1'(dGPU) | 'NPU'
            self._ov_model = core.read_model(self.cfg.model_id)  # путь к .xml
            self._ov_compiled = core.compile_model(self._ov_model, self.cfg.device)
            self._impl = "openvino"
            print(f"[sam] openvino backend on {self.cfg.device}")
        except Exception as e:
            print(f"[sam] OpenVINO backend недоступен: {e}; пробую torch/cpu")
            self.cfg.backend = "torch"
            self.cfg.device = "cpu"
            self._init_torch()

    # ---- основной вызов ----------------------------------------------------
    def segment(self, frame_bgr: np.ndarray,
                roi: Optional[Tuple[int, int, int, int]] = None,
                prompts: Optional[List[Tuple[float, float]]] = None) -> SamResult:
        """
        Сегментация кадра или ROI.
          roi      — (x, y, w, h) область интереса (предсказанная позиция объекта).
                     Если None и cfg.roi_only — берём центр кадра как фолбэк.
          prompts  — точки-подсказки (u,v) в координатах ПОЛНОГО кадра (Sam3Tracker).
        Возвращает объекты с масками, поднятыми обратно в координаты полного кадра.
        """
        if self._impl is None:
            return SamResult(backend="none")

        H, W = frame_bgr.shape[:2]
        if roi is None and self.cfg.roi_only:
            s = int(min(H, W) * 0.5)
            roi = ((W - s) // 2, (H - s) // 2, s, s)
        x, y, w, h = roi if roi else (0, 0, W, H)
        x = max(0, x - self.cfg.roi_pad_px); y = max(0, y - self.cfg.roi_pad_px)
        w = min(W - x, w + 2 * self.cfg.roi_pad_px)
        h = min(H - y, h + 2 * self.cfg.roi_pad_px)
        crop = frame_bgr[y:y + h, x:x + w]

        # даунскейл ради скорости
        scale = self.cfg.input_downscale
        small = cv2.resize(crop, None, fx=scale, fy=scale) if scale != 1.0 else crop

        if self._impl == "torch":
            masks_small = self._segment_torch(small, prompts, (x, y), scale)
        else:
            masks_small = self._segment_openvino(small, prompts)

        objects: List[SamObject] = []
        for m in masks_small:
            # вернуть маску в полный размер кадра
            m_full = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            full = np.zeros((H, W), np.uint8)
            full[y:y + h, x:x + w] = m_full
            ys, xs = np.where(full > 0)
            if len(xs) < 10:
                continue
            cu, cv_ = float(xs.mean()), float(ys.mean())
            bx, by, bw, bh = int(xs.min()), int(ys.min()), int(xs.ptp() + 1), int(ys.ptp() + 1)
            objects.append(SamObject(self._next_id, full, (cu, cv_),
                                     (bx, by, bw, bh), int(len(xs))))
            self._next_id += 1

        return SamResult(objects=objects, roi=(x, y, w, h), backend=self._impl)

    def _segment_torch(self, img_bgr, prompts, roi_origin, scale):
        torch = self._torch
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        kwargs = {}
        if prompts:  # перенос точек в координаты кропа+скейла
            ox, oy = roi_origin
            pts = [[[(u - ox) * scale, (v - oy) * scale] for (u, v) in prompts]]
            kwargs["input_points"] = pts
        inputs = self._processor(images=rgb, return_tensors="pt", **kwargs).to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        masks = self._processor.post_process_masks(
            out.pred_masks, inputs.get("original_sizes"))
        res = []
        for m in masks[0]:
            arr = m.cpu().numpy()
            arr = arr[0] if arr.ndim == 3 else arr
            res.append(arr > 0.5)
        return res

    def _segment_openvino(self, img_bgr, prompts):
        # Заглушка препроцесса под конкретную IR-модель SAM (зависит от конверсии).
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(rgb, (2, 0, 1))[None]
        out = self._ov_compiled(inp)
        logits = list(out.values())[0]
        masks = logits[0] > 0.0
        return [m for m in masks] if masks.ndim == 3 else [masks]
