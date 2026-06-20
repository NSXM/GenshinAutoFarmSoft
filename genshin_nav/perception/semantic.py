"""
Шаг 3 ТЗ — смысловая разметка поверх масок SAM (опционально).

SAM даёт «что-то-объект» без смысла. Семантическая сеть (sithu31296/
semantic-segmentation: SegFormer/BiSeNet/DDRNet и т.п.) даёт карту меток
классов. Объединяем: для каждой маски SAM берём преобладающую семантическую
метку внутри неё → объект получает смысл (дерево / монстр / камень / трава).

Монстров, как и отмечено в ТЗ, проще детектить именно по семантическим меткам
(класс person/animal), а не по геометрии маски.

Чтобы не возвращать контеншн на дискретку — эту сеть тоже логично гонять через
OpenVINO на CPU/iGPU (см. SemanticCfg.device).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class SemanticLabeler:
    def __init__(self, cfg_semantic):
        self.cfg = cfg_semantic
        self._model = None
        self._labels: List[str] = []
        if cfg_semantic.enabled:
            self._init_model()

    def _init_model(self):
        try:
            # sithu31296/semantic-segmentation ставится как пакет `semseg`
            import torch
            from semseg.models import SegFormer  # пример; имя зависит от чекпойнта
            self._torch = torch
            self._model = SegFormer(backbone="MiT-B0", num_classes=150)
            if self.cfg.weights_path:
                self._model.load_state_dict(torch.load(self.cfg.weights_path,
                                                       map_location="cpu"))
            self._model.eval().to(self.cfg.device if torch.cuda.is_available() else "cpu")
            # ADE20K-подобные имена классов — грузятся из чекпойнта/датасета
            self._labels = self._load_label_names()
            print(f"[semantic] модель поднята на {self.cfg.device}")
        except Exception as e:
            print(f"[semantic] недоступна ({e}); смысловая разметка пропускается")
            self._model = None

    def _load_label_names(self) -> List[str]:
        # В реальности грузится из json датасета; здесь — минимальный плейсхолдер.
        return []

    def label_map(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Полная карта меток классов (H, W) int. None, если модель не поднята."""
        if self._model is None:
            return None
        import torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ten = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0
        ten = ten.to(next(self._model.parameters()).device)
        with torch.no_grad():
            logits = self._model(ten)
        return logits.argmax(1)[0].cpu().numpy().astype(np.int32)

    def classify_masks(self, sam_objects, label_map: Optional[np.ndarray]) -> Dict[int, str]:
        """
        Для каждой маски SAM — преобладающая семантическая метка → категория
        (monster / obstacle / other), которую дальше использует навигатор.
        """
        out: Dict[int, str] = {}
        if label_map is None:
            return out
        for obj in sam_objects:
            m = obj.mask > 0
            if m.sum() == 0:
                continue
            vals = label_map[m]
            if vals.size == 0:
                continue
            top = int(np.bincount(vals).argmax())
            name = self._labels[top] if top < len(self._labels) else str(top)
            out[obj.obj_id] = self._to_category(name)
            obj.label = out[obj.obj_id]
        return out

    def _to_category(self, name: str) -> str:
        if name in self.cfg.monster_labels:
            return "monster"
        if name in self.cfg.obstacle_labels:
            return "obstacle"
        return "other"
