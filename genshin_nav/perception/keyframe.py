"""
Событийный планировщик вызова тяжёлого SAM-кейфрейма.

НЕ таймер. Триггеры (по твоей обратной связи):
  1. on_camera_wobble  — покачивание камеры под известным углом (шаг 4 ТЗ).
       Именно в этот момент и так нужна точная маска SAM для расчёта смещения
       объекта → синхронизируем, не тратим лишний вызов.
  2. on_low_confidence — трекер optical flow потерял точки или residual
       (ожид-vs-факт) превысил порог → дрейф накопился, нужен ресинк, даже
       если камера не двигалась.
  3. max_interval_frames — верхний предел: форс-ресинк, чтобы дрейф в принципе
       не сломал математику триангуляции.

Чисто временной триггер плох: либо слишком часто (контеншн не уходит), либо
слишком редко (дрейф ломает триангуляцию). Событийный — попадает точно.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TriggerDecision:
    fire: bool
    reason: str


class KeyframeTrigger:
    def __init__(self, cfg_keyframe, cfg_sam):
        self.kf = cfg_keyframe
        self.sam = cfg_sam
        self._frames_since = 10 ** 9     # форсим первый кейфрейм сразу
        self._pending_wobble = False

    def notify_camera_wobble(self):
        """Навигатор вызывает это перед/во время покачивания камеры."""
        self._pending_wobble = True

    def update(self, flow_confident: bool, good_features: int,
               residual_px: float) -> TriggerDecision:
        self._frames_since += 1

        # антидребезг
        if self._frames_since < self.kf.min_frames_between:
            self._pending_wobble = False
            return TriggerDecision(False, "debounce")

        reason: Optional[str] = None
        if self.kf.on_camera_wobble and self._pending_wobble:
            reason = "camera_wobble"
        elif self.kf.on_low_confidence and (
                good_features < self.kf.min_good_features or
                residual_px > self.kf.max_flow_residual_px or
                not flow_confident):
            reason = f"low_confidence(gf={good_features},res={residual_px:.1f})"
        elif self._frames_since >= self.sam.max_interval_frames:
            reason = "max_interval"

        if reason:
            self._frames_since = 0
            self._pending_wobble = False
            return TriggerDecision(True, reason)
        return TriggerDecision(False, "skip")
