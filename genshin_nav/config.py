"""Загрузка и хранение конфигурации (config.yaml) в виде датаклассов."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple


def update_config_value(path: str, dotted_key: str, value) -> bool:
    """
    Точечно заменить значение одного ключа в config.yaml, сохранив комментарии и
    форматирование (pyyaml при перезаписи комментарии теряет — поэтому правим
    текстом построчно).

    dotted_key: "section.key", например "minimap.region" или
                "camera.deg_per_mouse_unit". value — то, что подставить
                (списки сериализуются как YAML flow-list [a, b, c]).
    Возвращает True, если ключ найден и заменён.
    """
    section, _, key = dotted_key.partition(".")
    if not key:
        raise ValueError("dotted_key должен быть вида 'section.key'")

    if isinstance(value, (list, tuple)):
        val_str = "[" + ", ".join(str(v) for v in value) + "]"
    else:
        val_str = str(value)

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_section = False
    sec_re = re.compile(r"^(\w+):")
    key_re = re.compile(r"^(\s+)" + re.escape(key) + r":\s*(.*?)(\s*#.*)?$")
    for i, line in enumerate(lines):
        m_sec = sec_re.match(line)
        if m_sec:
            in_section = (m_sec.group(1) == section)
            continue
        if in_section:
            m_key = key_re.match(line)
            if m_key:
                indent = m_key.group(1)
                comment = m_key.group(3) or ""
                lines[i] = f"{indent}{key}: {val_str}{comment}\n"
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                return True
    return False

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class CaptureCfg:
    # Прямоугольник окна игры (left, top, width, height). None -> весь экран / автопоиск окна.
    region: Optional[Tuple[int, int, int, int]] = None
    window_title: str = "Genshin Impact"
    backend: str = "dxcam"          # dxcam | mss
    target_fps: int = 60


@dataclass
class MinimapCfg:
    # Регион миникарты ВНУТРИ кадра игры (left, top, w, h) в пикселях.
    # Значения по умолчанию — под 1920x1080, левый верхний угол.
    region: Tuple[int, int, int, int] = (52, 22, 200, 200)
    # Путь к заранее сшитому атласу мира (north-up). Если None — глобальная
    # локализация по атласу отключена, работаем только в относительных координатах.
    world_atlas_path: Optional[str] = None
    # Масштаб атласа: сколько игровых метров приходится на 1 пиксель атласа.
    atlas_meters_per_px: float = 1.0
    # Цвет маркера игрока (стрелка) в HSV. В Genshin стрелка ГОЛУБАЯ (cyan), а не
    # белая; и hue у неё близок к фону карты — отличается яркостью (V) и тем, что
    # стрелка всегда в центре. Поэтому ищем cyan в узкой центральной зоне.
    arrow_hsv_low: Tuple[int, int, int] = (85, 90, 140)
    arrow_hsv_high: Tuple[int, int, int] = (110, 255, 255)
    # Доля размера миникарты под центральную зону поиска стрелки (узкая, чтобы не
    # слипалась с посторонним cyan фона/иконок). ~0.10 => радиус ~20px при 200px.
    arrow_roi_frac: float = 0.10
    # Масштаб миникарты: игровых метров на 1 пиксель миникарты.
    minimap_meters_per_px: float = 0.6
    # Минимальный сдвиг карты за кадр (метры), ниже которого курс из движения не
    # определяем (считаем, что стоим на месте).
    move_min_shift_m: float = 0.05
    # Курс копим как СУММУ сдвигов кадров, разнесённых во ВРЕМЕНИ на heading_sample_dt
    # (на 100 FPS сдвиг за кадр суб-пиксельный и тонет в шуме — сравниваем кадры с
    # интервалом, как в дампе). Когда сумма наберёт столько пикселей — берём
    # направление и обнуляем накопитель.
    heading_sample_dt: float = 0.15
    # Минимальное качество совпадения (response из phaseCorrelate). Сэмплы ниже —
    # чистый шум, не учитываем.
    heading_min_response: float = 0.15
    # Минимальный сдвиг карты за сэмпл (пиксели), чтобы считать, что МЫ ДВИЖЕМСЯ.
    # Каждый такой сэмпл уже даёт надёжное направление (response ~0.9) — копить не
    # нужно, берём курс сразу и сглаживаем.
    heading_move_floor_px: float = 0.15
    # Сглаживание курса (круговое EMA): 0..1, больше = быстрее реакция, больше шум.
    heading_smooth_alpha: float = 0.4
    # Резервный курс по ГОЛУБОМУ КОНУСУ камеры (текстуро-независим): на неоткрытой
    # местности (прозрачная карта, мало текстуры) курс по сдвигу карты буксует —
    # тогда падаем на направление конуса. Конус = направление камеры = куда пойдёт
    # персонаж по W. Используется, когда курс по карте «устарел» (давно не движемся
    # по текстуре).
    use_cone_fallback: bool = True
    cone_hsv_low: Tuple[int, int, int] = (85, 90, 150)   # ярче, чем фон-вода
    cone_hsv_high: Tuple[int, int, int] = (110, 255, 255)
    cone_roi_frac: float = 0.22          # центральная зона поиска конуса
    cone_min_area: int = 25
    heading_stale_after_s: float = 0.5   # если курс по карте не обновлялся дольше — берём конус
    # Знак направления движения относительно сдвига текстуры карты (сверено с
    # конусом камеры на цветном дампе: -1). Если курс перевёрнут — поменять знак.
    move_sign: int = -1
    # Кольцевая маска для корреляции: убрать центральный маркер/конус игрока
    # (статичен, тянет сдвиг к нулю) и углы квадрата (3D-сцена за кругом карты).
    motion_ring_inner_frac: float = 0.25
    motion_ring_outer_frac: float = 0.48


@dataclass
class CameraCfg:
    # Внутренние параметры виртуальной камеры игры.
    fov_horizontal_deg: float = 45.0     # Genshin ~ 45 по вертикали; задаём гориз. эквивалент
    image_width: int = 1920
    image_height: int = 1080
    # Орбитальная модель: камера 3-го лица вращается вокруг персонажа на радиусе R (метры).
    orbit_radius_m: float = 4.5
    # Чувствительность мыши: сколько градусов yaw на 1 единицу относительного сдвига мыши.
    deg_per_mouse_unit: float = 0.12


@dataclass
class SamCfg:
    enabled: bool = True
    model_id: str = "facebook/sam3"
    # Бэкенд инференса. openvino позволяет увести SAM на CPU/iGPU/NPU и
    # физически освободить дискретную видеокарту под игру — это устраняет
    # контеншн в корне, а не просто снижает частоту фризов (WDDM не умеет
    # application-aware приоритизацию между DirectX-игрой и CUDA-процессом).
    backend: str = "torch"           # torch | openvino
    device: str = "cuda"             # torch: cuda|cpu ; openvino: GPU.0|GPU.1|CPU|NPU
    # SAM вызывается ПО СОБЫТИЯМ (см. KeyframeCfg), таймер — лишь верхний предел
    # форс-ресинка, чтобы дрейф трекера не сломал математику триангуляции.
    max_interval_frames: int = 120
    input_downscale: float = 0.5     # ужать кадр перед SAM ради скорости
    points_per_side: int = 16        # для automatic mask generation
    # ROI-режим: сегментировать только предсказанную область (кроп вокруг
    # ожидаемой позиции объекта), а не весь кадр. Меньше compute и короче
    # GPU-стоппер в момент кейфрейма.
    roi_only: bool = True
    roi_pad_px: int = 24
    # Лёгкий промптовый трекер SAM3 (Sam3Tracker) между кейфреймами. Если
    # включён — держать его на openvino/CPU/iGPU, не на дискретке.
    use_tracker: bool = False


@dataclass
class KeyframeCfg:
    """Когда дёргать тяжёлый SAM-кейфрейм. Триггеры событийные, не по таймеру."""
    on_camera_wobble: bool = True       # покачивание камеры (шаг триангуляции)
    on_low_confidence: bool = True      # дрейф трекера / потеря точек optical flow
    min_good_features: int = 40         # меньше этого числа точек -> ресинк
    max_flow_residual_px: float = 6.0   # расхождение ожид-vs-факт -> ресинк
    min_frames_between: int = 8         # антидребезг: не чаще, чем раз в N кадров


@dataclass
class SemanticCfg:
    enabled: bool = False
    model_name: str = "segformer"
    weights_path: Optional[str] = None
    device: str = "cuda"
    # категория -> список меток датасета, которые в неё попадают
    monster_labels: Tuple[str, ...] = ("person", "animal")
    obstacle_labels: Tuple[str, ...] = ("tree", "rock", "building", "wall", "fence")


@dataclass
class ControlCfg:
    move_key: str = "w"
    jump_key: str = "space"
    sprint_key: str = "shift"
    # Детект застревания.
    stuck_window_s: float = 1.5
    stuck_min_displacement_m: float = 0.4
    # Допуск прибытия к точке маршрута.
    waypoint_tolerance_m: float = 2.0
    # Порог доворота: если |ошибка курса на цель| меньше — считаем, что смотрим
    # «достаточно на точку» и идём прямо; иначе сначала доворачиваем.
    align_tolerance_deg: float = 12.0
    # Руление при ЖИВОМ вождении (замкнутый контур по курсу-движения).
    turn_deadzone_deg: float = 5.0    # ошибка меньше — не подруливаем (без дёрганья)
    turn_gain: float = 0.5            # доля ошибки за один доворот (демпфирование, <1)
    turn_throttle_s: float = 0.20     # не чаще этого шлём доворот (курс обновляется с лагом)
    turn_max_units: int = 40          # максимум ед. мыши за доворот (мягко: бежать дугой, не пируэт)
    # Pure-pursuit: рулим не на ближнюю точку (там пеленг шумит на малой дистанции),
    # а на точку в ~lookahead_m впереди по пути — пеленг стабилен, бот не дёргается.
    lookahead_m: float = 6.0
    # Коридор впереди персонажа для проверки препятствий.
    corridor_half_width_m: float = 1.2
    corridor_length_m: float = 6.0
    input_backend: str = "pydirectinput"   # pydirectinput | pynput
    # dry-run: всё считается и логируется, но реальный ввод НЕ выполняется.
    dry_run: bool = False


@dataclass
class FusionCfg:
    # Доверие к источникам (дисперсии измерений, м^2).
    minimap_pos_var: float = 4.0     # миникарта — грубая абсолютная привязка
    flow_vel_var: float = 0.25       # оптический поток — точная относительная скорость
    process_var: float = 0.5


@dataclass
class Config:
    capture: CaptureCfg = field(default_factory=CaptureCfg)
    minimap: MinimapCfg = field(default_factory=MinimapCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    sam: SamCfg = field(default_factory=SamCfg)
    keyframe: KeyframeCfg = field(default_factory=KeyframeCfg)
    semantic: SemanticCfg = field(default_factory=SemanticCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    fusion: FusionCfg = field(default_factory=FusionCfg)

    @staticmethod
    def load(path: str = "config.yaml") -> "Config":
        cfg = Config()
        if yaml is None or not os.path.exists(path):
            return cfg
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for section, sub in data.items():
            if hasattr(cfg, section) and isinstance(sub, dict):
                obj = getattr(cfg, section)
                for k, v in sub.items():
                    if hasattr(obj, k):
                        # списки -> кортежи для совместимости с дефолтами
                        setattr(obj, k, tuple(v) if isinstance(v, list) else v)
        return cfg
