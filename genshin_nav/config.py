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
    # Источник курса в read(): "arrow" — продакшн (cyan-стрелка «центроид→дальняя
    # точка» + круговая медиана), "tracker" — внешний MinimapTracker (ось по
    # минимуму поперечной дисперсии + разрешение 180° по конусу FOV). Сравнить
    # детекторы: scripts/compare_heading.py. Переключать на "tracker" — только
    # если он реально выиграл на --live.
    heading_source: str = "arrow"
    # Поправка конвенции для MinimapTracker: его компас отличается от arrow на
    # константу. Рекомендованное значение печатает compare_heading.py.
    tracker_heading_offset_deg: float = 0.0
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
    # Знак: переводит "курс по компасу, по часовой" в знак физического движения
    # мыши по X. 1 = положительный dyaw поворачивает камеру вправо/по часовой;
    # -1 = инвертировано (нужно, если на твоей системе/в игре поворот выходит
    # в противоположную сторону от расчётной). КАЛИБРОВАТЬ: довернуть камеру
    # на цель справа -> если уходит влево, поставить -1.
    yaw_sign: float = 1.0


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
    turn_deadzone_deg: float = 8.0    # МЯГКАЯ зона (вычитается из ошибки в _steer): внутри руль
                                      # молчит (едем прямо), у края команда плавно растёт от 0 —
                                      # гасит микроповороты от остаточного промаха БЕЗ чанковых рывков
    turn_gain: float = 0.8            # доля ошибки за один доворот (курс по стрелке без лага)
    turn_throttle_s: float = 0.20     # не чаще этого шлём доворот (даём персонажу довернуться)
    # Скорость для DEAD-RECKONING позиции (единиц маршрута/сек). Сдвиг карты на этом
    # зуме слепой → позицию интегрируем из курса×скорость. Подстроить: точки проходит
    # слишком рано → уменьшить; слишком поздно/не доезжает → увеличить.
    dead_reckon_speed: float = 5.5
    # Источник позиции при следовании:
    #   "odometry"    — та же одометрия миникарты (pose.player_xy), что и при ЗАПИСИ
    #                   маршрута. Маршрут и следование в ОДНОЙ системе координат —
    #                   нет зеркала/поворота между записью и прогоном (РЕКОМЕНДУЕТСЯ).
    #   "dead_reckon" — позиция из курс×скорость (старое поведение; уходит в зеркало,
    #                   если запись была в одометрии).
    position_source: str = "odometry"
    # АБСОЛЮТНАЯ локализация по миникарте (шаг 1 ТЗ, teach-and-repeat). На прогоне
    # сравнивает живую миникарту с отпечатками точек (record → <маршрут>.fp.npz) и
    # подтягивает позицию к фактическому месту — скорость/стамина/дрейф перестают
    # влиять. Работает поверх dead_reckon (комплементарно). Нужны отпечатки в маршруте.
    use_minimap_localize: bool = False
    localize_threshold: float = 0.55     # мин. корреляция отпечатков, чтобы доверять (0..1)
    # Окно поиска — по ДИСТАНЦИИ (м) вокруг текущей оценки: достаточно широкое, чтобы
    # поймать накопленный дрейф, но не пускающее ложные совпадения из других мест.
    # На зуме ~4 м/px разрешение грубое (~15-30 м) — это ловит крупный дрейф, не метры.
    localize_radius_m: float = 40.0
    localize_correct_gain: float = 0.34  # доля коррекции позиции к найденной точке за кадр (0..1)
    # Бег с зажатым Shift (спринт). ВНИМАНИЕ: в Genshin спринт тратит стамину —
    # на длинных отрезках выгода ограничена (стамина кончится → обычный бег).
    sprint: bool = False
    # Спринтовать ТОЛЬКО когда ошибка курса мала (едем ~прямо). На крутых поворотах
    # и стартовом развороте Shift отпускается → персонаж идёт шагом → тугой разворот
    # с малым заносом (на спринте большой разворот уводит на десятки метров вбок).
    # Заодно при возврате к спринту даётся свежий фронт нажатия Shift.
    sprint_max_err_deg: float = 30.0
    # Подстройка курса (градусы) для компенсации СТАБИЛЬНОГО увода в сторону.
    # Прибавляется к курсу. Если бот стабильно уходит ВПРАВО — ставь ПОЛОЖИТЕЛЬНОЕ
    # (например +2..+4); если влево — отрицательное. Если стало хуже — поменяй знак.
    heading_trim_deg: float = 0.0
    # --- Активация телепорта на точке маршрута с action="teleport" ---
    # Сценарий: доехал до точки → стоп → жмём teleport_activate_key (F) → ждём меню
    # → ЛКМ по teleport_dismiss_xy (синий ромбик «Точка телепортации») → ждём → дальше.
    teleport_activate_key: str = "f"
    teleport_dismiss_xy: Tuple[int, int] = (960, 540)  # ЭКРАННЫЕ коорд. ромбика; КАЛИБРОВАТЬ: scripts/calibrate_click.py
    teleport_wait_menu_s: float = 1.5      # пауза после F (дать всплывашке появиться)
    teleport_after_s: float = 1.2          # пауза после клика, прежде чем ехать дальше
    teleport_tolerance_m: float = 2.0      # на каком расстоянии до точки телепорта срабатывает (фолбэк по позиции)
    # КАРАБКАНЬЕ (action="climb"): подъём на стену/скалу по времени (высоту миникарта
    # не видит). dur берётся из точки маршрута; ниже — общие настройки исполнения.
    climb_tolerance_m: float = 2.5         # на каком расстоянии до climb-точки начинаем лезть
    climb_jump: bool = True                # прыгать (Space) перед подъёмом и на выступах
    climb_jump_interval_s: float = 1.5     # период до-прыжков во время карабканья (сек)
    # НАДЁЖНЫЙ триггер телепорта: на подъезде к точке искать на экране подсказку
    # «F Точка телепортации» (template-match по картинке) и жать F, когда она реально
    # видна — не зависит от дрейфа dead-reckon. Если шаблон не найден на диске —
    # тихо откатывается на триггер по расстоянию (teleport_tolerance_m).
    teleport_detect_template: bool = True
    teleport_template_path: str = "Fteleport.png"   # путь от корня проекта
    teleport_template_threshold: float = 0.7        # 0..1, выше = строже совпадение
    teleport_detect_radius_m: float = 8.0           # с какого расстояния начинать искать подсказку
    # --- Фаст-тревел через карту (после F): M → карта → клики по картинкам ---
    teleport_open_map_key: str = "m"                # клавиша открыть карту мира
    teleport_map_wait_s: float = 1.0                # пауза после M (карта открывается не сразу)
    teleport_step_wait_s: float = 0.8               # пауза между кликами в меню карты
    teleport_click_threshold: float = 0.7           # порог совпадения для кликов по картинкам
    teleport_rombik_template: str = "rombik2.png"       # жёлтый ромбик БЕЗ синего кристалла (обычная точка на карте)
    teleport_point_template: str = "tockateleport.png"  # панель «Точка телепортации» — клик по иконке СЛЕВА
    teleport_point_click_dx: int = 30               # смещение клика от лев-верх угла панели (иконка слева)
    teleport_point_click_dy: int = 42
    teleport_confirm_template: str = "teleport.png"     # кнопка «Телепорт» снизу — клик по центру
    turn_max_units: int = 120         # потолок ед. мыши за доворот ≈14°/такт. 40 насыщался и
                                      # не закрывал промах; 700 (старый пивот) перелетал. 120 —
                                      # закрывает типовой промах ~25° за 2-3 шага без перелёта
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
