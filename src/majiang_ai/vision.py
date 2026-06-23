"""牌面图像识别模块。

将 capture.py 抓取的截图转换为编码后的牌面字符串 (如 123m555678p1122s)。

核心功能:
- 手牌区域分割 (connected components)
- 牌河牌型检测 (色块聚类)
- 单张牌的分类 (基于颜色直方图 + 结构特征)
- 模板校准: 支持从截图中采集每张牌的模板图片
- 全图模板匹配: 不依赖固定坐标，窗口缩放/UI变化自适应
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# 牌面编码定义
# ---------------------------------------------------------------------------

# 麻将牌的名称映射: "(rank, suit)" -> 中文名
TILE_NAMES: dict[tuple[int, str], str] = {}
for _suit_char, _suit_name in [("m", "万"), ("p", "筒"), ("s", "条")]:
    for _r in range(1, 10):
        TILE_NAMES[(_r, _suit_char)] = f"{_r}{_suit_name}"
for _r, _name in enumerate(("东", "南", "西", "北", "白", "发", "中"), start=1):
    TILE_NAMES[(_r, "z")] = _name

ALL_TILE_KEYS = sorted(TILE_NAMES.keys())


@dataclass
class RecognizedTile:
    rank: int
    suit: str
    code: str  # e.g. "3m"
    name: str  # e.g. "三万"
    confidence: float = 1.0

    def __repr__(self) -> str:
        return f"<{self.code} {self.name} conf={self.confidence:.2f}>"


@dataclass
class RecognitionResult:
    hand_tiles: list[RecognizedTile] = field(default_factory=list)
    river_tiles: list[RecognizedTile] = field(default_factory=list)
    hand_compact: str = ""
    river_compact: str = ""
    visible_compact: str = ""  # 所有可见牌 (牌河 + 对手打出的牌)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 颜色直方图特征 (区分万/筒/条/字牌)
# ---------------------------------------------------------------------------

# 各花色在 BGR 空间中的大致颜色范围 (腾讯麻将常见配色)
SUIT_COLOR_PROFILES = {
    "m": {  # 万子 - 深蓝/黑色大字
        "hue_range": (90, 140),
        "sat_range": (10, 100),
        "bright_range": (30, 100),
        "desc": "万",
    },
    "p": {  # 饼子 - 偏红/棕色
        "hue_range": (0, 20),
        "sat_range": (40, 200),
        "bright_range": (60, 200),
        "desc": "筒",
    },
    "s": {  # 条子 - 偏绿
        "hue_range": (35, 80),
        "sat_range": (30, 180),
        "bright_range": (50, 180),
        "desc": "条",
    },
    "z": {  # 字牌 - 红色大字
        "hue_range": (0, 10),
        "sat_range": (80, 255),
        "bright_range": (40, 150),
        "desc": "字",
    },
}


def _classify_suit_by_color(tile_crop: np.ndarray) -> tuple[str, float]:
    """根据牌面颜色直方图猜测花色。

    Returns (suit_guess, confidence)。
    当无法可靠判断时，返回 ("?", 0.0)。
    """
    if tile_crop.size == 0:
        return ("?", 0.0)
    try:
        import cv2

        hsv = cv2.cvtColor(tile_crop, cv2.COLOR_BGR2HSV)
    except Exception:
        return ("?", 0.0)

    # 提取牌面中心区域，避免边框干扰
    h, w = hsv.shape[:2]
    cy, cx = h // 2, w // 2
    r_h, r_w = int(h * 0.35), int(w * 0.35)
    y1 = max(0, cy - r_h)
    y2 = min(h, cy + r_h)
    x1 = max(0, cx - r_w)
    x2 = min(w, cx + r_w)
    center = hsv[y1:y2, x1:x2]

    # 只关注非白色区域 (排除牌面底色)
    mask = center[:, :, 1] > 20  # 饱和度 > 20
    if mask.sum() < 50:
        return ("?", 0.0)

    hue_vals = center[:, :, 0][mask]
    sat_vals = center[:, :, 1][mask]

    # 取 hue 中位数
    if len(hue_vals) == 0:
        return ("?", 0.0)
    median_hue = float(np.median(hue_vals))
    median_sat = float(np.median(sat_vals))

    # 字牌特征: 红色大字 = 低 hue (< 15) + 高饱和
    if median_hue < 25 and median_sat > 60:
        return ("z", 0.55)

    # 万子: 深蓝/黑色 (hue 100-130)
    if 85 < median_hue < 145:
        return ("m", 0.50)

    # 条子: 绿色 (hue 40-80)
    if 30 < median_hue < 85:
        return ("s", 0.45)

    # 饼子: 红/棕 (hue 0-25, 低饱和)
    if median_hue < 30:
        return ("p", 0.40)

    return ("?", 0.0)


# ---------------------------------------------------------------------------
# 全图模板匹配找牌 — 不依赖固定切割坐标
# ---------------------------------------------------------------------------


def _scan_hand_by_templates(
    hand_frame: np.ndarray,
    templates: dict[str, np.ndarray],
    min_conf: float = 0.38,
) -> tuple[list[np.ndarray], list[str]]:
    """全图模板匹配找所有牌的位置，不依赖固定坐标。

    Returns:
        (crops, codes) — 切出的牌图块和对应编码。
        窗口缩放、UI 变化、牌数变化都自适应。
    """
    import cv2

    if not templates:
        return [], []

    gray = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    detections: list[tuple[int, int, int, int, str, float]] = []

    # 先用所有缩放因子尝试，取匹配数量最多的
    best_detections = []

    for scale in [1.0, 0.95, 0.90, 0.85]:
        sc_tmpl: dict[str, np.ndarray] = {}
        for code, tmpl in templates.items():
            tmpl_h, tmpl_w = tmpl.shape[:2]
            if scale != 1.0:
                new_w, new_h = int(tmpl_w * scale), int(tmpl_h * scale)
                if new_w < 10 or new_h < 10:
                    continue
                tmpl = cv2.resize(tmpl, (new_w, new_h))
            if tmpl.shape[0] > h or tmpl.shape[1] > w:
                continue
            sc_tmpl[code] = tmpl

        scale_dets: list[tuple[int, int, int, int, str, float]] = []
        for code, tmpl in sc_tmpl.items():
            tmpl_h, tmpl_w = tmpl.shape[:2]
            tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY) if len(tmpl.shape) == 3 else tmpl

            try:
                result = cv2.matchTemplate(gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue

            locations = np.where(result >= min_conf)
            for py, px in zip(*locations):
                conf = float(result[py, px])
                scale_dets.append((px, py, px + tmpl_w, py + tmpl_h, code, conf))

        if not scale_dets:
            continue

        # NMS
        scale_dets.sort(key=lambda d: -d[5])
        keep: list[tuple[int, int, int, int, str, float]] = []
        for d in scale_dets:
            x1, y1, x2, y2, _, _ = d
            area = (x2 - x1) * (y2 - y1)
            if area <= 0:
                continue
            overlap = False
            for k in keep:
                kx1, ky1, kx2, ky2, _, _ = k
                ox1, ox2 = max(x1, kx1), min(x2, kx2)
                oy1, oy2 = max(y1, ky1), min(y2, ky2)
                if ox1 < ox2 and oy1 < oy2:
                    if (ox2 - ox1) * (oy2 - oy1) > area * 0.5:
                        overlap = True
                        break
            if not overlap:
                keep.append(d)

        if len(keep) > len(best_detections):
            best_detections = keep

    if not best_detections:
        return [], []

    # 按 x 排序
    best_detections.sort(key=lambda d: d[0])

    tile_crops = []
    tile_codes = []
    for x1, y1, x2, y2, code, _conf in best_detections:
        crop = hand_frame[y1:y2, x1:x2]
        if crop.size > 0:
            tile_crops.append(crop)
            tile_codes.append(code)

    return tile_crops, tile_codes


# ---------------------------------------------------------------------------
# 传统分割 (固定坐标优先，自动检测降级)
# ---------------------------------------------------------------------------


def segment_hand_strip(
    hand_frame: np.ndarray,
    expected_count: int = 14,
    tile_bounds: list | None = None,
    templates: dict | None = None,
) -> list[np.ndarray]:
    """从手牌区域截图中纵向切分出每张独立牌的图块。

    优先级:
    1. tile_bounds 精确坐标
    2. templates 全图模板匹配（自适应，不需要每局重调）
    3. 列投影 + Otsu 自动检测（降级）
    """
    import cv2

    h, w = hand_frame.shape[:2]

    # ---- 优先级 1: 精确切割坐标 ----
    if tile_bounds and len(tile_bounds) > 0:
        # 检测坐标类型：全<2 → 相对比例，否则 → 绝对像素
        is_fraction = all(tb["right"] < 2.0 for tb in tile_bounds)
        tile_crops = []
        for tb in tile_bounds:
            if is_fraction:
                x1 = int(tb["left"] * w)
                x2 = int(tb["right"] * w)
            else:
                x1 = int(tb["left"])
                x2 = int(tb["right"])
            crop = hand_frame[:, max(0, x1):min(w, x2)]
            if crop.size > 0 and crop.shape[1] > 5:
                tile_crops.append(crop)
        if len(tile_crops) >= 3:
            return tile_crops

    # ---- 优先级 2: 全图模板匹配 ----
    if templates and len(templates) >= 4:
        crops, _codes = _scan_hand_by_templates(hand_frame, templates)
        if crops and len(crops) >= 3:
            return crops

    # ---- 优先级 3: 列投影自动检测 ----
    gray = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2GRAY)

    # 步骤1: Otsu 二值化 + 列投影
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    col_profile = np.sum(binary < 128, axis=0) / h
    mean_dark = float(np.mean(col_profile))
    threshold = max(0.03, min(0.15, mean_dark * 0.5))
    col_mask = col_profile > threshold

    # 步骤2: 找连续区域
    segments: list[tuple[int, int]] = []
    in_segment = False
    start = 0
    min_width = max(10, w // 20)
    max_gap = max(5, w // 30)

    x = 0
    while x < w:
        if col_mask[x] and not in_segment:
            in_segment = True
            start = x
        elif not col_mask[x] and in_segment:
            gap_end = x
            while gap_end < w and not col_mask[gap_end]:
                gap_end += 1
            gap_size = gap_end - x
            if gap_size <= max_gap and gap_end < w and col_mask[gap_end]:
                x = gap_end
                continue
            else:
                if x - start >= min_width:
                    segments.append((start, x))
                in_segment = False
        x += 1

    if in_segment and w - start >= min_width:
        segments.append((start, w))

    # 等宽降级
    if not segments:
        tile_w = w // max(1, expected_count)
        segments = [(i * tile_w, (i + 1) * tile_w) for i in range(expected_count)]
    else:
        widths = [x2 - x1 for x1, x2 in segments]
        avg_w = sum(widths) / len(widths)
        too_wide = any(w > avg_w * 2.5 for w in widths)
        too_few = len(segments) < max(2, expected_count // 2)
        if too_few or too_wide:
            tile_w = w // max(1, expected_count)
            segments = [(i * tile_w, (i + 1) * tile_w) for i in range(expected_count)]

    tile_crops = []
    for x1, x2 in segments:
        crop = hand_frame[:, max(0, x1):min(w, x2)]
        if crop.size > 0:
            tile_crops.append(crop)
    return tile_crops


# ---------------------------------------------------------------------------
# 数字提取 & 主识别流程
# ---------------------------------------------------------------------------


def _digit_from_image(crop: np.ndarray) -> tuple[Optional[int], float]:
    """从单张牌图片中提取数字 (1-9)，基于连通组件数量。"""
    import cv2

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = binary.shape
    edge = binary[:5, :].mean()
    center = binary[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean()
    if center > edge:
        binary = 255 - binary

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    min_area = (w * h) * 0.005
    max_area = (w * h) * 0.6
    valid_components = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            valid_components += 1

    if 1 <= valid_components <= 9:
        return (valid_components, 0.7)
    elif valid_components > 9:
        if valid_components <= 15:
            return (min(9, valid_components - 3), 0.35)
        return (None, 0.0)
    elif valid_components == 0:
        min_area2 = (w * h) * 0.002
        retry = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area2 < area < max_area:
                retry += 1
        if 1 <= retry <= 9:
            return (retry, 0.35)
        return (None, 0.0)
    return (None, 0.0)


def recognize_tiles(
    hand_frame: np.ndarray,
    river_frame: Optional[np.ndarray] = None,
    discards: Optional[list[np.ndarray]] = None,
    templates: Optional[dict[str, np.ndarray]] = None,
    tile_bounds: list | None = None,
) -> RecognitionResult:
    """从截图中识别麻将牌面。

    Args:
        hand_frame: 手牌区域截图 (BGR)。
        river_frame: 牌河区域截图 (可选)。
        discards: 已单独切出的出牌截图列表 (可选)。
        templates: 牌面模板 {code: image}，优先使用模板匹配。
        tile_bounds: 每张牌相对手牌区域的 left/right 坐标 (可选)。
    Returns:
        RecognitionResult 包含识别出的牌面和紧凑编码。
    """
    result = RecognitionResult()

    # ----- 手牌 -----
    tile_crops = segment_hand_strip(hand_frame, tile_bounds=tile_bounds, templates=templates)

    def _valid_rank(rank: int, suit: str) -> bool:
        if suit == "z":
            return 1 <= rank <= 7
        return 1 <= rank <= 9

    _use_templates = templates is not None and len(templates) > 0

    hand_tiles: list[RecognizedTile] = []
    for crop in tile_crops:
        if _use_templates:
            code, tmpl_conf = match_with_templates(crop, templates)
            if code != "?" and tmpl_conf >= 0.5:
                try:
                    rank = int(code[:-1])
                    suit = code[-1]
                except ValueError:
                    rank, suit = None, "?"
                if rank is not None and _valid_rank(rank, suit):
                    name = TILE_NAMES.get((rank, suit), code)
                    hand_tiles.append(RecognizedTile(rank=rank, suit=suit, code=code, name=name, confidence=tmpl_conf))
                    continue

        # 降级: 启发式识别
        suit, suit_conf = _classify_suit_by_color(crop)
        rank, rank_conf = _digit_from_image(crop)

        if suit == "?" or rank is None:
            continue
        if not _valid_rank(rank, suit):
            continue

        code = f"{rank}{suit}"
        name = TILE_NAMES.get((rank, suit), code)
        conf = (suit_conf * 0.4 + rank_conf * 0.6) if rank_conf > 0 else suit_conf
        hand_tiles.append(RecognizedTile(rank=rank, suit=suit, code=code, name=name, confidence=conf))

    result.hand_tiles = hand_tiles

    if hand_tiles:
        groups: dict[str, list[str]] = {s: [] for s in "mpsz"}
        for t in hand_tiles:
            groups[t.suit].append(str(t.rank))
        parts = []
        for suit in "mpsz":
            if groups[suit]:
                parts.append("".join(sorted(groups[suit])) + suit)
        result.hand_compact = "".join(parts)
    else:
        result.errors.append("手牌识别失败：未能识别出任何有效牌面。")

    # ----- 牌河 -----
    if river_frame is not None and river_frame.size > 0:
        river_tiles = _detect_river_tiles(river_frame)
        result.river_tiles = river_tiles
        if river_tiles:
            groups: dict[str, list[str]] = {s: [] for s in "mpsz"}
            for t in river_tiles:
                groups[t.suit].append(str(t.rank))
            parts = []
            for suit in "mpsz":
                if groups[suit]:
                    parts.append("".join(sorted(groups[suit])) + suit)
            result.river_compact = "".join(parts)

    # ----- 综合可见牌 -----
    visible_codes: list[str] = []
    for t in result.river_tiles:
        visible_codes.append(t.code)
    if discards:
        for disc in discards:
            suit, _ = _classify_suit_by_color(disc)
            rank, _ = _digit_from_image(disc)
            if suit != "?" and rank is not None:
                if suit == "z" and not (1 <= rank <= 7):
                    continue
                if suit != "z" and not (1 <= rank <= 9):
                    continue
                visible_codes.append(f"{rank}{suit}")

    result.visible_compact = ",".join(visible_codes) if visible_codes else ""
    return result


def _detect_river_tiles(river_frame: np.ndarray) -> list[RecognizedTile]:
    """从牌河截图中检测已打出的牌 (简化版: 色块轮廓检测)。"""
    import cv2

    tiles: list[RecognizedTile] = []
    gray = cv2.cvtColor(river_frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = river_frame.shape[:2]
    min_area = (w * h) * 0.003
    max_area = (w * h) * 0.08

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(1, ch)
        if aspect < 0.4 or aspect > 2.5:
            continue
        crop = river_frame[y : y + ch, x : x + cw]
        suit, _ = _classify_suit_by_color(crop)
        rank, _ = _digit_from_image(crop)
        if suit != "?" and rank is not None:
            if suit == "z" and not (1 <= rank <= 7):
                continue
            if suit != "z" and not (1 <= rank <= 9):
                continue
            code = f"{rank}{suit}"
            name = TILE_NAMES.get((rank, suit), code)
            tiles.append(RecognizedTile(rank=rank, suit=suit, code=code, name=name, confidence=0.5))
    return tiles


# ---------------------------------------------------------------------------
# 模板管理
# ---------------------------------------------------------------------------


def capture_templates(capturer, output_dir: str = "tile_templates") -> dict[str, np.ndarray]:
    """交互式采集每张牌的模板图片。"""
    import cv2

    os.makedirs(output_dir, exist_ok=True)

    print("=== 牌面模板采集 (交互式) ===")
    print("正在截取当前手牌...")

    hand = capturer.capture_hand()
    crops = segment_hand_strip(hand, expected_count=14)

    if not crops:
        print("错误: 未能从手牌区域分割出牌面。")
        return {}

    print(f"\n已分割出 {len(crops)} 张牌。")
    print("请按从左到右的顺序，依次输入每张牌的编码。")
    print("编码格式: 1m~9m(万), 1p~9p(筒), 1s~9s(条), 1z~7z(字)")
    print("字牌对应: 1z=东, 2z=南, 3z=西, 4z=北, 5z=白, 6z=发, 7z=中")
    print("输入 's' 跳过当前牌，输入 'q' 结束采集。\n")

    templates: dict[str, np.ndarray] = {}

    for i, crop in enumerate(crops):
        preview_path = os.path.join(output_dir, f"numbered_slot_{i+1:02d}.png")
        cv2.imwrite(preview_path, crop)
        print(f"  牌位 #{i+1} 已保存为: {preview_path}")

    print(f"\n共 {len(crops)} 张牌待标注。请查看 {output_dir}/ 目录的预览图。\n")

    for i in range(len(crops)):
        preview_path = os.path.join(output_dir, f"numbered_slot_{i+1:02d}.png")
        while True:
            code = input(f"  牌位 #{i+1} ({preview_path}): ").strip().lower()
            if code == 's':
                print("    已跳过")
                break
            if code == 'q':
                return templates
            if len(code) >= 2 and code[-1] in "mpsz" and code[:-1].isdigit():
                rank = int(code[:-1])
                suit = code[-1]
                if suit == "z" and not (1 <= rank <= 7):
                    print(f"    无效编码: {code}")
                    continue
                if suit != "z" and not (1 <= rank <= 9):
                    print(f"    无效编码: {code}")
                    continue
                template_path = os.path.join(output_dir, f"{code}.png")
                crop = cv2.imread(preview_path)
                if crop is not None:
                    cv2.imwrite(template_path, crop)
                    templates[code] = crop
                    print(f"    已保存: {code} -> {template_path}")
                break
            else:
                print(f"    无效格式，请使用如 1m, 2p, 7z 的格式")

    print(f"\n采集完成！共 {len(templates)} 张模板。")
    return templates


def load_templates(template_dir: str) -> dict[str, np.ndarray]:
    """从目录加载已保存的模板图片。只加载编码命名的文件（如 3m.png），忽略 slot_NN。"""
    import cv2
    import re

    templates: dict[str, np.ndarray] = {}
    if not os.path.isdir(template_dir):
        return templates
    valid_code = re.compile(r'^[1-9][mpsz]$')
    for fname in sorted(os.listdir(template_dir)):
        if fname.endswith((".png", ".jpg", ".jpeg")):
            code = os.path.splitext(fname)[0]
            if not valid_code.match(code):
                continue
            path = os.path.join(template_dir, fname)
            img = cv2.imread(path)
            if img is not None:
                templates[code] = img
    return templates


def match_with_templates(tile_crop: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[str, float]:
    """用模板匹配识别单张牌。

    Returns (code, confidence)。
    """
    import cv2

    if not templates:
        return ("?", 0.0)

    best_code = "?"
    best_score = 0.0

    for code, tmpl in templates.items():
        if tmpl.shape[0] > tile_crop.shape[0] or tmpl.shape[1] > tile_crop.shape[1]:
            tmpl_resized = cv2.resize(tmpl, (tile_crop.shape[1], tile_crop.shape[0]))
        else:
            tmpl_resized = tmpl

        try:
            result = cv2.matchTemplate(tile_crop, tmpl_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_code = code
        except cv2.error:
            continue

    return (best_code, float(best_score))
