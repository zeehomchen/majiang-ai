"""牌面图像识别模块。

将 capture.py 抓取的截图转换为编码后的牌面字符串 (如 123m555678p1122s)。

核心功能:
- 手牌区域分割 (connected components)
- 牌河牌型检测 (色块聚类)
- 单张牌的分类 (基于颜色直方图 + 结构特征)
- 模板校准: 支持从截图中采集每张牌的模板图片
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

    # 根据色调范围判断花色
    best_suit = "?"
    best_score = 0.0

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
# 手牌分割 -- 从横向手牌截图中分离出每张独立的牌
# ---------------------------------------------------------------------------


def segment_hand_strip(hand_frame: np.ndarray, expected_count: int = 14) -> list[np.ndarray]:
    """从手牌区域截图中纵向切分出每张独立牌的图块。

    假设手牌是从左到右紧密排列的，每张牌宽度大致相等。
    实际手牌可能 13 或 14 张 (摸牌后)。
    """
    import cv2

    gray = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 自适应阈值，把牌从背景中分离出来
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 纵向投影 -- 计算每列的非零像素数
    col_profile = np.sum(binary < 128, axis=0) / h  # 暗像素比例 (牌面内容)
    col_profile = col_profile > 0.15  # 至少有 15% 暗像素才算有牌

    # 找连续区域
    segments: list[tuple[int, int]] = []
    in_segment = False
    start = 0
    min_width = max(10, w // 20)  # 最小牌宽

    for x in range(w):
        if col_profile[x] and not in_segment:
            in_segment = True
            start = x
        elif not col_profile[x] and in_segment:
            if x - start >= min_width:
                segments.append((start, x))
            in_segment = False
    if in_segment and w - start >= min_width:
        segments.append((start, w))

    # 如果分割结果超出预期，按宽度合并/拆分
    if not segments:
        # 退而求其次: 按等宽切分
        tile_w = w // max(1, expected_count)
        segments = [(i * tile_w, (i + 1) * tile_w) for i in range(expected_count)]

    # 提取每张牌
    tile_crops = []
    for x1, x2 in segments:
        crop = hand_frame[:, max(0, x1) : min(w, x2)]
        if crop.size > 0:
            tile_crops.append(crop)

    return tile_crops


def _digit_from_image(crop: np.ndarray) -> tuple[Optional[int], float]:
    """从单张牌图片中提取数字 (1-9)。

    基于区域的连通组件数量 (饼子) 或笔画宽度 (万/条/字)。
    当前为启发式实现，返回 (rank_or_none, confidence)。
    """
    import cv2

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 反转: 让数字区域变成白色前景
    h, w = binary.shape
    edge = binary[:5, :].mean()
    center = binary[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean()
    if center > edge:
        binary = 255 - binary

    # 找连通组件
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 过滤掉太小或太大的组件
    min_area = (w * h) * 0.005
    max_area = (w * h) * 0.6
    valid_components = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            valid_components += 1

    # 对于筒子，连通组件数量直接对应数字
    if 1 <= valid_components <= 9:
        return (valid_components, 0.7)
    elif valid_components > 9:
        # 可能是万子/条子 (有额外笔画)
        return (None, 0.0)

    return (None, 0.0)


# ---------------------------------------------------------------------------
# 主识别流程
# ---------------------------------------------------------------------------


def recognize_tiles(
    hand_frame: np.ndarray,
    river_frame: Optional[np.ndarray] = None,
    discards: Optional[list[np.ndarray]] = None,
) -> RecognitionResult:
    """从截图中识别麻将牌面。

    Args:
        hand_frame: 手牌区域截图 (BGR)。
        river_frame: 牌河区域截图 (可选)。
        discards: 已单独切出的出牌截图列表 (可选)。

    Returns:
        RecognitionResult 包含识别出的牌面和紧凑编码。
    """
    result = RecognitionResult()

    # ----- 手牌 -----
    tile_crops = segment_hand_strip(hand_frame)

    hand_tiles: list[RecognizedTile] = []
    for crop in tile_crops:
        suit, suit_conf = _classify_suit_by_color(crop)
        rank, rank_conf = _digit_from_image(crop)

        if suit == "?" or rank is None:
            continue

        code = f"{rank}{suit}"
        name = TILE_NAMES.get((rank, suit), code)
        conf = (suit_conf * 0.4 + rank_conf * 0.6) if rank_conf > 0 else suit_conf

        hand_tiles.append(RecognizedTile(rank=rank, suit=suit, code=code, name=name, confidence=conf))

    result.hand_tiles = hand_tiles

    if hand_tiles:
        # 按花色分组构建紧凑编码
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

    # ----- 牌河 (简化: 直接统计颜色区域数量作为可见牌) -----
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

    # ----- 综合可见牌 (手牌 = 只有自己能看到的，这里不暴露) -----
    visible_codes: list[str] = []
    for t in result.river_tiles:
        visible_codes.append(t.code)
    if discards:
        for disc in discards:
            # 简单地从弃牌指示区检测最后打出的牌
            suit, _ = _classify_suit_by_color(disc)
            rank, _ = _digit_from_image(disc)
            if suit != "?" and rank is not None:
                visible_codes.append(f"{rank}{suit}")

    result.visible_compact = ",".join(visible_codes) if visible_codes else ""

    return result


def _detect_river_tiles(river_frame: np.ndarray) -> list[RecognizedTile]:
    """从牌河截图中检测已打出的牌 (简化版: 色块轮廓检测)。"""
    import cv2

    tiles: list[RecognizedTile] = []
    gray = cv2.cvtColor(river_frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # 找轮廓
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
            code = f"{rank}{suit}"
            name = TILE_NAMES.get((rank, suit), code)
            tiles.append(RecognizedTile(rank=rank, suit=suit, code=code, name=name, confidence=0.5))

    return tiles


# ---------------------------------------------------------------------------
# 模板校准 -- 采集每张牌的模板图片
# ---------------------------------------------------------------------------


def capture_templates(capturer, output_dir: str = "tile_templates") -> dict[str, np.ndarray]:
    """交互式采集每张牌的模板图片。

    需要用户手动把每张牌拖到视窗内并确认。
    流程:
        1. 抓取当前手牌截图
        2. 分割出每张牌
        3. 让用户确认每张牌的内容
        4. 保存为模板

    Args:
        capturer: Capturer 实例 (已启动)。
        output_dir: 模板保存目录。

    Returns:
        {code: image} 字典。
    """
    import cv2

    os.makedirs(output_dir, exist_ok=True)

    print("=== 牌面模板采集 ===")
    print("请在游戏中把要采集的牌放在手牌区域...")

    hand = capturer.capture_hand()
    crops = segment_hand_strip(hand)

    templates: dict[str, np.ndarray] = {}

    # 提示用户按顺序确认每张牌
    tile_map = [
        "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
        "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
        "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
        "1z", "2z", "3z", "4z", "5z", "6z", "7z",
    ]

    idx = 0
    while idx < min(len(crops), len(tile_map)):
        code = tile_map[idx]
        crop = crops[idx]
        filename = os.path.join(output_dir, f"{code}.png")
        cv2.imwrite(filename, crop)
        templates[code] = crop
        print(f"  已保存模板: {code} -> {filename}")
        idx += 1

    print(f"共采集 {len(templates)} 张模板，保存在 {output_dir}/")
    return templates


def load_templates(template_dir: str) -> dict[str, np.ndarray]:
    """从目录加载已保存的模板图片。"""
    import cv2

    templates: dict[str, np.ndarray] = {}
    if not os.path.isdir(template_dir):
        return templates
    for fname in sorted(os.listdir(template_dir)):
        if fname.endswith((".png", ".jpg", ".jpeg")):
            code = os.path.splitext(fname)[0]
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
