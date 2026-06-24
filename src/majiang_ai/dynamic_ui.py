"""
动态自适应UI识别模块
支持基于牌体特征的手牌/副露识别
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class TileType(Enum):
    """牌的类型"""
    HAND = "hand"  # 手牌
    MELD = "meld"  # 副露
    UNKNOWN = "unknown"


@dataclass
class TileDetection:
    """单张牌检测结果"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    tile_type: TileType
    confidence: float
    aspect_ratio: float  # 宽高比
    height_ratio: float  # 高度比例（相对于区域高度）


@dataclass
class DynamicUIAnalysis:
    """动态UI分析结果"""
    hand_tiles: List[TileDetection]  # 手牌列表
    meld_tiles: List[TileDetection]  # 副露列表
    split_x: Optional[int]  # 分割点X坐标（手牌/副露边界）
    hand_region: Tuple[int, int, int, int]  # 手牌区域
    meld_region: Tuple[int, int, int, int]  # 副露区域


class DynamicUIRecognizer:
    """
    动态UI识别器
    支持基于牌体特征的手牌/副露自动识别
    """

    def __init__(
        self,
        hand_height_threshold: float = 0.7,  # 手牌高度阈值（相对于区域高度）
        meld_height_threshold: float = 0.5,  # 副露高度阈值
        hand_aspect_ratio: float = 0.77,  # 手牌标准宽高比 (约1:1.3)
        meld_aspect_ratio: float = 1.3,  # 副露碰牌标准宽高比 (约1.3:1)
        debug: bool = False
    ):
        self.hand_height_threshold = hand_height_threshold
        self.meld_height_threshold = meld_height_threshold
        self.hand_aspect_ratio = hand_aspect_ratio
        self.meld_aspect_ratio = meld_aspect_ratio
        self.debug = debug

    def analyze_bottom_region(
        self,
        region_image: np.ndarray
    ) -> DynamicUIAnalysis:
        """
        分析底部持牌区域，自动分割手牌和副露

        Args:
            region_image: 底部持牌区域的BGR图像

        Returns:
            DynamicUIAnalysis分析结果
        """
        if not CV2_AVAILABLE:
            raise ImportError("需要opencv-python: pip install opencv-python")

        h, w = region_image.shape[:2]

        # 步骤1: 检测所有牌的轮廓
        tile_bboxes = self._detect_tile_contours(region_image)

        if not tile_bboxes:
            return DynamicUIAnalysis(
                hand_tiles=[],
                meld_tiles=[],
                split_x=None,
                hand_region=(0, 0, 0, 0),
                meld_region=(0, 0, 0, 0)
            )

        # 步骤2: 对每个牌进行分类（手牌/副露）
        tile_detections: List[TileDetection] = []
        for bbox in tile_bboxes:
            x1, y1, x2, y2 = bbox
            tile_w = x2 - x1
            tile_h = y2 - y1
            aspect_ratio = tile_w / max(tile_h, 1)
            height_ratio = tile_h / h

            tile_type, confidence = self._classify_tile(aspect_ratio, height_ratio)
            tile_detections.append(TileDetection(
                bbox=bbox,
                tile_type=tile_type,
                confidence=confidence,
                aspect_ratio=aspect_ratio,
                height_ratio=height_ratio
            ))

        # 步骤3: 找到手牌和副露的分割点
        split_x = self._find_split_point(tile_detections, w)

        # 步骤4: 根据分割点重新分类（增强鲁棒性）
        hand_tiles, meld_tiles = self._split_by_x(tile_detections, split_x)

        # 步骤5: 确定手牌和副露区域
        hand_region = self._calculate_region(hand_tiles, w, h, left=True)
        meld_region = self._calculate_region(meld_tiles, w, h, left=False)

        return DynamicUIAnalysis(
            hand_tiles=hand_tiles,
            meld_tiles=meld_tiles,
            split_x=split_x,
            hand_region=hand_region,
            meld_region=meld_region
        )

    def _detect_tile_contours(
        self,
        image: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        检测牌的轮廓

        Args:
            image: 输入图像

        Returns:
            牌的bbox列表 [(x1, y1, x2, y2), ...]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值处理
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 形态学操作去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        bboxes: List[Tuple[int, int, int, int]] = []
        h, w = image.shape[:2]
        min_area = (w * h) * 0.01  # 最小面积
        max_area = (w * h) * 0.15  # 最大面积
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                # 过滤异常宽高比
                aspect_ratio = bw / max(bh, 1)
                if 0.3 < aspect_ratio < 2.5:
                    bboxes.append((x, y, x + bw, y + bh))
        
        # 按X坐标排序
        bboxes.sort(key=lambda b: b[0])
        return bboxes

    def _classify_tile(
        self,
        aspect_ratio: float,
        height_ratio: float
    ) -> Tuple[TileType, float]:
        """
        根据宽高比和高度比分类牌的类型

        Args:
            aspect_ratio: 宽高比
            height_ratio: 高度比例

        Returns:
            (TileType, confidence)
        """
        # 计算手牌相似度
        hand_ar_diff = abs(aspect_ratio - self.hand_aspect_ratio)
        hand_height_score = 1.0 if height_ratio >= self.hand_height_threshold else height_ratio / self.hand_height_threshold
        hand_confidence = max(0.0, 1.0 - hand_ar_diff * 0.5) * hand_height_score

        # 计算副露相似度
        meld_ar_diff = abs(aspect_ratio - self.meld_aspect_ratio)
        meld_height_score = 1.0 if height_ratio <= self.meld_height_threshold else (1.0 - (height_ratio - self.meld_height_threshold))
        meld_confidence = max(0.0, 1.0 - meld_ar_diff * 0.5) * max(0.0, meld_height_score)

        if hand_confidence > meld_confidence:
            return TileType.HAND, hand_confidence
        elif meld_confidence > hand_confidence:
            return TileType.MELD, meld_confidence
        else:
            return TileType.UNKNOWN, 0.5

    def _find_split_point(
        self,
        detections: List[TileDetection],
        image_width: int
    ) -> Optional[int]:
        """
        找到手牌和副露的分割点

        策略：找到手牌和副露之间的最大间隙

        Args:
            detections: 牌检测列表
            image_width: 图像宽度

        Returns:
            分割点X坐标
        """
        if len(detections) < 2:
            return image_width // 2

        # 按X坐标排序
        sorted_dets = sorted(detections, key=lambda d: d.bbox[0])

        max_gap = 0
        split_x = image_width // 2

        for i in range(1, len(sorted_dets)):
            prev_x2 = sorted_dets[i-1].bbox[2]
            curr_x1 = sorted_dets[i].bbox[0]
            gap = curr_x1 - prev_x2

            # 计算间隙两侧的类型变化
            prev_type = sorted_dets[i-1].tile_type
            curr_type = sorted_dets[i].tile_type

            # 类型变化的间隙权重更高
            type_change_bonus = 2.0 if prev_type != curr_type else 1.0
            weighted_gap = gap * type_change_bonus

            if weighted_gap > max_gap:
                max_gap = weighted_gap
                split_x = (prev_x2 + curr_x1) // 2

        return split_x

    def _split_by_x(
        self,
        detections: List[TileDetection],
        split_x: Optional[int]
    ) -> Tuple[List[TileDetection], List[TileDetection]]:
        """
        根据X坐标分割手牌和副露

        Args:
            detections: 牌检测列表
            split_x: 分割点X坐标

        Returns:
            (hand_tiles, meld_tiles)
        """
        if split_x is None:
            return detections, []

        hand_tiles: List[TileDetection] = []
        meld_tiles: List[TileDetection] = []

        for det in detections:
            x_center = (det.bbox[0] + det.bbox[2]) // 2
            if x_center < split_x:
                hand_tiles.append(det)
            else:
                meld_tiles.append(det)

        return hand_tiles, meld_tiles

    def _calculate_region(
        self,
        tiles: List[TileDetection],
        image_width: int,
        image_height: int,
        left: bool
    ) -> Tuple[int, int, int, int]:
        """
        根据牌的列表计算区域边界

        Args:
            tiles: 牌检测列表
            image_width: 图像宽度
            image_height: 图像高度
            left: 是否为左侧区域（手牌）

        Returns:
            (x1, y1, x2, y2)
        """
        if not tiles:
            if left:
                return (0, 0, image_width // 2, image_height)
            else:
                return (image_width // 2, 0, image_width, image_height)

        x_coords = [t.bbox[0] for t in tiles] + [t.bbox[2] for t in tiles]
        y_coords = [t.bbox[1] for t in tiles] + [t.bbox[3] for t in tiles]

        x1 = min(x_coords) - 10
        y1 = min(y_coords) - 5
        x2 = max(x_coords) + 10
        y2 = max(y_coords) + 5

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)

        return (x1, y1, x2, y2)

    def visualize_analysis(
        self,
        image: np.ndarray,
        analysis: DynamicUIAnalysis
    ) -> np.ndarray:
        """
        可视化分析结果（调试用）

        Args:
            image: 原始图像
            analysis: 分析结果

        Returns:
            标注后的图像
        """
        vis = image.copy()

        # 绘制分割线
        if analysis.split_x is not None:
            cv2.line(
                vis, (analysis.split_x, 0), (analysis.split_x, vis.shape[0]),
                (0, 255, 255), 2
            )
            cv2.putText(
                vis, "SPLIT", (analysis.split_x + 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

        # 绘制手牌（绿色）
        for tile in analysis.hand_tiles:
            x1, y1, x2, y2 = tile.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis, f"HAND {tile.confidence:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

        # 绘制副露（红色）
        for tile in analysis.meld_tiles:
            x1, y1, x2, y2 = tile.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                vis, f"MELD {tile.confidence:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )

        # 绘制区域
        cv2.rectangle(vis, analysis.hand_region[:2], analysis.hand_region[2:], (0, 255, 0), 1)
        cv2.rectangle(vis, analysis.meld_region[:2], analysis.meld_region[2:], (0, 0, 255), 1)

        return vis
