"""
增强版麻将算牌算法
实现：
1. 严格14张切牌限制
2. 期望收益(EV)模型
3. 速度优先的推倒胡策略
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from .models import (
    NUMBER_SUITS,
    AnalysisResult,
    DiscardOption,
    TableContext,
    Tile,
)
from .parser import count_codes, parse_tile_list, sort_tiles, tiles_to_compact
from .evaluator import (
    _copy_counts,
    _structure_metrics,
    _route_scores,
    _effective_draws,
    _remaining_copies,
)


@dataclass
class EVCalculation:
    """期望收益计算结果"""
    win_probability: float  # 胡牌概率
    expected_value: float  # 期望收益
    speed_score: float  # 速度评分
    safety_score: float  # 安全评分
    improvement_tiles: List[str]  # 改良牌列表


@dataclass
class EnhancedDiscardOption(DiscardOption):
    """增强版切牌选项"""
    ev: Optional[EVCalculation] = None


class EnhancedMahjongEvaluator:
    """
    增强版麻将评估器
    """

    def __init__(
        self,
        speed_priority: float = 0.6,  # 速度优先权重
        safety_priority: float = 0.3,  # 安全权重
        fan_priority: float = 0.1,  # 番型权重
        base_win_points: int = 1,  # 基础胡牌分
        kang_points: int = 2,  # 杠分
        horse_points: int = 1,  # 买马分
    ):
        self.speed_priority = speed_priority
        self.safety_priority = safety_priority
        self.fan_priority = fan_priority
        self.base_win_points = base_win_points
        self.kang_points = kang_points
        self.horse_points = horse_points

    def evaluate(
        self,
        hand: str,
        visible_tiles: str = "",
        exposed_triplets: str = "",
        round_phase: str = "early",
        closed_hand: bool = True,
    ) -> AnalysisResult:
        """
        评估手牌

        严格约束：只有14张牌才进行切牌推荐
        """
        tiles = sort_tiles(parse_tile_list(hand))

        if len(tiles) == 14:
            return self._evaluate_discard(tiles, visible_tiles, exposed_triplets, round_phase, closed_hand)
        elif len(tiles) == 13:
            return self._evaluate_pre_draw(tiles, visible_tiles, exposed_triplets, round_phase, closed_hand)
        else:
            raise ValueError(f"手牌数量必须为13或14张，当前: {len(tiles)}张")

    def _evaluate_discard(
        self,
        tiles: List[Tile],
        visible_tiles: str,
        exposed_triplets: str,
        round_phase: str,
        closed_hand: bool,
    ) -> AnalysisResult:
        """
        14张手牌时评估切牌
        """
        visible_list = parse_tile_list(visible_tiles)
        exposed_list = parse_tile_list(exposed_triplets)
        original_counts = _copy_counts(tiles)
        
        context = TableContext(
            visible_counts=count_codes(visible_list),
            exposed_triplets={tile.code for tile in exposed_list},
            round_phase=round_phase,
            closed_hand=closed_hand,
        )

        # 评估每个可能的切牌
        options: List[EnhancedDiscardOption] = []
        grouped_tiles: List[Tuple[str, List[Tile]]] = []
        
        from itertools import groupby
        for _, group in groupby(sort_tiles(tiles), key=lambda tile: tile.code):
            group_list = list(group)
            grouped_tiles.append((group_list[0].code, group_list))

        for discard_code, group in grouped_tiles:
            remaining_tiles = tiles.copy()
            remaining_tiles.remove(group[0])
            
            # 计算EV
            ev = self._calculate_ev(remaining_tiles, context, original_counts)
            
            # 计算综合分数
            base_score = self._calculate_speed_score(remaining_tiles, context)
            safety_score = self._calculate_safety_score(discard_code, context, original_counts, round_phase)
            fan_score = self._calculate_fan_score(remaining_tiles, context)
            
            total_score = (
                base_score * self.speed_priority +
                safety_score * self.safety_priority +
                fan_score * self.fan_priority
            )
            
            effective_draws, improving_tiles = _effective_draws(remaining_tiles, context)
            route_scores = _route_scores(remaining_tiles, context)
            
            option = EnhancedDiscardOption(
                discard=discard_code,
                total_score=round(total_score, 2),
                base_score=round(base_score, 2),
                effective_draws=effective_draws,
                improving_tiles=improving_tiles[:12],
                route_scores={
                    k: round(v, 2) for k, v in sorted(route_scores.items(), key=lambda x: -x[1])[:6]
                },
                summary=self._make_summary(remaining_tiles, context, route_scores),
                ev=ev
            )
            options.append(option)

        # 按总分数排序
        options.sort(key=lambda o: o.total_score, reverse=True)

        return AnalysisResult(
            hand=tiles_to_compact(tiles),
            tile_count=14,
            options=options,  # type: ignore
            visible_counts=context.visible_counts,
            exposed_triplets=context.exposed_triplets,
        )

    def _evaluate_pre_draw(
        self,
        tiles: List[Tile],
        visible_tiles: str,
        exposed_triplets: str,
        round_phase: str,
        closed_hand: bool,
    ) -> AnalysisResult:
        """
        13张手牌时评估（摸牌前分析
        """
        visible_list = parse_tile_list(visible_tiles)
        exposed_list = parse_tile_list(exposed_triplets)
        
        hand_counts = _copy_counts(tiles)
        context = TableContext(
            visible_counts=count_codes(visible_list),
            exposed_triplets={tile.code for tile in exposed_list},
            round_phase=round_phase,
            closed_hand=closed_hand,
        )

        draw_results: List[Dict] = []
        for code in ALL_TILE_CODES:
            remaining = _remaining_copies(code, hand_counts, context.visible_counts)
            if remaining <= 0:
                continue

            drawn_tile = Tile(rank=int(code[0]), suit=code[1])
            hand_14 = sort_tiles(tiles + [drawn_tile])

            # 分析14张手牌的切牌
            result_14 = self._evaluate_discard(hand_14, visible_tiles, exposed_triplets, round_phase, closed_hand)
            best_option = result_14.options[0] if result_14.options else None

            draw_results.append({
                "draw": code,
                "is_win": False,  # 暂时简化
                "was_tenpai": False,  # 暂时简化
                "tenpai_waits": [],
                "best_discard": best_option.discard if best_option else "",
                "score": best_option.total_score if best_option else 0,
                "remaining": remaining,
                "effective_draws": best_option.effective_draws if best_option else 0,
            })

        # 排序：高分优先
        draw_results.sort(
            key=lambda d: d["score"],
            reverse=True
        )

        return AnalysisResult(
            hand=tiles_to_compact(tiles),
            tile_count=13,
            options=[],
            visible_counts=context.visible_counts,
            exposed_triplets=context.exposed_triplets,
            pre_draw=draw_results,
        )

    def _calculate_ev(
        self,
        tiles: List[Tile],
        context: TableContext,
        original_counts: Counter[str]
    ) -> EVCalculation:
        """
        计算期望收益
        """
        hand_counts = _copy_counts(tiles)
        
        # 计算有效进张数
        effective_draws, improving_tiles = _effective_draws(tiles, context)
        
        # 计算速度评分
        speed_score = self._calculate_speed_score(tiles, context)
        
        # 计算安全评分
        safety_score = 1.0  # 占位，这里不计算具体某张的安全
        
        # 计算期望收益
        expected_value = (
            effective_draws * 0.1 +
            speed_score * 0.5
        )

        return EVCalculation(
            win_probability=0.0,  # 暂时简化
            expected_value=expected_value,
            speed_score=speed_score,
            safety_score=safety_score,
            improvement_tiles=improving_tiles
        )

    def _calculate_speed_score(
        self,
        tiles: List[Tile],
        context: TableContext
    ) -> float:
        """
        计算速度评分（基于结构指标
        """
        metrics = _structure_metrics(tiles)
        effective_draws, _ = _effective_draws(tiles, context)
        
        # 使用结构指标估算速度
        base_score = (
            metrics["sequence_count"] * 10
            + metrics["triplet_count"] * 8
            + metrics["pair_count"] * 5
            + metrics["partial_sequence_count"] * 3
            - metrics["isolated_count"] * 2
        )
        
        # 有效进张越多越好
        draw_bonus = effective_draws * 0.5
        
        return base_score + draw_bonus

    def _calculate_safety_score(
        self,
        discard_code: str,
        context: TableContext,
        original_counts: Counter[str],
        round_phase: str
    ) -> float:
        """
        计算安全评分
        """
        visible = context.visible_counts.get(discard_code, 0)
        
        # 直接安全：已出多张
        if visible >= 3:
            return 10.0
        if visible >= 2:
            return 7.0
        if visible >= 1:
            return 4.0
        
        # 字牌相对安全
        if discard_code[1] == "z":
            return 3.0
        
        # 晚期更注重安全
        phase_multiplier = {"early": 1.0, "mid": 1.5, "late": 2.0}.get(round_phase, 1.0)
        
        return max(0, 5.0 - phase_multiplier * 2.0)

    def _calculate_fan_score(
        self,
        tiles: List[Tile],
        context: TableContext
    ) -> float:
        """
        计算番型评分（推倒胡中番型权重较低
        """
        route_scores = _route_scores(tiles, context)
        
        # 推倒胡不追大番，只给基础分
        max_route = max(route_scores.values()) if route_scores else 0
        
        return max_route * 0.3  # 番型权重降低

    def _make_summary(
        self,
        tiles: List[Tile],
        context: TableContext,
        route_scores: Dict[str, float]
    ) -> List[str]:
        """生成摘要"""
        metrics = _structure_metrics(tiles)
        top_routes = [
            k for k, v in sorted(route_scores.items(), key=lambda x: -x[1])
            if v > 0
        ][:3]

        summary = [
            f"对子{metrics['pair_count']}刻子{metrics['triplet_count']}顺子{metrics['sequence_count']}",
            f"搭子{metrics['partial_sequence_count']}孤张{metrics['isolated_count']}",
        ]
        if top_routes:
            summary.append(f"趋势: {'/'.join(top_routes)}")
        return summary


# 全局实例
DEFAULT_EVALUATOR = EnhancedMahjongEvaluator()


def evaluate_hand_enhanced(
    hand: str,
    visible_tiles: str = "",
    exposed_triplets: str = "",
    round_phase: str = "early",
    closed_hand: bool = True,
) -> AnalysisResult:
    """
    增强版手牌评估函数
    """
    return DEFAULT_EVALUATOR.evaluate(
        hand, visible_tiles, exposed_triplets, round_phase, closed_hand
    )


# 重新导出ALL_TILE_CODES
ALL_TILE_CODES = [
    *(f"{rank}{suit}" for suit in NUMBER_SUITS for rank in range(1, 10)),
    *(f"{rank}z" for rank in range(1, 8)),
]
