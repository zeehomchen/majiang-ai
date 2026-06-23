from __future__ import annotations

from collections import Counter
from itertools import groupby

from .models import (
    DRAGON_CODES,
    NUMBER_SUITS,
    ORPHAN_CODES,
    WIND_CODES,
    AnalysisResult,
    DiscardOption,
    TableContext,
    Tile,
)
from .parser import count_codes, parse_tile_list, sort_tiles, tiles_to_compact


ALL_TILE_CODES = [
    *(f"{rank}{suit}" for suit in NUMBER_SUITS for rank in range(1, 10)),
    *(f"{rank}z" for rank in range(1, 8)),
]


def _copy_counts(tiles: list[Tile]) -> Counter[str]:
    return count_codes(tiles)


def _sequence_count(counts: Counter[str]) -> int:
    sequences = 0
    for suit in NUMBER_SUITS:
        ranks = Counter({rank: counts[f"{rank}{suit}"] for rank in range(1, 10)})
        for start in range(1, 8):
            while ranks[start] and ranks[start + 1] and ranks[start + 2]:
                ranks[start] -= 1
                ranks[start + 1] -= 1
                ranks[start + 2] -= 1
                sequences += 1
    return sequences


def _partial_sequence_count(counts: Counter[str]) -> int:
    partials = 0
    for suit in NUMBER_SUITS:
        ranks = Counter({rank: counts[f"{rank}{suit}"] for rank in range(1, 10)})
        for start in range(1, 9):
            while ranks[start] and ranks[start + 1]:
                ranks[start] -= 1
                ranks[start + 1] -= 1
                partials += 1
        for start in range(1, 8):
            while ranks[start] and ranks[start + 2]:
                ranks[start] -= 1
                ranks[start + 2] -= 1
                partials += 1
    return partials


def _isolated_tile_count(counts: Counter[str]) -> int:
    isolated = 0
    for code, amount in counts.items():
        rank = int(code[0])
        suit = code[1]
        if amount >= 2:
            continue
        if suit == "z":
            isolated += amount
            continue
        left_1 = counts[f"{rank - 1}{suit}"] if rank - 1 >= 1 else 0
        left_2 = counts[f"{rank - 2}{suit}"] if rank - 2 >= 1 else 0
        right_1 = counts[f"{rank + 1}{suit}"] if rank + 1 <= 9 else 0
        right_2 = counts[f"{rank + 2}{suit}"] if rank + 2 <= 9 else 0
        if left_1 or left_2 or right_1 or right_2:
            continue
        isolated += amount
    return isolated


def _structure_metrics(tiles: list[Tile]) -> dict[str, int]:
    counts = _copy_counts(tiles)
    number_suits = {tile.suit for tile in tiles if tile.suit in NUMBER_SUITS}
    honor_count = sum(1 for tile in tiles if tile.suit == "z")
    terminal_or_honor_count = sum(1 for tile in tiles if tile.is_terminal_or_honor)
    return {
        "pair_count": sum(1 for amount in counts.values() if amount >= 2),
        "triplet_count": sum(1 for amount in counts.values() if amount >= 3),
        "quad_count": sum(1 for amount in counts.values() if amount == 4),
        "sequence_count": _sequence_count(counts),
        "partial_sequence_count": _partial_sequence_count(counts),
        "isolated_count": _isolated_tile_count(counts),
        "honor_count": honor_count,
        "number_suit_count": len(number_suits),
        "terminal_or_honor_count": terminal_or_honor_count,
    }


def _route_scores(tiles: list[Tile], context: TableContext) -> dict[str, float]:
    counts = _copy_counts(tiles)
    metrics = _structure_metrics(tiles)
    route_scores: dict[str, float] = {}

    pengpeng = (
        metrics["triplet_count"] * 24
        + metrics["pair_count"] * 12
        + metrics["quad_count"] * 10
        - metrics["sequence_count"] * 9
    )
    route_scores["碰碰胡"] = pengpeng

    number_suits = {tile.suit for tile in tiles if tile.suit in NUMBER_SUITS}
    honor_count = metrics["honor_count"]
    if len(number_suits) == 1 and honor_count == 0:
        route_scores["清一色"] = 90 + len(tiles) * 2
    elif len(number_suits) == 1 and honor_count > 0:
        route_scores["混一色"] = 70 + (len(tiles) - honor_count) * 2 + honor_count
    else:
        suit_load = Counter(tile.suit for tile in tiles if tile.suit in NUMBER_SUITS)
        if suit_load:
            major_suit, major_count = suit_load.most_common(1)[0]
            other_count = sum(suit_load.values()) - major_count
            if major_count >= 5:
                route_scores["清一色趋势"] = major_count * 8 - other_count * 6
            if major_count >= 4 and honor_count:
                route_scores["混一色趋势"] = major_count * 7 + honor_count * 4 - other_count * 5

    if context.closed_hand:
        seven_pairs = metrics["pair_count"] * 18 - metrics["triplet_count"] * 6
        route_scores["七小对"] = seven_pairs
        if any(amount == 4 for amount in counts.values()):
            route_scores["豪华七对"] = seven_pairs + 26

    orphan_unique = sum(1 for code in ORPHAN_CODES if counts[code] > 0)
    orphan_pair = sum(1 for code in ORPHAN_CODES if counts[code] >= 2)
    non_orphan_tiles = sum(amount for code, amount in counts.items() if code not in ORPHAN_CODES)
    route_scores["十三幺"] = orphan_unique * 12 + orphan_pair * 18 - non_orphan_tiles * 8

    dragon_triplets = sum(1 for code in DRAGON_CODES if counts[code] >= 3)
    dragon_pairs = sum(1 for code in DRAGON_CODES if counts[code] >= 2)
    if dragon_triplets == 3:
        route_scores["大三元"] = 135
    elif dragon_triplets >= 2 and dragon_pairs >= 3:
        route_scores["小三元"] = 105
    else:
        route_scores["三元牌趋势"] = dragon_triplets * 24 + dragon_pairs * 10

    wind_triplets = sum(1 for code in WIND_CODES if counts[code] >= 3)
    wind_pairs = sum(1 for code in WIND_CODES if counts[code] >= 2)
    if wind_triplets == 4:
        route_scores["大四喜"] = 150
    elif wind_triplets >= 3 and wind_pairs >= 4:
        route_scores["小四喜"] = 115
    else:
        route_scores["风牌趋势"] = wind_triplets * 20 + wind_pairs * 8

    if tiles and all(tile.suit == "z" for tile in tiles):
        route_scores["字一色"] = 150

    if tiles and all(tile.is_terminal_or_honor for tile in tiles):
        if all(tile.is_terminal for tile in tiles):
            route_scores["清幺九"] = 115
        else:
            route_scores["混幺九"] = 100

    exposed_pressure = 0
    for code in context.exposed_triplets:
        if counts[code] > 0:
            exposed_pressure += 12 + counts[code] * 5
    if exposed_pressure:
        route_scores["抢杠胡机会"] = exposed_pressure

    if metrics["quad_count"] or metrics["triplet_count"] >= 2:
        route_scores["杠上开花潜力"] = metrics["quad_count"] * 26 + metrics["triplet_count"] * 6

    return route_scores


def _phase_multiplier(round_phase: str) -> tuple[float, float]:
    if round_phase == "late":
        return (0.8, 1.35)
    if round_phase == "mid":
        return (1.0, 1.1)
    return (1.2, 0.95)


def _base_score(tiles: list[Tile], context: TableContext) -> float:
    metrics = _structure_metrics(tiles)
    route_scores = _route_scores(tiles, context)
    early_factor, risk_factor = _phase_multiplier(context.round_phase)
    score = 0.0
    score += metrics["sequence_count"] * 18
    score += metrics["triplet_count"] * 20
    score += metrics["pair_count"] * 11
    score += metrics["partial_sequence_count"] * 6
    score -= metrics["isolated_count"] * 5 * risk_factor

    best_routes = sorted(route_scores.values(), reverse=True)[:2]
    if best_routes:
        score += best_routes[0] * 0.55
    if len(best_routes) > 1:
        score += best_routes[1] * 0.25

    if metrics["pair_count"] >= 3 and metrics["triplet_count"] >= 1:
        score += metrics["pair_count"] * 18 + metrics["triplet_count"] * 14
        score -= metrics["sequence_count"] * 14

    if context.closed_hand:
        score += 4
    score *= early_factor
    return score


def _remaining_copies(code: str, hand_counts: Counter[str], visible_counts: Counter[str]) -> int:
    return max(0, 4 - hand_counts[code] - visible_counts[code])


def _effective_draws(tiles: list[Tile], context: TableContext) -> tuple[int, list[str]]:
    baseline = _base_score(tiles, context)
    hand_counts = _copy_counts(tiles)
    improving_tiles: list[str] = []
    total_remaining = 0
    for code in ALL_TILE_CODES:
        remaining = _remaining_copies(code, hand_counts, context.visible_counts)
        if remaining <= 0:
            continue
        candidate_tiles = tiles + [Tile(rank=int(code[0]), suit=code[1])]
        candidate_score = _base_score(candidate_tiles, context)
        if candidate_score > baseline + 6:
            improving_tiles.append(code)
            total_remaining += remaining
    return total_remaining, improving_tiles


def _option_summary(tiles: list[Tile], context: TableContext, route_scores: dict[str, float]) -> list[str]:
    metrics = _structure_metrics(tiles)
    top_routes = [
        route
        for route, value in sorted(route_scores.items(), key=lambda item: item[1], reverse=True)
        if value > 0
    ][:3]
    summary = [
        f"对子 {metrics['pair_count']} 组 / 刻子 {metrics['triplet_count']} 组 / 顺子 {metrics['sequence_count']} 组",
        f"搭子 {metrics['partial_sequence_count']} 组 / 孤张 {metrics['isolated_count']} 张",
    ]
    if top_routes:
        summary.append("高权重路线: " + "、".join(top_routes))
    return summary


def evaluate_hand(
    hand: str,
    visible_tiles: str = "",
    exposed_triplets: str = "",
    round_phase: str = "early",
    closed_hand: bool = True,
) -> AnalysisResult:
    tiles = sort_tiles(parse_tile_list(hand))
    if len(tiles) < 2:
        raise ValueError("Hand must contain at least 2 tiles")

    visible_list = parse_tile_list(visible_tiles)
    exposed_list = parse_tile_list(exposed_triplets)
    original_counts = _copy_counts(tiles)
    context = TableContext(
        visible_counts=count_codes(visible_list),
        exposed_triplets={tile.code for tile in exposed_list},
        round_phase=round_phase,
        closed_hand=closed_hand,
    )
    original_suit_load = Counter(tile.suit for tile in tiles if tile.suit in NUMBER_SUITS)
    major_suit = ""
    major_suit_count = 0
    if original_suit_load:
        major_suit, major_suit_count = original_suit_load.most_common(1)[0]

    grouped_tiles: list[tuple[str, list[Tile]]] = []
    for _, group in groupby(sort_tiles(tiles), key=lambda tile: tile.code):
        group_list = list(group)
        grouped_tiles.append((group_list[0].code, group_list))

    options: list[DiscardOption] = []
    for discard_code, grouped in grouped_tiles:
        remaining_tiles = tiles.copy()
        remaining_tiles.remove(grouped[0])
        base_score = _base_score(remaining_tiles, context)
        route_scores = _route_scores(remaining_tiles, context)
        effective_draws, improving_tiles = _effective_draws(remaining_tiles, context)
        total_score = base_score + effective_draws * 2.5
        discard_tile = grouped[0]
        if major_suit_count >= 6 and discard_tile.suit in NUMBER_SUITS:
            if discard_tile.suit == major_suit:
                total_score -= 22
            else:
                total_score += 14
        if original_counts[discard_code] == 1 and discard_tile.suit in NUMBER_SUITS:
            total_score += discard_tile.rank * 0.01
        options.append(
            DiscardOption(
                discard=discard_code,
                total_score=round(total_score, 2),
                base_score=round(base_score, 2),
                effective_draws=effective_draws,
                improving_tiles=improving_tiles[:12],
                route_scores={
                    route: round(value, 2)
                    for route, value in sorted(
                        route_scores.items(), key=lambda item: item[1], reverse=True
                    )[:6]
                },
                summary=_option_summary(remaining_tiles, context, route_scores),
            )
        )

    options.sort(key=lambda option: option.total_score, reverse=True)
    return AnalysisResult(
        hand=tiles_to_compact(tiles),
        options=options,
        visible_counts=context.visible_counts,
        exposed_triplets=context.exposed_triplets,
    )


def result_to_dict(result: AnalysisResult) -> dict:
    return {
        "hand": result.hand,
        "visible_counts": dict(result.visible_counts),
        "exposed_triplets": sorted(result.exposed_triplets),
        "options": [
            {
                "discard": option.discard,
                "total_score": option.total_score,
                "base_score": option.base_score,
                "effective_draws": option.effective_draws,
                "improving_tiles": option.improving_tiles,
                "route_scores": option.route_scores,
                "summary": option.summary,
            }
            for option in result.options
        ],
    }
