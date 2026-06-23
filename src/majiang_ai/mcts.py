"""MCTS 搜索引擎。

在扑克/麻将类不完全信息游戏中，MCTS 的标准做法是"根节点并行采样"：
- 对每个可出牌选项，并行运行 N 次对局模拟
- 在模拟中用启发式策略引导自己的出牌 (而非纯随机)
- 取平均回报最高的选项

性能目标：每秒钟能跑 500-1000 次高质量模拟。
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import simulator as sim

# ---------------------------------------------------------------------------
# 节点定义
# ---------------------------------------------------------------------------


@dataclass
class MCTSNode:
    state: sim.GameState
    parent: Optional["MCTSNode"] = None
    action: Optional[str] = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    untried_actions: list[str] = field(default_factory=list)

    @property
    def value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb1(self, parent_visits: int, exploration: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        return self.value + exploration * math.sqrt(math.log(parent_visits) / self.visits)


# ---------------------------------------------------------------------------
# 结果类型
# ---------------------------------------------------------------------------


@dataclass
class MCTSResult:
    best_discard: str
    best_score: float
    options: list[dict]
    simulations: int
    elapsed_ms: float
    mode: str


# ---------------------------------------------------------------------------
# 启发式出牌 (用于模拟中引导决策，来自 evaluator 的简化版)
# ---------------------------------------------------------------------------

def _tile_value_in_context(code: str, hand: list[str]) -> float:
    """评估单张牌在手牌中的保留价值。值越高 = 越不应打出去。"""
    counts = Counter(hand)
    cnt = counts[code]
    rank, suit = sim._parse_code(code)

    # 刻子/杠: 极高保留价值
    if cnt >= 3:
        return 180.0 + cnt * 15
    # 对子: 高保留价值 (能碰)
    if cnt == 2:
        return 70.0

    v = cnt * 20.0

    # 数牌: 看周围搭子
    if sim._is_number(code):
        for offset in (-2, -1, 1, 2):
            neighbor = f"{rank + offset}{suit}"
            v += counts.get(neighbor, 0) * 7

    # 字牌: 孤张不太值钱
    if suit == "z":
        v -= 10

    # 边张 (1/9) 稍扣分
    if suit != "z" and rank in (1, 9):
        v -= 3

    return v


def _smart_discard(hand: list[str]) -> str:
    """智能出牌: 用启发出牌策略。"""
    if not hand:
        return ""
    candidates = list(set(hand))
    candidates.sort(key=lambda c: _tile_value_in_context(c, hand))
    return candidates[0]


def _our_smart_policy(state: sim.GameState) -> str:
    """我们自己的智能出牌策略 (模拟中用)。"""
    hand = state.players[0].hand
    if not hand:
        return ""

    # 先检查是否听牌
    if sim.is_tenpai(hand):
        # 听牌了就不换了，出最后摸到的牌
        return hand[-1]

    return _smart_discard(hand)


# ---------------------------------------------------------------------------
# MCTS 搜索
# ---------------------------------------------------------------------------


def search(
    hand: list[str],
    visible_codes: Optional[list[str]] = None,
    exposed_triplets: Optional[set[str]] = None,
    num_simulations: int = 5000,
    mode: str = "flat",
    time_limit_ms: float = 8000,
    exploration_const: float = 1.4,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> MCTSResult:
    """MCTS 搜索最优出牌。

    Args:
        hand: 当前手牌编码列表
        visible_codes: 已知可见牌
        exposed_triplets: 已知他家碰杠的牌
        num_simulations: 总模拟次数
        mode: "flat" (根并行) 或 "tree" (UCB1 树)
        time_limit_ms: 时间上限
        exploration_const: UCB1 探索参数
    """
    visible = list(visible_codes or [])
    discard_options = sim.get_possible_discards(hand)

    if not discard_options:
        raise ValueError("手牌为空，无法决策")

    start_time = time.time()

    if mode == "flat":
        return _flat_mc_search(hand, visible, discard_options, num_simulations,
                               start_time, time_limit_ms, progress_callback)
    else:
        return _tree_mcts_search(hand, visible, discard_options, num_simulations,
                                 start_time, time_limit_ms, exploration_const, progress_callback)


def _flat_mc_search(
    hand: list[str],
    visible: list[str],
    options: list[str],
    num_simulations: int,
    start_time: float,
    time_limit: float,
    progress_cb=None,
) -> MCTSResult:
    """扁平蒙特卡洛: 每个出牌选项均分模拟次数。"""
    stats: dict[str, dict] = defaultdict(lambda: {"total": 0.0, "wins": 0, "count": 0})
    sims_per_option = max(1, num_simulations // len(options))
    total_simulations = 0

    for opt in options:
        remaining = [c for c in hand if c != opt]
        # 如果手中有多张相同的，只移除一张
        removed = False
        remaining = []
        for c in hand:
            if c == opt and not removed:
                removed = True
                continue
            remaining.append(c)

        for _ in range(sims_per_option):
            elapsed = (time.time() - start_time) * 1000
            if elapsed > time_limit:
                break

            extra_visible = list(visible) + [opt]
            state = sim.new_game(remaining, extra_visible)
            policy = lambda s: _our_smart_policy(s)
            outcome, winner = sim.run_rollout(state, policy, max_turns=60)

            score = sim.evaluate_outcome(outcome, state, remaining, 0)
            stats[opt]["total"] += score
            stats[opt]["count"] += 1
            if outcome == 0:
                stats[opt]["wins"] += 1
            total_simulations += 1

        if progress_cb:
            progress_cb(total_simulations, len(options) * sims_per_option)

        if (time.time() - start_time) * 1000 > time_limit:
            break

    result_options = []
    for opt in options:
        s = stats[opt]
        cnt = s["count"]
        avg = s["total"] / cnt if cnt > 0 else -999.0
        win_rate = s["wins"] / cnt * 100 if cnt > 0 else 0
        result_options.append({
            "discard": opt,
            "score": round(avg, 2),
            "win_rate": round(win_rate, 1),
            "simulations": cnt,
        })

    result_options.sort(key=lambda x: x["score"], reverse=True)
    elapsed_ms = (time.time() - start_time) * 1000

    return MCTSResult(
        best_discard=result_options[0]["discard"] if result_options else "",
        best_score=result_options[0]["score"] if result_options else 0,
        options=result_options,
        simulations=total_simulations,
        elapsed_ms=round(elapsed_ms, 1),
        mode="flat",
    )


def _tree_mcts_search(
    hand: list[str],
    visible: list[str],
    options: list[str],
    num_simulations: int,
    start_time: float,
    time_limit: float,
    exploration_const: float,
    progress_cb=None,
) -> MCTSResult:
    """UCB1 树搜索模式。"""
    root_state = sim.new_game(hand, visible)
    root = MCTSNode(state=root_state, untried_actions=list(options))

    for sim_i in range(num_simulations):
        elapsed = (time.time() - start_time) * 1000
        if elapsed > time_limit:
            break

        node = _select(root, exploration_const)
        if node.untried_actions and not node.state.is_finished:
            node = _expand(node)

        outcome, state = _simulate_from(node)
        value = sim.evaluate_outcome(outcome, state, hand, 0)
        _backpropagate(node, value)

        if progress_cb and sim_i % 500 == 0:
            progress_cb(sim_i, num_simulations)

    elapsed_ms = (time.time() - start_time) * 1000
    result_options = []

    for child in sorted(root.children, key=lambda c: c.value, reverse=True):
        result_options.append({
            "discard": child.action or "?",
            "score": round(child.value, 2),
            "win_rate": 0,
            "simulations": child.visits,
        })

    return MCTSResult(
        best_discard=result_options[0]["discard"] if result_options else "",
        best_score=result_options[0]["score"] if result_options else 0,
        options=result_options,
        simulations=sum(c.visits for c in root.children),
        elapsed_ms=round(elapsed_ms, 1),
        mode="tree",
    )


def _select(node: MCTSNode, c: float) -> MCTSNode:
    current = node
    while not current.state.is_finished and not current.untried_actions and current.children:
        best = max(current.children, key=lambda ch: ch.ucb1(current.visits, c), default=None)
        if best is None:
            break
        current = best
    return current


def _expand(node: MCTSNode) -> MCTSNode:
    action = node.untried_actions.pop()
    new_state = _clone_state(node.state)
    remaining = []
    removed = False
    for c in new_state.players[0].hand:
        if c == action and not removed:
            removed = True
            continue
        remaining.append(c)
    new_state.players[0].hand = remaining
    new_state.players[0].discards.append(action)
    new_state.last_discard = action
    new_state.last_discard_player = 0

    child = MCTSNode(state=new_state, parent=node, action=action, untried_actions=[])
    node.children.append(child)
    return child


def _simulate_from(node: MCTSNode) -> tuple[int, sim.GameState]:
    state = _clone_state(node.state)
    policy = lambda s: _our_smart_policy(s)
    outcome, winner = sim.run_rollout(state, policy, max_turns=60)
    return (outcome, state)


def _backpropagate(node: MCTSNode, value: float) -> None:
    current: Optional[MCTSNode] = node
    while current is not None:
        current.visits += 1
        current.total_value += value
        current = current.parent


def _clone_state(state: sim.GameState) -> sim.GameState:
    players = [
        sim.Player(
            hand=list(p.hand),
            melds=list(p.melds),
            discards=list(p.discards),
            in_riichi=p.in_riichi,
        )
        for p in state.players
    ]
    return sim.GameState(
        wall=list(state.wall),
        players=players,
        current_player=state.current_player,
        phase=state.phase,
        last_discard=state.last_discard,
        last_discard_player=state.last_discard_player,
        draw_pile_pos=state.draw_pile_pos,
        turn_count=state.turn_count,
        max_turns=state.max_turns,
    )


# ---------------------------------------------------------------------------
# 便捷封装
# ---------------------------------------------------------------------------

def mcts_discard(
    hand_compact: str,
    visible_compact: str = "",
    simulations: int = 5000,
    mode: str = "flat",
    time_limit_ms: float = 8000,
    verbose: bool = False,
) -> MCTSResult:
    from .parser import parse_tile_list

    hand_tiles = parse_tile_list(hand_compact)
    hand_codes = [t.code for t in hand_tiles]
    visible_codes = [t.code for t in parse_tile_list(visible_compact)] if visible_compact else []

    if verbose:
        print(f"MCTS 搜索: 手牌={hand_compact}, 模拟次数={simulations}, 模式={mode}")
        t0 = time.time()

    result = search(
        hand=hand_codes,
        visible_codes=visible_codes,
        num_simulations=simulations,
        mode=mode,
        time_limit_ms=time_limit_ms,
    )

    if verbose:
        print(f"  完成: {result.simulations} 次模拟, {result.elapsed_ms:.0f}ms")
        print(f"  最优切牌: {result.best_discard} (评分 {result.best_score:.1f})")
        for i, opt in enumerate(result.options[:5]):
            print(f"    {i+1}. 打 {opt['discard']}: 评分 {opt['score']:.1f}, "
                  f"胜率 {opt.get('win_rate', 0):.1f}%, 模拟 {opt['simulations']} 次")

    return result
