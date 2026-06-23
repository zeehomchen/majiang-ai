"""对局模拟器。

模拟广东推倒胡 4 人对局：牌墙、抽牌、碰/杠、和牌判定。
为 MCTS 搜索提供高速 rollout 环境。

规则要点：
- 不能吃牌 (chi)
- 可以碰 (peng)、明杠/暗杠/补杠 (kong)
- 自摸和点炮均可和牌
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---- 牌集 ----
SUITS = ("m", "p", "s", "z")
NUMBER_SUITS = ("m", "p", "s")
DRAGON_CODES = ("5z", "6z", "7z")
WIND_CODES = ("1z", "2z", "3z", "4z")

# 136 张牌 (万/筒/条各 36 + 字牌 28)
ALL_TILES_136: list[str] = []
for s in NUMBER_SUITS:
    for r in range(1, 10):
        ALL_TILES_136.extend([f"{r}{s}"] * 4)
for r in range(1, 8):
    ALL_TILES_136.extend([f"{r}z"] * 4)
assert len(ALL_TILES_136) == 136


def _parse_code(code: str) -> tuple[int, str]:
    return int(code[0]), code[1]


def _is_number(code: str) -> bool:
    return code[1] in NUMBER_SUITS


# ---- 手牌分析 ----
def _counts(hand: list[str]) -> Counter[str]:
    return Counter(hand)


def _has_pair(counts: Counter[str]) -> bool:
    return any(v >= 2 for v in counts.values())


def _has_triplet(counts: Counter[str], code: str) -> bool:
    return counts[code] >= 3


def _can_kong(counts: Counter[str], code: str) -> bool:
    return counts[code] == 4


def _can_peng(counts: Counter[str], code: str) -> bool:
    return counts[code] >= 2


# ---- 和牌判定 (推倒胡简化) ----
def is_winning_hand(hand: list[str]) -> bool:
    """判定手牌是否和牌 (14 张: 4 面子 + 1 雀头 或 七对子)。

    简化版：因为不能吃牌，面子里只含刻子/杠子+雀头，
    或者碰碰胡、七对子、十三幺均可。
    """
    n = len(hand)
    if n % 3 != 2:
        return False

    counts = _counts(hand)

    # 七对子
    if n == 14 and sum(1 for v in counts.values() if v >= 2) == 7:
        return True

    # 标准判定：递归消除雀头 + 刻子
    def _can_form_melds(remaining: Counter[str]) -> bool:
        total = sum(remaining.values())
        if total == 0:
            return True
        sorted_codes = sorted(k for k, v in remaining.items() if v > 0)
        for code in sorted_codes:
            cnt = remaining[code]
            if cnt >= 3:
                # 尝试移除一个刻子
                remaining[code] -= 3
                if _can_form_melds(remaining):
                    remaining[code] += 3
                    return True
                remaining[code] += 3
            if cnt >= 1 and _is_number(code):
                rank, suit = _parse_code(code)
                c2 = f"{rank+1}{suit}"
                c3 = f"{rank+2}{suit}"
                if rank <= 7 and remaining[c2] >= 1 and remaining[c3] >= 1:
                    remaining[code] -= 1
                    remaining[c2] -= 1
                    remaining[c3] -= 1
                    if _can_form_melds(remaining):
                        remaining[code] += 1
                        remaining[c2] += 1
                        remaining[c3] += 1
                        return True
                    remaining[code] += 1
                    remaining[c2] += 1
                    remaining[c3] += 1
            return False
        return False

    # 枚举雀头
    for code, cnt in counts.items():
        if cnt >= 2:
            c = counts.copy()
            c[code] -= 2
            if _can_form_melds(c):
                return True

    return False


def is_tenpai(hand: list[str]) -> bool:
    """判定是否听牌 (13 张, 差 1 张即和)。"""
    if len(hand) % 3 != 1:
        return False
    counts = _counts(hand)
    for code in _unique_tile_codes():
        if counts[code] >= 4:
            continue
        test = hand + [code]
        if is_winning_hand(test):
            return True
    return False


def tenpai_waiting_tiles(hand: list[str]) -> list[str]:
    """返回听牌时等待的牌列表。"""
    if len(hand) % 3 != 1:
        return []
    counts = _counts(hand)
    waits = []
    for code in _unique_tile_codes():
        if counts[code] >= 4:
            continue
        if is_winning_hand(hand + [code]):
            waits.append(code)
    return waits


def _unique_tile_codes() -> list[str]:
    codes = []
    for s in NUMBER_SUITS:
        codes.extend(f"{r}{s}" for r in range(1, 10))
    codes.extend(f"{r}z" for r in range(1, 8))
    return codes


# ---- 牌效简算 ----
def _count_xiangting(hand: list[str]) -> int:
    """快速估算向听数。"""
    if is_winning_hand(hand):
        return 0
    if len(hand) % 3 == 1 and is_tenpai(hand):
        return 1
    # 粗糙估算
    counts = _counts(hand)
    pairs = sum(1 for v in counts.values() if v >= 2)
    triplets = sum(1 for v in counts.values() if v >= 3)
    partial_seqs = 0
    for s in NUMBER_SUITS:
        ranks = {r: counts.get(f"{r}{s}", 0) for r in range(1, 10)}
        for start in range(1, 9):
            if ranks[start] and ranks[start + 1]:
                partial_seqs += 1
    total = len(hand) // 3
    have = triplets + pairs + partial_seqs
    return max(0, total - have + 1)


# ---- 对局状态 ----
class Phase(Enum):
    DRAW = "draw"
    DISCARD = "discard"
    PENG_CHECK = "peng_check"
    KONG_CHECK = "kong_check"
    WIN_CHECK = "win_check"
    FINISHED = "finished"


@dataclass
class Player:
    hand: list[str] = field(default_factory=list)
    melds: list[tuple[str, ...]] = field(default_factory=list)  # 副露 (碰/杠)
    discards: list[str] = field(default_factory=list)  # 已打出的牌
    in_riichi: bool = False

    @property
    def effective_hand(self) -> list[str]:
        h = list(self.hand)
        for m in self.melds:
            h.extend(m)
        return h


@dataclass
class GameState:
    wall: list[str] = field(default_factory=list)
    players: list[Player] = field(default_factory=list)
    current_player: int = 0
    phase: Phase = Phase.DRAW
    last_discard: str = ""
    last_discard_player: int = -1
    draw_pile_pos: int = 0  # 牌墙当前位置
    turn_count: int = 0
    max_turns: int = 70

    @property
    def is_finished(self) -> bool:
        return self.phase == Phase.FINISHED

    @property
    def draw_pile_remaining(self) -> int:
        return len(self.wall) - self.draw_pile_pos


# ---- 工厂函数 ----
def new_game(known_hand: list[str], extra_visible: Optional[list[str]] = None) -> GameState:
    """创建新对局：给定我们的手牌，其余牌随机分配。

    Args:
        known_hand: 我们自己的手牌 (已知)
        extra_visible: 已知的可见牌 (牌河/碰杠, 会从剩余池中移除)
    """
    pool = list(ALL_TILES_136)
    for code in known_hand:
        pool.remove(code)
    if extra_visible:
        for code in extra_visible:
            if code in pool:
                pool.remove(code)
    random.shuffle(pool)

    # 分牌: 我们已有手牌，其他三家各 13 张
    p0_hand = list(known_hand)
    p1_hand = pool[:13]
    p2_hand = pool[13:26]
    p3_hand = pool[26:39]
    wall = pool[39:]

    players = [
        Player(hand=p0_hand),
        Player(hand=p1_hand),
        Player(hand=p2_hand),
        Player(hand=p3_hand),
    ]

    return GameState(wall=wall, players=players, current_player=0, phase=Phase.DRAW)


# ---- 游戏逻辑 ----
def get_winner(state: GameState) -> Optional[int]:
    """检查是否有人和牌。返回赢家 player index 或 None。"""
    for pi, p in enumerate(state.players):
        if is_winning_hand(p.hand):
            return pi
    return None


def step_game(state: GameState, our_policy_func=None) -> GameState:
    """执行一步游戏逻辑。

    our_policy_func(state) -> 返回我们出哪张牌 (code str)。
    用于 MCTS 里用我们的策略选择。
    """
    if state.is_finished:
        return state
    if state.turn_count >= state.max_turns:
        state.phase = Phase.FINISHED
        return state

    pi = state.current_player
    player = state.players[pi]

    if state.phase == Phase.DRAW:
        # 从牌墙抽一张
        if state.draw_pile_pos >= len(state.wall):
            state.phase = Phase.FINISHED
            return state
        drawn = state.wall[state.draw_pile_pos]
        state.draw_pile_pos += 1
        player.hand.append(drawn)
        player.hand.sort(key=lambda c: (c[1], int(c[0])))

        # 暗杠检查
        counts = _counts(player.hand)
        for code in list(counts.keys()):
            if counts[code] == 4:
                # 自动暗杠 (简单策略)
                for _ in range(4):
                    player.hand.remove(code)
                player.melds.append((code,) * 4)
                state.phase = Phase.DRAW  # 杠后补牌
                return step_game(state, our_policy_func)

        # 检查自摸
        if is_winning_hand(player.hand):
            state.phase = Phase.FINISHED
            state.current_player = pi
            return state

        state.phase = Phase.DISCARD
        state.turn_count += 1
        return state

    elif state.phase == Phase.DISCARD:
        # 选择出牌
        if pi == 0 and our_policy_func:
            discard = our_policy_func(state)
        else:
            discard = _heuristic_discard(player.hand)

        if discard in player.hand:
            player.hand.remove(discard)
        else:
            # fallback: 出最后一张
            discard = player.hand.pop()
        player.discards.append(discard)
        state.last_discard = discard
        state.last_discard_player = pi

        # 给其他三人检查碰/杠/和
        state.phase = Phase.WIN_CHECK
        return state

    elif state.phase == Phase.WIN_CHECK:
        # 其他玩家检查是否可以和这张牌
        discard = state.last_discard
        for other_pi in range(4):
            if other_pi == state.last_discard_player:
                continue
            test_hand = state.players[other_pi].hand + [discard]
            if is_winning_hand(test_hand):
                state.phase = Phase.FINISHED
                state.current_player = other_pi
                return state

        # 检查碰/杠
        state.phase = Phase.KONG_CHECK
        return step_game(state, our_policy_func)

    elif state.phase == Phase.KONG_CHECK:
        discard = state.last_discard
        # 只让下家杠 (简化; 实际规则任何家都可以)
        for other_pi in range(4):
            if other_pi == state.last_discard_player:
                continue
            counts = _counts(state.players[other_pi].hand)
            if counts[discard] >= 3:
                # 明杠
                p = state.players[other_pi]
                for _ in range(3):
                    p.hand.remove(discard)
                p.melds.append((discard,) * 4)
                state.current_player = other_pi
                state.phase = Phase.DRAW
                return step_game(state, our_policy_func)

        state.phase = Phase.PENG_CHECK
        return step_game(state, our_policy_func)

    elif state.phase == Phase.PENG_CHECK:
        discard = state.last_discard
        for other_pi in range(4):
            if other_pi == state.last_discard_player:
                continue
            counts = _counts(state.players[other_pi].hand)
            if counts[discard] >= 2:
                # 碰
                p = state.players[other_pi]
                for _ in range(2):
                    p.hand.remove(discard)
                p.melds.append((discard,) * 3)
                state.current_player = other_pi
                state.phase = Phase.DISCARD
                return state

        # 没人碰/杠/和，轮到下家
        state.current_player = (state.last_discard_player + 1) % 4
        state.phase = Phase.DRAW
        return state

    return state


def _heuristic_discard(hand: list[str]) -> str:
    """对手的快速出牌策略：打孤张或低价值牌。"""
    if len(hand) == 0:
        return ""
    if len(hand) <= 2:
        return hand[0]

    counts = _counts(hand)
    candidates = list(set(hand))
    if not candidates:
        return ""

    def _value(code: str) -> float:
        cnt = counts[code]
        if cnt >= 3:
            return 100
        if cnt == 2:
            return 50
        rank, suit = _parse_code(code)
        seq_score = 0
        if _is_number(code):
            for offset in (-2, -1, 1, 2):
                neighbor = f"{rank + offset}{suit}"
                if counts.get(neighbor, 0) > 0:
                    seq_score += 8
        honor_penalty = 18 if suit == "z" else 0
        return cnt * 35 + seq_score - honor_penalty - rank * 0.5

    candidates.sort(key=_value)
    return candidates[0]


def run_rollout(state: GameState, our_policy_func=None, max_turns: int = 80) -> tuple[int, int]:
    """从当前状态模拟到结束。返回 (outcome_code, winner)。

    outcome_code:
        0 = 我们 (player 0) 自摸
        1 = 我们点炮 (别人和我们的弃牌)
        2 = 别人和牌 (和我们无关)
        3 = 流局 / 其他
    """
    while not state.is_finished and state.turn_count < max_turns:
        step_game(state, our_policy_func)

    if state.phase != Phase.FINISHED:
        return (3, -1)  # 流局

    winner = state.current_player
    if winner == 0:
        return (0, 0)  # 我们和了
    elif state.last_discard_player == 0 and winner != 0:
        return (1, winner)  # 我们点炮
    else:
        return (2, winner)


# ---- MCTS 决策辅助 ----
def get_possible_discards(hand: list[str]) -> list[str]:
    """返回所有可出牌选项 (去重)。"""
    return sorted(set(hand), key=lambda c: (c[1], int(c[0])))


def evaluate_outcome(outcome: int, state: GameState, known_hand: list[str], discards: int) -> float:
    """评估对局结果的分数。"""
    if outcome == 0:
        # 自摸：高回报
        base = 100.0
        bonus = 0
        # 番型额外加分
        for m in state.players[0].melds:
            bonus += len(m) * 3
        return base + bonus
    elif outcome == 1:
        return -80.0  # 点炮
    elif outcome == 2:
        return -20.0  # 别人和牌
    else:
        # 流局: 听牌有分, 没听牌扣分
        if is_tenpai(known_hand):
            return 15.0
        return -5.0
