"""4 个 AI 机器人自对弈模拟训练 —— 差异化策略版。

策略分配:
  AI-1 (MCTS搜索):    mcts_discard 深度搜索，每次决策模拟 500 局
  AI-2 (启发式评估):   evaluator 评分选最优切牌
  AI-3 (激进进攻型):   优先组牌进攻，忽略防守安全
  AI-4 (保守防守型):   优先打安全牌，听牌才进攻

支持模式:
  python self_play.py 100          # 100局统计
  python self_play.py --detail     # 单局详细回放
"""

import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Callable

sys.path.insert(0, "src")
from majiang_ai.simulator import (
    ALL_TILES_136, _counts, is_winning_hand, is_tenpai, _parse_code, _is_number,
)


_SUIT_CN = {"m": "万", "p": "筒", "s": "条"}
_HONOR_CN = {"1z": "东", "2z": "南", "3z": "西", "4z": "北", "5z": "白", "6z": "发", "7z": "中"}

# 策略名称
_STRATEGY_LABELS = [
    "AI-1 (MCTS搜索)",
    "AI-2 (启发式评估)",
    "AI-3 (激进进攻型)",
    "AI-4 (保守防守型)",
]
_STRATEGY_SHORT = ["AI-1", "AI-2", "AI-3", "AI-4"]


def _cc(code: str) -> str:
    """编码 -> 中文牌名"""
    if code in _HONOR_CN:
        return _HONOR_CN[code]
    s = _SUIT_CN.get(code[1], "")
    return f"{int(code[0])}{s}"


def _sort_key(code: str) -> tuple:
    return ({"m": 0, "p": 1, "s": 2, "z": 3}.get(code[1], 9), int(code[0]))


def _sort_hand(hand: list[str]) -> list[str]:
    hand.sort(key=_sort_key)
    return hand


def _hand_str(hand: list[str]) -> str:
    return " ".join(_cc(c) for c in hand)


def _compact(hand: list[str]) -> str:
    groups: dict[str, list[str]] = {"m": [], "p": [], "s": [], "z": []}
    for c in hand:
        groups[c[1]].append(c[0])
    parts = []
    for s in "mpsz":
        if groups[s]:
            parts.append("".join(sorted(groups[s])) + s)
    return "".join(parts)


# ══════════════════════════════════════════════
# 四种差异化 AI 决策函数
# ══════════════════════════════════════════════

# ── AI-1: MCTS 搜索 ──

def _ai_mcts_discard(hand: list[str], visible_str: str) -> str:
    """MCTS 搜索最优切牌。"""
    if not hand:
        return ""
    if len(hand) <= 2:
        return hand[0]
    compact = _compact(hand)
    try:
        from majiang_ai.mcts import mcts_discard
        result = mcts_discard(
            hand_compact=compact,
            visible_compact=visible_str,
            simulations=200,
            mode="flat",
            time_limit_ms=3000,
        )
        if result.best_discard and result.best_discard in hand:
            return result.best_discard
    except Exception:
        pass
    from majiang_ai.simulator import _heuristic_discard
    return _heuristic_discard(hand)


# ── AI-2: 启发式评估 ──

def _ai_heuristic_discard(hand: list[str], visible_str: str) -> str:
    """evaluator 评分选最优切牌。"""
    if not hand:
        return ""
    if len(hand) <= 2:
        return hand[0]
    compact = _compact(hand)
    try:
        from majiang_ai.evaluator import evaluate_hand
        result = evaluate_hand(hand=compact, visible_tiles=visible_str)
        if result.options:
            return result.options[0].discard
    except Exception:
        pass
    from majiang_ai.simulator import _heuristic_discard
    return _heuristic_discard(hand)


# ── AI-3: 激进进攻型 ──

def _ai_aggressive_discard(hand: list[str], visible_str: str) -> str:
    """激进策略：优先进攻，忽略防守。

    特点:
      - 对子/刻子权重极高（积极碰杠）
      - 边张/孤张字牌最先丢
      - 不考虑对手河牌（不防守）
      - 哪怕快听牌了也敢打危险牌
    """
    if not hand:
        return ""
    if len(hand) <= 2:
        return hand[0]

    counts = _counts(hand)
    candidates = list(set(hand))

    def _value(code: str) -> float:
        cnt = counts[code]
        # 刻子: 极高保留
        if cnt >= 3:
            return 200
        # 对子: 高保留（能碰）
        if cnt == 2:
            return 90
        rank, suit = _parse_code(code)
        v = 20.0
        if _is_number(code):
            # 数牌: 周围搭子加分（大幅加码）
            for offset in (-2, -1, 1, 2):
                neighbor = f"{rank + offset}{suit}"
                if counts.get(neighbor, 0) > 0:
                    v += 12
            # 边张 1/9 大幅扣分（容易凑顺子）
            if rank in (1, 9):
                v -= 15
        else:
            # 字牌: 孤张字牌极低价值（激进型直接丢）
            v -= 30
        # 没有防守意识 = 不扣安全牌分
        return v

    candidates.sort(key=_value)
    return candidates[0]


# ── AI-4: 保守防守型 ──

def _ai_conservative_discard(hand: list[str], visible_str: str) -> str:
    """保守策略：优先防守，打安全牌。

    特点:
      - 分析对手河牌，避免点炮
      - 倾向保留字牌作为安全牌
      - 听牌后才转为进攻
      - 流局不扣分，宁愿流局也不点炮
    """
    if not hand:
        return ""
    if len(hand) <= 2:
        return hand[0]

    counts = _counts(hand)

    # 解析对手河牌（危险牌池）
    dangerous: set[str] = set()
    all_discarded: Counter[str] = Counter()
    if visible_str:
        for token in visible_str.split(","):
            token = token.strip()
            if token and len(token) >= 2:
                all_discarded[token] += 1
                # 数牌：周围 ±1 的牌也变危险（对手可能在等）
                if _is_number(token):
                    rank, suit = _parse_code(token)
                    for offset in (-2, -1, 1, 2):
                        neighbor = f"{rank + offset}{suit}"
                        dangerous.add(neighbor)
                # 字牌：被丢弃少的更危险
                elif token[1] == "z" and all_discarded[token] <= 1:
                    dangerous.add(token)

    # 安全牌 = 已被打出2张以上的牌
    safe_tiles = {c for c, n in all_discarded.items() if n >= 2}

    is_tenpai_now = is_tenpai(hand)

    candidates = list(set(hand))

    def _value(code: str) -> float:
        cnt = counts[code]
        rank, suit = _parse_code(code)

        # 听牌模式：切为进攻
        if is_tenpai_now:
            base = 150.0  # 高基线
            if cnt >= 3:
                return 500
            if cnt == 2:
                return 200
            if _is_number(code):
                for offset in (-2, -1, 1, 2):
                    if counts.get(f"{rank + offset}{suit}", 0) > 0:
                        base += 15
            return base

        # 非听牌模式：防守优先
        # 刻子/对子: 必须保留
        if cnt >= 3:
            return 500
        if cnt == 2:
            return 150

        v = 30.0

        # 数牌: 搭子加分
        if _is_number(code):
            for offset in (-2, -1, 1, 2):
                neighbor = f"{rank + offset}{suit}"
                if counts.get(neighbor, 0) > 0:
                    v += 10
            if rank in (1, 9):
                v -= 5
        else:
            # 字牌: 防守型倾向保留作为安全牌
            if code in safe_tiles or all_discarded.get(code, 0) >= 2:
                v += 30  # 安全字牌可保留
            else:
                v += 5   # 危险字牌先留

        # 惩罚危险牌
        if code in dangerous:
            v -= 40

        # 安全牌奖励
        if code in safe_tiles:
            v += 25

        return v

    candidates.sort(key=_value)
    return candidates[0]


# ── 策略分发 ──

_AI_STRATEGIES: dict[int, Callable] = {
    0: _ai_mcts_discard,
    1: _ai_heuristic_discard,
    2: _ai_aggressive_discard,
    3: _ai_conservative_discard,
}


def ai_discard(hand: list[str], visible_str: str, player_index: int) -> str:
    """按玩家索引分发到对应策略。"""
    func = _AI_STRATEGIES.get(player_index, _ai_heuristic_discard)
    return func(hand, visible_str)


# ══════════════════════════════════════════════
# 单局完整回放（逐回合）
# ══════════════════════════════════════════════

def play_one_detailed(max_turns: int = 100):
    """单局详细回放。"""
    pool = list(ALL_TILES_136)
    random.shuffle(pool)

    hands = [pool[0:13], pool[13:26], pool[26:39], pool[39:52]]
    for h in hands:
        _sort_hand(h)
    wall = pool[52:]
    wall_pos = 0
    melds = [[] for _ in range(4)]
    discards = [[] for _ in range(4)]
    current = 0

    def _vis(player_i: int) -> str:
        parts = []
        for p in range(4):
            if p != player_i:
                parts.extend(discards[p])
            for m in melds[p]:
                parts.extend(m)
        return ",".join(parts)

    pnames = _STRATEGY_LABELS

    print("=" * 70)
    print("  4-AI 自对弈 · 完整回放")
    print("=" * 70)
    print()
    for pi in range(4):
        print(f"  {pnames[pi]} 起手: {_hand_str(hands[pi])}")
    print()

    for turn in range(max_turns):
        pi = current
        hand = hands[pi]

        # ── 摸牌 ──
        if wall_pos >= len(wall):
            print(f"\n  牌墙耗尽，流局。")
            _print_final_state(hands, melds, discards, -1, "流局")
            return
        drawn = wall[wall_pos]; wall_pos += 1
        hand.append(drawn); _sort_hand(hand)
        print(f"  ── 回合 {turn+1} ──")
        print(f"  {pnames[pi]} 摸: {_cc(drawn)}")

        # 暗杠
        cnts = _counts(hand)
        did_kong = False
        for code in list(cnts.keys()):
            if cnts[code] == 4:
                for _ in range(4):
                    hand.remove(code)
                melds[pi].append([code]*4)
                if wall_pos < len(wall):
                    hand.append(wall[wall_pos]); wall_pos += 1; _sort_hand(hand)
                print(f"    → 暗杠 {_cc(code)}！补牌 {_cc(hand[-1])}")
                did_kong = True
                break

        # 自摸
        if is_winning_hand(hand):
            win_type = "杠上开花" if did_kong else "自摸"
            print(f"    ★ {pnames[pi]} {win_type}！")
            _print_final_state(hands, melds, discards, pi, win_type)
            return

        # 出牌
        visible = _vis(pi)
        discard = ai_discard(hand, visible, pi)
        if discard not in hand:
            discard = hand[-1]
        hand.remove(discard)
        discards[pi].append(discard)
        print(f"    → 打 {_cc(discard)}  | 手牌({len(hand)}张): {_hand_str(hand)}")

        # 他家响应：荣和
        ron = -1
        for other in range(4):
            if other == pi: continue
            if is_winning_hand(hands[other] + [discard]):
                ron = other; break
        if ron >= 0:
            print(f"  >>> {pnames[ron]} 荣和 {_cc(discard)}！")
            _print_final_state(hands, melds, discards, ron, "点炮")
            return

        # 杠
        kongger = -1
        for other in range(4):
            if other == pi: continue
            if _counts(hands[other])[discard] >= 3:
                kongger = other; break
        if kongger >= 0:
            hk = hands[kongger]
            for _ in range(3): hk.remove(discard)
            melds[kongger].append([discard]*4)
            if wall_pos < len(wall):
                hk.append(wall[wall_pos]); wall_pos += 1; _sort_hand(hk)
            print(f"  >>> {pnames[kongger]} 明杠 {_cc(discard)}！补牌 {_cc(hk[-1])}")
            if is_winning_hand(hk):
                print(f"  ★ {pnames[kongger]} 杠上开花！")
                _print_final_state(hands, melds, discards, kongger, "杠上开花")
                return
            current = kongger; continue

        # 碰
        penger = -1
        for other in range(4):
            if other == pi: continue
            if _counts(hands[other])[discard] >= 2:
                penger = other; break
        if penger >= 0:
            hp = hands[penger]
            hp.remove(discard); hp.remove(discard)
            melds[penger].append([discard]*3)
            hv = _vis(penger)
            pd = ai_discard(hp, hv, penger)
            if not pd or pd not in hp:
                pd = hp[-1] if hp else ""
            if pd:
                hp.remove(pd)
                discards[penger].append(pd)
            print(f"  >>> {pnames[penger]} 碰 {_cc(discard)} → 打 {_cc(pd)}  | 手牌({len(hp)}张): {_hand_str(hp)}")
            current = penger; continue

        # 下家
        current = (pi + 1) % 4

    _print_final_state(hands, melds, discards, -1, "流局(超回合)")


def _print_final_state(hands, melds, discards, winner, win_type):
    pnames = _STRATEGY_LABELS
    print()
    print("=" * 70)
    print(f"  终局: {win_type}" + (f" | 赢家: {pnames[winner]}" if winner >= 0 else ""))
    print("=" * 70)

    for pi in range(4):
        m = " ★" if winner == pi else ""
        print(f"\n  {pnames[pi]}{m}:")
        print(f"    手牌: {_hand_str(hands[pi]) if hands[pi] else '(无)'}")
        if melds[pi]:
            meld_str = " | ".join(
                "".join(_cc(c) for c in meld) for meld in melds[pi]
            )
            print(f"    副露: {meld_str}")
        if discards[pi]:
            print(f"    牌河({len(discards[pi])}张): {_hand_str(discards[pi])}")
        else:
            print(f"    牌河: (空)")

    print()
    for pi in range(4):
        if is_tenpai(hands[pi]):
            print(f"  {pnames[pi]} 已听牌✓")
    print("=" * 70)


# ══════════════════════════════════════════════
# 批量对局
# ══════════════════════════════════════════════

@dataclass
class GameResult:
    game_id: int
    winner: int
    win_type: str
    turns: int
    players_hand: list[list[str]] = field(default_factory=list)
    players_melds: list[list] = field(default_factory=list)
    players_discards: list[list[str]] = field(default_factory=list)
    dealer: int = 0  # 点炮者
    tenpai_players: list[int] = field(default_factory=list)


def play_one_fast(max_turns: int = 120) -> GameResult:
    pool = list(ALL_TILES_136)
    random.shuffle(pool)
    hands = [pool[0:13], pool[13:26], pool[26:39], pool[39:52]]
    for h in hands: _sort_hand(h)
    wall = pool[52:]; wall_pos = 0
    melds = [[] for _ in range(4)]
    discards = [[] for _ in range(4)]
    current = 0

    def _vis(pi):
        parts = []
        for p in range(4):
            if p != pi: parts.extend(discards[p])
            for m in melds[p]: parts.extend(m)
        return ",".join(parts)

    for turn in range(max_turns):
        pi = current; hand = hands[pi]
        if wall_pos >= len(wall):
            tenpai_list = [i for i in range(4) if is_tenpai(hands[i])]
            return GameResult(0, -1, "流局", turn, hands, melds, discards, -1, tenpai_list)

        hand.append(wall[wall_pos]); wall_pos += 1; _sort_hand(hand)

        # 暗杠
        cnts = _counts(hand)
        for code in list(cnts.keys()):
            if cnts[code] == 4:
                for _ in range(4): hand.remove(code)
                melds[pi].append([code]*4)
                if wall_pos < len(wall):
                    hand.append(wall[wall_pos]); wall_pos += 1; _sort_hand(hand)
                break

        if is_winning_hand(hand):
            return GameResult(0, pi, "自摸", turn, hands, melds, discards)

        discard = ai_discard(hand, _vis(pi), pi)
        if not discard or discard not in hand:
            discard = hand[-1]
        hand.remove(discard); discards[pi].append(discard)

        # 荣和
        for o in range(4):
            if o == pi: continue
            if is_winning_hand(hands[o] + [discard]):
                tenpai_list = [i for i in range(4) if i != o and is_tenpai(hands[i])]
                return GameResult(0, o, "点炮", turn+1, hands, melds, discards, pi, tenpai_list)

        # 杠
        for o in range(4):
            if o == pi: continue
            if _counts(hands[o])[discard] >= 3:
                ho = hands[o]
                for _ in range(3): ho.remove(discard)
                melds[o].append([discard]*4)
                if wall_pos < len(wall):
                    ho.append(wall[wall_pos]); wall_pos += 1; _sort_hand(ho)
                if is_winning_hand(ho):
                    return GameResult(0, o, "杠上开花", turn+2, hands, melds, discards)
                current = o; break
        else:
            # 碰
            for o in range(4):
                if o == pi: continue
                if _counts(hands[o])[discard] >= 2:
                    ho = hands[o]
                    ho.remove(discard); ho.remove(discard)
                    melds[o].append([discard]*3)
                    if not ho:
                        current = o; break
                    pd = ai_discard(ho, _vis(o), o)
                    if not pd or pd not in ho:
                        pd = ho[-1] if ho else ""
                    if pd:
                        ho.remove(pd); discards[o].append(pd)
                    current = o; break
            else:
                current = (pi + 1) % 4

    tenpai_list = [i for i in range(4) if is_tenpai(hands[i])]
    return GameResult(0, -1, "流局", max_turns, hands, melds, discards, -1, tenpai_list)


def run_tournament(n: int):
    wins = [0, 0, 0, 0]
    wtype = Counter()           # 全体和牌类型
    per_wtype = [Counter() for _ in range(4)]  # 每人自己的和牌类型分布
    self_draws = [0, 0, 0, 0]   # 每人自摸次数
    ron_wins = [0, 0, 0, 0]    # 每人荣和次数
    deal_in = [0, 0, 0, 0]     # 每人点炮次数
    draws = 0
    tturns = 0
    tenpai_counts = [0, 0, 0, 0]  # 流局时听牌次数
    last = None
    t0 = time.time()

    for g in range(n):
        r = play_one_fast()
        tturns += r.turns; last = r
        if r.winner >= 0:
            wins[r.winner] += 1
            wtype[r.win_type] += 1
            per_wtype[r.winner][r.win_type] += 1
            if r.win_type == "自摸" or r.win_type == "杠上开花":
                self_draws[r.winner] += 1
            elif r.win_type == "点炮":
                ron_wins[r.winner] += 1
                deal_in[r.dealer] += 1
        else:
            draws += 1
            for pi in r.tenpai_players:
                tenpai_counts[pi] += 1

        if (g + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  [{g+1}/{n}] {el:.1f}s")

    el = time.time() - t0
    total_valid = n - draws  # 有效对局（有人和牌）

    print(f"\n{'='*65}")
    print(f"  总局数: {n} | 有效对局: {total_valid} | 流局: {draws}")
    print(f"  耗时: {el:.1f}s | 平均回合: {tturns/max(1,n):.0f}")
    print()
    print(f"  {'策略':<22} {'胜场':>4} {'胜率':>7} {'自摸':>4} {'荣和':>4} {'点炮':>4} {'流局听牌':>8}")
    print(f"  {'─'*22} {'─'*4} {'─'*7} {'─'*4} {'─'*4} {'─'*4} {'─'*8}")

    for i in range(4):
        w = wins[i]
        pct = w / max(1, n) * 100
        bar = "█" * int(pct / 2)
        print(f"  {_STRATEGY_LABELS[i]:<22} {w:>4} {pct:>6.1f}% {self_draws[i]:>4} {ron_wins[i]:>4} {deal_in[i]:>4} {tenpai_counts[i]:>8}  {bar}")

    print()
    print("  全局和牌类型:")
    for t, c in wtype.most_common():
        print(f"    {t}: {c}次 ({c/max(1,total_valid)*100:.1f}%)")

    # 防守评分: 越低越好（点炮少+流局听牌多）
    print()
    print("  防守评分 (点炮越少/流局听牌越多越好):")
    for i in range(4):
        score = -deal_in[i] * 10 + tenpai_counts[i] * 3
        print(f"    {_STRATEGY_LABELS[i]}: {score:+d}")

    print("=" * 65)

    if last:
        if last.winner >= 0:
            print(f"\n  最近一局: {_STRATEGY_SHORT[last.winner]} {last.win_type}")
            if last.win_type == "点炮":
                print(f"    点炮者: {_STRATEGY_SHORT[last.dealer]}")
        else:
            print(f"\n  最近一局: 流局")
        for pi in range(4):
            m = " ★" if last.winner == pi else ""
            print(f"  {_STRATEGY_SHORT[pi]}{m}: 手牌{_hand_str(last.players_hand[pi])}")
            if last.players_melds[pi]:
                ms = "|".join("".join(_cc(c) for c in mld) for mld in last.players_melds[pi])
                print(f"    副露: {ms}")
            print(f"    牌河: {_hand_str(last.players_discards[pi])}")


# ══════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--detail":
        play_one_detailed()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
        print(f"开始 {n} 局 4-AI 自对弈 (差异化策略)...\n")
        run_tournament(n)
        print(f"\n运行单局回放: python self_play.py --detail")
