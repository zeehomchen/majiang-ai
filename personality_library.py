"""人格库系统 —— 30种预设人格 + 参数化决策 + 动态人格切换器。

理念:
  1. 预设数十种"人格"（不同权重偏好的打牌风格）
  2. 自对弈锦标赛，统计各人格胜率
  3. 人格切换器：根据当前手牌特征，动态选择最优人格
  4. 长期来看，切换器 > 任何单一人格

用法:
  python personality_library.py tournament 200   # 200局锦标赛
  python personality_library.py switcher 100     # 切换器 vs 静态人格 对比
  python personality_library.py list             # 列出所有人格
"""

from __future__ import annotations

import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

sys.path.insert(0, "src")
from majiang_ai.simulator import (
    ALL_TILES_136, _counts, is_winning_hand, is_tenpai,
    _parse_code, _is_number,
)

_SUIT_CN = {"m": "万", "p": "筒", "s": "条"}
_HONOR_CN = {"1z": "东", "2z": "南", "3z": "西", "4z": "北", "5z": "白", "6z": "发", "7z": "中"}


# ══════════════════════════════════════════════
# 人格参数定义
# ══════════════════════════════════════════════

@dataclass
class Personality:
    """一种打牌人格，7 维权重控制决策偏好。"""
    name: str           # 人格名称
    desc: str           # 简短描述
    # ── 核心权重 ──
    pair_weight: float = 70.0        # 对子保留价值
    triplet_weight: float = 180.0    # 刻子保留价值
    sequence_weight: float = 25.0    # 顺子/搭子加分
    # ── 攻防 ──
    defense_weight: float = 0.0      # 防守意识（0=不防，1=全力防）
    honor_keep: float = -15.0        # 字牌保留意愿（越高越留字牌）
    # ── 风格偏好 ──
    speed_bias: float = 0.0          # 速攻倾向（高=优先快听牌）
    terminal_bias: float = -3.0      # 幺九端牌偏好（高=喜欢留1/9）
    suit_focus: float = 0.0          # 单一花色倾向（0=不偏好）
    risk_tolerance: float = 0.5      # 风险容忍度（0=胆小，1=大胆）
    # ── 统计 ──
    wins: int = 0
    games: int = 0
    total_score: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.games, 1) * 100

    def params_dict(self) -> dict:
        return {
            "pair": self.pair_weight,
            "triplet": self.triplet_weight,
            "seq": self.sequence_weight,
            "defense": self.defense_weight,
            "honor": self.honor_keep,
            "speed": self.speed_bias,
            "terminal": self.terminal_bias,
            "suit": self.suit_focus,
            "risk": self.risk_tolerance,
        }


# ══════════════════════════════════════════════
# 参数化切牌函数
# ══════════════════════════════════════════════

def personality_discard(
    hand: list[str],
    personality: Personality,
    visible_str: str = "",
) -> str:
    """按人格参数决定切牌。"""
    if not hand:
        return ""
    if len(hand) <= 2:
        return hand[0]

    p = personality
    counts = _counts(hand)

    # ── 解析河牌（防守用）──
    dangerous: set[str] = set()
    all_discarded: Counter[str] = Counter()
    if visible_str and p.defense_weight > 0:
        for token in visible_str.split(","):
            token = token.strip()
            if token and len(token) >= 2:
                all_discarded[token] += 1
                if _is_number(token):
                    rank, suit = _parse_code(token)
                    for offset in (-2, -1, 1, 2):
                        dangerous.add(f"{rank + offset}{suit}")
                elif token[1] == "z" and all_discarded[token] <= 1:
                    dangerous.add(token)

    safe_tiles = {c for c, n in all_discarded.items() if n >= 2}
    is_tenpai_now = is_tenpai(hand)

    candidates = list(set(hand))

    def _value(code: str) -> float:
        cnt = counts[code]
        rank, suit = _parse_code(code)

        # 听牌模式：大幅提升保留价值
        if is_tenpai_now:
            if cnt >= 3:
                return 999
            if cnt == 2:
                return 500
            return 300

        # ── 核心评分 ──
        if cnt >= 3:
            return p.triplet_weight + cnt * 15
        if cnt == 2:
            return p.pair_weight

        v = p.speed_bias * 30 + 5.0

        # 顺子/搭子
        if _is_number(code):
            for offset in (-2, -1, 1, 2):
                neighbor = f"{rank + offset}{suit}"
                if counts.get(neighbor, 0) > 0:
                    v += p.sequence_weight
            # 幺九端牌
            if rank in (1, 9):
                v += p.terminal_bias
        else:
            # 字牌
            v += p.honor_keep

        # 防守
        if p.defense_weight > 0:
            if code in dangerous:
                v -= p.defense_weight * 80
            if code in safe_tiles:
                v += p.defense_weight * 40

        # 花色偏好
        if p.suit_focus > 0 and _is_number(code):
            target_suit = {"m": "m", "p": "p", "s": "s"}.get(code[1], "")
            if target_suit == "m":
                v += p.suit_focus * 8
            else:
                v -= p.suit_focus * 3

        return v

    # 风险容忍度影响排序方向
    candidates.sort(key=_value)

    # 高风险: 不打最低分 = 打相对高分牌（留好牌）
    # 低风险: 打最低分牌（打最没用的）
    if p.risk_tolerance > 0.7:
        return candidates[-1]  # 打高分牌 = 激进冒险
    return candidates[0]  # 打最低分牌 = 正常保守


# ══════════════════════════════════════════════
# 30 种预设人格
# ══════════════════════════════════════════════

PERSONALITY_LIBRARY: list[Personality] = [
    # ── 激进流 (5种) ──
    Personality("闪电侠", "极速抢听，不计后果",
                pair_weight=90, triplet_weight=200, sequence_weight=30, defense_weight=0.0, honor_keep=-30, speed_bias=1.0, terminal_bias=-10, risk_tolerance=0.9),
    Personality("抢攻手", "对子权重极高，碰牌狂魔",
                pair_weight=110, triplet_weight=220, sequence_weight=20, defense_weight=0.0, honor_keep=-25, speed_bias=0.8, terminal_bias=-5, risk_tolerance=0.8),
    Personality("冲锋队长", "宁可点炮也要进攻",
                pair_weight=80, triplet_weight=190, sequence_weight=25, defense_weight=-0.1, honor_keep=-35, speed_bias=1.0, terminal_bias=-8, risk_tolerance=1.0),
    Personality("快枪手", "只听不守，第一优先切听牌",
                pair_weight=75, triplet_weight=170, sequence_weight=35, defense_weight=0.0, honor_keep=-20, speed_bias=1.0, terminal_bias=-5, risk_tolerance=0.85),
    Personality("破釜沉舟", "全押进攻型，不留任何安全牌",
                pair_weight=100, triplet_weight=210, sequence_weight=15, defense_weight=-0.2, honor_keep=-40, speed_bias=1.0, terminal_bias=-12, risk_tolerance=1.0),

    # ── 碰碰胡专精 (5种) ──
    Personality("碰碰胡大师", "全力凑对子，不理顺子",
                pair_weight=130, triplet_weight=250, sequence_weight=-10, defense_weight=0.1, honor_keep=0, speed_bias=0.5, terminal_bias=5, risk_tolerance=0.5),
    Personality("刻子猎人", "见对就碰，三刻子起步",
                pair_weight=120, triplet_weight=260, sequence_weight=-5, defense_weight=0.05, honor_keep=5, speed_bias=0.4, terminal_bias=8, risk_tolerance=0.4),
    Personality("杠上开花", "疯狂凑杠，蹭杠上开花",
                pair_weight=100, triplet_weight=230, sequence_weight=0, defense_weight=0.1, honor_keep=0, speed_bias=0.3, terminal_bias=0, risk_tolerance=0.6),
    Personality("对对碰手", "小对子路线，稳中求快",
                pair_weight=110, triplet_weight=200, sequence_weight=10, defense_weight=0.2, honor_keep=5, speed_bias=0.6, terminal_bias=3, risk_tolerance=0.5),
    Personality("碰碰将军", "三刻子+两对子=完美手牌",
                pair_weight=115, triplet_weight=240, sequence_weight=5, defense_weight=0.15, honor_keep=2, speed_bias=0.5, terminal_bias=6, risk_tolerance=0.45),

    # ── 清/混一色流 (5种) ──
    Personality("清一色执念", "万子狂魔，只要万牌",
                pair_weight=70, triplet_weight=170, sequence_weight=30, defense_weight=0.2, honor_keep=-5, speed_bias=0.3, terminal_bias=0, suit_focus=2.5, risk_tolerance=0.4),
    Personality("混一色专家", "筒子+字牌，混一色路线",
                pair_weight=65, triplet_weight=160, sequence_weight=25, defense_weight=0.3, honor_keep=10, speed_bias=0.3, terminal_bias=0, suit_focus=1.5, risk_tolerance=0.3),
    Personality("条子世家", "只留条子，纯条路线",
                pair_weight=70, triplet_weight=180, sequence_weight=28, defense_weight=0.15, honor_keep=-10, speed_bias=0.4, terminal_bias=0, suit_focus=2.0, risk_tolerance=0.35),
    Personality("花色洁癖", "看不上杂色牌",
                pair_weight=60, triplet_weight=165, sequence_weight=22, defense_weight=0.25, honor_keep=0, speed_bias=0.35, terminal_bias=0, suit_focus=1.8, risk_tolerance=0.3),
    Personality("混清双修", "混一色优先，不行转清",
                pair_weight=65, triplet_weight=170, sequence_weight=26, defense_weight=0.2, honor_keep=8, speed_bias=0.4, terminal_bias=2, suit_focus=1.2, risk_tolerance=0.4),

    # ── 防守流 (5种) ──
    Personality("铜墙铁壁", "铁桶防守，绝不打危险牌",
                pair_weight=55, triplet_weight=140, sequence_weight=20, defense_weight=0.9, honor_keep=20, speed_bias=0.0, terminal_bias=0, risk_tolerance=0.05),
    Personality("忍者", "忍到听牌才出手",
                pair_weight=60, triplet_weight=150, sequence_weight=22, defense_weight=0.7, honor_keep=15, speed_bias=0.1, terminal_bias=2, risk_tolerance=0.1),
    Personality("乌龟战术", "活着就是胜利",
                pair_weight=50, triplet_weight=130, sequence_weight=18, defense_weight=1.0, honor_keep=30, speed_bias=-0.2, terminal_bias=5, risk_tolerance=0.0),
    Personality("安全第一", "先保证不点炮，再考虑赢",
                pair_weight=55, triplet_weight=145, sequence_weight=19, defense_weight=0.8, honor_keep=25, speed_bias=0.0, terminal_bias=3, risk_tolerance=0.1),
    Personality("后发制人", "对手先出招，我再反击",
                pair_weight=58, triplet_weight=148, sequence_weight=21, defense_weight=0.6, honor_keep=12, speed_bias=0.2, terminal_bias=1, risk_tolerance=0.2),

    # ── 均衡流 (5种) ──
    Personality("太极手", "不偏不倚，中庸之道",
                pair_weight=70, triplet_weight=180, sequence_weight=25, defense_weight=0.4, honor_keep=0, speed_bias=0.4, terminal_bias=0, risk_tolerance=0.5),
    Personality("随机应变", "什么来牌就走什么路",
                pair_weight=65, triplet_weight=175, sequence_weight=28, defense_weight=0.35, honor_keep=2, speed_bias=0.5, terminal_bias=0, risk_tolerance=0.5),
    Personality("万能型", "所有路线都试，灵活切换",
                pair_weight=68, triplet_weight=178, sequence_weight=26, defense_weight=0.3, honor_keep=0, speed_bias=0.45, terminal_bias=1, risk_tolerance=0.45),
    Personality("中庸大师", "不求有功，但求无过",
                pair_weight=62, triplet_weight=165, sequence_weight=24, defense_weight=0.5, honor_keep=5, speed_bias=0.3, terminal_bias=0, risk_tolerance=0.35),
    Personality("变通王", "根据局势动态调整权重",
                pair_weight=72, triplet_weight=182, sequence_weight=27, defense_weight=0.3, honor_keep=0, speed_bias=0.5, terminal_bias=0, risk_tolerance=0.5),

    # ── 幺九/字牌流 (5种) ──
    Personality("十三幺信徒", "一心奔十三幺",
                pair_weight=40, triplet_weight=120, sequence_weight=-15, defense_weight=0.3, honor_keep=20, speed_bias=0.1, terminal_bias=25, risk_tolerance=0.2),
    Personality("字牌收藏家", "字牌一张不丢",
                pair_weight=45, triplet_weight=130, sequence_weight=0, defense_weight=0.4, honor_keep=40, speed_bias=0.1, terminal_bias=10, risk_tolerance=0.15),
    Personality("混幺九使者", "幺九字牌一手抓",
                pair_weight=50, triplet_weight=140, sequence_weight=-5, defense_weight=0.3, honor_keep=15, speed_bias=0.2, terminal_bias=20, risk_tolerance=0.2),
    Personality("大三元梦想家", "三元牌一个不放过",
                pair_weight=55, triplet_weight=150, sequence_weight=10, defense_weight=0.2, honor_keep=25, speed_bias=0.2, terminal_bias=5, risk_tolerance=0.25),
    Personality("风牌爱好者", "东南西北全留",
                pair_weight=48, triplet_weight=135, sequence_weight=0, defense_weight=0.35, honor_keep=30, speed_bias=0.15, terminal_bias=5, risk_tolerance=0.15),
]


# ══════════════════════════════════════════════
# 人格切换器 —— 根据手牌动态选人格
# ══════════════════════════════════════════════

@dataclass
class PersonalitySwitcher:
    """人格切换器：开局锁定，中途仅在大事件后切换。

    核心逻辑：开局分析手牌→选定最优人格→锁定。
    只在碰/杠/牌型发生重大变化时才重新评估。
    避免"每回合换人格"导致的策略漂移。
    """
    library: list[Personality] = field(default_factory=lambda: PERSONALITY_LIBRARY)
    locked: Optional[Personality] = None
    locked_name: str = ""
    last_hand_hash: int = 0
    switch_count: int = 0
    _last_pairs: int = 0
    _last_triplets: int = 0
    _last_honors: int = 0
    _last_suit_conc: float = 0.0

    def analyze_hand(self, hand: list[str]) -> dict:
        """分析手牌特征，返回特征向量。"""
        counts = _counts(hand)
        pairs = sum(1 for c in counts.values() if c >= 2)
        triplets = sum(1 for c in counts.values() if c >= 3)
        quads = sum(1 for c in counts.values() if c == 4)

        suits: Counter[str] = Counter()
        honors = 0
        terminals = 0
        for c in hand:
            if c[1] == "z":
                honors += 1
            else:
                suits[c[1]] += 1
                if int(c[0]) in (1, 9):
                    terminals += 1

        # 搭子/孤张
        isolated = 0
        partial_seq = 0
        for c in set(hand):
            cnt = counts[c]
            if cnt >= 2:
                continue
            if _is_number(c):
                rank, suit = _parse_code(c)
                has_neighbor = False
                for offset in (-2, -1, 1, 2):
                    if counts.get(f"{rank + offset}{suit}", 0) > 0:
                        has_neighbor = True
                        partial_seq += 1
                        break
                if not has_neighbor:
                    isolated += 1
            else:
                isolated += cnt

        # 花色集中度
        max_suit = suits.most_common(1)[0][1] if suits else 0
        suit_concentration = max_suit / max(len(hand), 1)

        return {
            "total": len(hand),
            "pairs": pairs,
            "triplets": triplets,
            "quads": quads,
            "honors": honors,
            "terminals": terminals,
            "isolated": isolated,
            "partial_seq": partial_seq,
            "suit_concentration": suit_concentration,
            "dominant_suit": suits.most_common(1)[0][0] if suits else "",
            "is_tenpai": is_tenpai(hand),
        }

    def select_personality(self, hand: list[str], visible_str: str = "",
                           turn_count: int = 0, force_reeval: bool = False) -> tuple[Personality, str]:
        """选择人格。已锁定时只在重大变化时切换。"""
        # 计算手牌特征哈希
        feat = self.analyze_hand(hand)
        hand_hash = hash((
            feat["pairs"], feat["triplets"], feat["suit_concentration"],
            feat["honors"], feat["terminals"], feat["isolated"],
        ))

        # 已有锁定人格：检查是否需要切换
        if self.locked is not None and not force_reeval:
            # 只在牌型发生重大变化时重新评估
            re_eval = False
            re_eval = re_eval or abs(feat["pairs"] - self._last_pairs) >= 2
            re_eval = re_eval or feat["triplets"] != self._last_triplets
            re_eval = re_eval or abs(feat["honors"] - self._last_honors) >= 3
            re_eval = re_eval or abs(feat["suit_concentration"] - self._last_suit_conc) > 0.25
            re_eval = re_eval or feat["is_tenpai"]  # 听牌了一定要切换

            if not re_eval:
                return self.locked, f"锁定: {self.locked_name}"

        # 重新评估
        chosen, reason = self._evaluate(feat)
        self.locked = chosen
        self.locked_name = chosen.name
        self._last_pairs = feat["pairs"]
        self._last_triplets = feat["triplets"]
        self._last_honors = feat["honors"]
        self._last_suit_conc = feat["suit_concentration"]
        self.switch_count += 1
        return chosen, reason

    def _evaluate(self, feat: dict) -> tuple[Personality, str]:

        # 听牌 → 保守防守人格（避免打危险牌炸听）
        if feat["is_tenpai"]:
            return self._match("乌龟战术"), "已听牌，转保守求稳"

        # 碰碰胡路线：对子+刻子多
        if feat["pairs"] + feat["triplets"] >= 4:
            return self._match("碰碰胡大师"), f"对子{feat['pairs']}+刻子{feat['triplets']} → 碰碰胡"

        # 清一色 / 混一色路线
        if feat["suit_concentration"] >= 0.6 and feat["honors"] <= 3:
            if feat["honors"] == 0:
                return self._match("清一色执念"), f"花色集中度{feat['suit_concentration']:.0%} → 清一色"
            else:
                return self._match("混一色专家"), f"花色集中度{feat['suit_concentration']:.0%}+字牌 → 混一色"

        # 幺九/字牌多 → 十三幺
        if feat["terminals"] + feat["honors"] >= 9:
            return self._match("十三幺信徒"), f"幺九字牌{feat['terminals']+feat['honors']}张 → 十三幺"

        # 字牌多 → 字牌流
        if feat["honors"] >= 5:
            return self._match("字牌收藏家"), f"字牌{feat['honors']}张 → 字牌流"

        # 孤张多 / 牌型散 → 速攻激进
        if feat["isolated"] >= 4:
            return self._match("闪电侠"), f"孤张{feat['isolated']}张 → 速攻清孤张"

        # 搭子多 → 均衡
        if feat["partial_seq"] >= 3:
            return self._match("随机应变"), f"搭子{feat['partial_seq']}组 → 均衡发育"

        # 默认：均衡型
        return self._match("太极手"), "牌型均衡 → 太极手"

    def _match(self, name: str) -> Personality:
        for p in self.library:
            if p.name == name:
                return p
        return self.library[0]


# ══════════════════════════════════════════════
# 锦标赛：所有人大乱斗
# ══════════════════════════════════════════════

@dataclass
class TournamentResult:
    name: str
    wins: int
    games: int
    self_draws: int
    ron_wins: int
    deal_in: int


def _shuffle_pool() -> tuple[list[list[str]], list[str], int]:
    pool = list(ALL_TILES_136)
    random.shuffle(pool)
    hands = [pool[0:13], pool[13:26], pool[26:39], pool[39:52]]
    for h in hands:
        h.sort(key=lambda c: ({"m": 0, "p": 1, "s": 2, "z": 3}.get(c[1], 9), int(c[0])))
    return hands, pool[52:], 0


def _visible_str(discards, melds, pi):
    parts = []
    for p in range(4):
        if p != pi:
            parts.extend(discards[p])
        for m in melds[p]:
            parts.extend(m)
    return ",".join(parts)


def play_one_with_personalities(personalities: list[Personality], max_turns: int = 120) -> tuple[int, str, int]:
    """4 人格对弈一局。返回 (winner_idx, win_type, dealer_idx)。"""
    hands, wall, wall_pos = _shuffle_pool()
    melds = [[] for _ in range(4)]
    discards = [[] for _ in range(4)]
    current = 0
    wall = list(wall)

    for turn in range(max_turns):
        pi = current
        hand = hands[pi]

        if wall_pos >= len(wall):
            return -1, "流局", -1

        hand.append(wall[wall_pos]); wall_pos += 1
        hand.sort(key=lambda c: ({"m": 0, "p": 1, "s": 2, "z": 3}.get(c[1], 9), int(c[0])))

        # 暗杠
        cnts = _counts(hand)
        for code in list(cnts.keys()):
            if cnts[code] == 4:
                for _ in range(4): hand.remove(code)
                melds[pi].append([code]*4)
                if wall_pos < len(wall):
                    hand.append(wall[wall_pos]); wall_pos += 1
                    hand.sort(key=lambda c: ({"m": 0, "p": 1, "s": 2, "z": 3}.get(c[1], 9), int(c[0])))
                break

        if is_winning_hand(hand):
            return pi, "自摸", -1

        discard = personality_discard(hand, personalities[pi], _visible_str(discards, melds, pi))
        if not discard or discard not in hand:
            discard = hand[-1]
        hand.remove(discard)
        discards[pi].append(discard)

        # 荣和
        for o in range(4):
            if o == pi: continue
            if is_winning_hand(hands[o] + [discard]):
                return o, "点炮", pi

        # 杠
        kong_done = False
        for o in range(4):
            if o == pi: continue
            if _counts(hands[o])[discard] >= 3:
                ho = hands[o]
                for _ in range(3): ho.remove(discard)
                melds[o].append([discard]*4)
                if wall_pos < len(wall):
                    ho.append(wall[wall_pos]); wall_pos += 1
                    ho.sort(key=lambda c: ({"m": 0, "p": 1, "s": 2, "z": 3}.get(c[1], 9), int(c[0])))
                if is_winning_hand(ho):
                    return o, "杠上开花", -1
                current = o
                kong_done = True
                break
        if kong_done:
            continue

        # 碰
        pong_done = False
        for o in range(4):
            if o == pi: continue
            if _counts(hands[o])[discard] >= 2:
                ho = hands[o]
                ho.remove(discard); ho.remove(discard)
                melds[o].append([discard]*3)
                if not ho:
                    current = o; pong_done = True; break
                pd = personality_discard(ho, personalities[o], _visible_str(discards, melds, o))
                if not pd or pd not in ho:
                    pd = ho[-1] if ho else ""
                if pd:
                    ho.remove(pd); discards[o].append(pd)
                current = o; pong_done = True; break
        if pong_done:
            continue

        current = (pi + 1) % 4

    return -1, "流局", -1


def run_tournament(n_games: int = 200):
    """所有人格随机组队大乱斗。"""
    print(f"人格库锦标赛: {n_games} 局\n")
    all_p = PERSONALITY_LIBRARY
    stats: dict[str, TournamentResult] = {}

    for p in all_p:
        stats[p.name] = TournamentResult(name=p.name, wins=0, games=0,
                                          self_draws=0, ron_wins=0, deal_in=0)

    t0 = time.time()
    for g in range(n_games):
        # 随机抽 4 个人格
        selected = random.sample(all_p, 4)
        for p in selected:
            stats[p.name].games += 1

        winner, wtype, dealer = play_one_with_personalities(selected)

        if winner >= 0:
            stats[selected[winner].name].wins += 1
            if wtype == "自摸" or wtype == "杠上开花":
                stats[selected[winner].name].self_draws += 1
            elif wtype == "点炮":
                stats[selected[winner].name].ron_wins += 1
                stats[selected[dealer].name].deal_in += 1

        if (g + 1) % 50 == 0:
            print(f"  [{g+1}/{n_games}] {time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  锦标赛完成: {n_games}局 | 耗时: {elapsed:.1f}s")
    print(f"\n  {'人格':<16} {'出场':>4} {'胜场':>4} {'胜率':>7} {'自摸':>4} {'荣和':>4} {'点炮':>4}")
    print(f"  {'─'*16} {'─'*4} {'─'*4} {'─'*7} {'─'*4} {'─'*4} {'─'*4}")

    ranked = sorted(stats.values(), key=lambda s: s.wins / max(s.games, 1), reverse=True)
    top10 = ranked[:10]
    for r in top10:
        wr = r.wins / max(r.games, 1) * 100
        print(f"  {r.name:<16} {r.games:>4} {r.wins:>4} {wr:>6.1f}% {r.self_draws:>4} {r.ron_wins:>4} {r.deal_in:>4}")

    print(f"\n  ... 共 {len(ranked)} 种人格参与")
    print(f"{'='*70}")

    return ranked


# ══════════════════════════════════════════════
# 切换器 vs 静态人格 对比测试
# ══════════════════════════════════════════════

def run_switcher_vs_static(n_games: int = 100):
    """人格切换器 vs 静态 Top 人格，对比胜率。

    切换器策略：
      开局分析手牌 → 从锦标赛Top人格中选最匹配的 → 全程锁定不换。
      只在碰/杠后、听牌后允许切换。
    """
    print(f"人格切换器对比测试: {n_games} 局\n")

    # 用锦标赛排名前5的人格作为候选
    static_contestants = ["碰碰胡大师", "闪电侠", "铜墙铁壁", "太极手", "清一色执念"]
    static_map = {n: _match_name(n) for n in static_contestants}

    wins = [0, 0, 0, 0]
    deal_ins = [0, 0, 0, 0]
    self_draws = [0, 0, 0, 0]
    labels = ["切换器(开局锁定)", "", "", ""]
    total_switches = 0
    t0 = time.time()

    for g in range(n_games):
        switcher = PersonalitySwitcher()
        opponents = random.sample(static_contestants, 3)
        labels[1] = opponents[0]
        labels[2] = opponents[1]
        labels[3] = opponents[2]
        static_ps = [static_map[n] for n in opponents]

        hands, wall, wall_pos = _shuffle_pool()
        melds = [[] for _ in range(4)]
        discards = [[] for _ in range(4)]
        current = 0
        wall = list(wall)

        # 开局立刻选人格并锁定
        init_vis = ""
        chosen_p, reason = switcher.select_personality(hands[0], init_vis, 0)
        # 增加日志（前5局）
        if g < 5:
            print(f"  [局{g+1}] 切换器开局手牌分析 → 选定: {chosen_p.name} ({reason})")

        game_over = False
        for turn in range(120):
            if game_over:
                break
            pi = current
            hand = hands[pi]
            if wall_pos >= len(wall):
                break

            hand.append(wall[wall_pos]); wall_pos += 1
            hand.sort(key=lambda c: ({"m": 0, "p": 1, "s": 2, "z": 3}.get(c[1], 9), int(c[0])))

            cnts = _counts(hand)
            for code in list(cnts.keys()):
                if cnts[code] == 4:
                    for _ in range(4): hand.remove(code)
                    melds[pi].append([code]*4)
                    if wall_pos < len(wall):
                        hand.append(wall[wall_pos]); wall_pos += 1
                        hand.sort(key=lambda c: ({"m": 0, "p": 1, "s": 2, "z": 3}.get(c[1], 9), int(c[0])))
                    break

            if is_winning_hand(hand):
                wins[pi] += 1; self_draws[pi] += 1
                break

            vis = _visible_str(discards, melds, pi)
            if pi == 0:
                # 只用锁定人格（允许碰/杠后强制重评）
                p = chosen_p
            else:
                p = static_ps[pi - 1]

            discard = personality_discard(hand, p, vis)
            if not discard or discard not in hand:
                discard = hand[-1]
            hand.remove(discard)
            discards[pi].append(discard)

            # 荣和
            for o in range(4):
                if o == pi: continue
                if is_winning_hand(hands[o] + [discard]):
                    wins[o] += 1; deal_ins[pi] += 1
                    game_over = True; break
            if game_over:
                break

            # 杠
            kong_done = False
            for o in range(4):
                if o == pi: continue
                if _counts(hands[o])[discard] >= 3:
                    ho = hands[o]
                    for _ in range(3): ho.remove(discard)
                    melds[o].append([discard]*4)
                    if wall_pos < len(wall):
                        ho.append(wall[wall_pos]); wall_pos += 1
                        ho.sort(key=lambda c: ({"m": 0, "p": 1, "s": 2, "z": 3}.get(c[1], 9), int(c[0])))
                    if is_winning_hand(ho):
                        wins[o] += 1; self_draws[o] += 1
                        game_over = True
                    else:
                        current = o; kong_done = True
                    break
            if game_over:
                break
            if kong_done:
                continue

            # 碰
            pong_done = False
            for o in range(4):
                if o == pi: continue
                if _counts(hands[o])[discard] >= 2:
                    ho = hands[o]
                    ho.remove(discard); ho.remove(discard)
                    melds[o].append([discard]*3)
                    if not ho:
                        current = o; pong_done = True; break
                    vis_o = _visible_str(discards, melds, o)
                    if o == 0:
                        # 碰后强制重评
                        chosen_p, _ = switcher.select_personality(ho, vis_o, turn, force_reeval=True)
                    cp = chosen_p if o == 0 else static_ps[o - 1]
                    pd = personality_discard(ho, cp, vis_o)
                    if not pd or pd not in ho:
                        pd = ho[-1] if ho else ""
                    if pd:
                        ho.remove(pd); discards[o].append(pd)
                    current = o; pong_done = True; break
            if pong_done:
                continue

            current = (pi + 1) % 4

        total_switches += switcher.switch_count

        if (g + 1) % 20 == 0:
            print(f"  [{g+1}/{n_games}] {time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  切换器对比测试: {n_games}局 | 耗时: {elapsed:.1f}s")
    print(f"  切换器平均切换: {total_switches/max(n_games,1):.1f} 次/局")
    print(f"\n  {'角色':<20} {'胜场':>4} {'胜率':>7} {'自摸':>4} {'点炮':>4}")
    print(f"  {'─'*20} {'─'*4} {'─'*7} {'─'*4} {'─'*4}")

    for i in range(4):
        wr = wins[i] / max(n_games, 1) * 100
        print(f"  {labels[i]:<20} {wins[i]:>4} {wr:>6.1f}% {self_draws[i]:>4} {deal_ins[i]:>4}")

    sw_wr = wins[0] / max(n_games, 1) * 100
    static_avg = sum(wins[1:]) / 3 / max(n_games, 1) * 100
    print(f"\n  切换器胜率: {sw_wr:.1f}%  |  静态对手平均: {static_avg:.1f}%")
    gap = sw_wr - static_avg
    if gap > 0:
        print(f"  ★ 切换器领先 +{gap:.1f}%")
    else:
        print(f"  差距: {gap:+.1f}%")
    print(f"{'='*65}")

    return wins, labels


def _match_name(name: str) -> Personality:
    for p in PERSONALITY_LIBRARY:
        if p.name == name:
            return p
    return PERSONALITY_LIBRARY[0]


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════

def list_personalities():
    print(f"人格库: {len(PERSONALITY_LIBRARY)} 种人格\n")
    print(f"  {'名称':<16} {'描述':<24} {'对子':>4} {'刻子':>4} {'顺子':>4} {'防守':>4} {'字牌':>4} {'速攻':>4} {'风险':>4}")
    print(f"  {'─'*16} {'─'*24} {'─'*4} {'─'*4} {'─'*4} {'─'*4} {'─'*4} {'─'*4} {'─'*4}")
    for p in PERSONALITY_LIBRARY:
        print(f"  {p.name:<16} {p.desc:<24} {p.pair_weight:>4.0f} {p.triplet_weight:>4.0f} {p.sequence_weight:>4.0f} "
              f"{p.defense_weight:>4.1f} {p.honor_keep:>4.0f} {p.speed_bias:>4.1f} {p.risk_tolerance:>4.1f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python personality_library.py list              # 列出所有人格")
        print("  python personality_library.py tournament 200    # 200局锦标赛")
        print("  python personality_library.py switcher 100      # 切换器对比测试")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "list":
        list_personalities()
    elif cmd == "tournament":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        run_tournament(n)
    elif cmd == "switcher":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        run_switcher_vs_static(n)
    else:
        print(f"未知命令: {cmd}")
