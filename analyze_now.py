"""分析当前手牌，给出最优出牌建议。"""
import sys, os, json
os.chdir(r"z:\app\majiang-ai-master")
sys.path.insert(0, "src")

from majiang_ai.capture import Capturer
from majiang_ai.vision import recognize_tiles, load_templates
from majiang_ai.evaluator import evaluate_hand, result_to_dict
from majiang_ai.simulator import is_tenpai, tenpai_waiting_tiles

templates = load_templates("tile_templates")
cfg = json.loads(open("calibration_config.json", encoding="utf-8").read())
tb = cfg.get("tiles")

cap = Capturer(); cap.start()
hand = cap.capture_hand()
river = cap.capture_river()
cap.stop()

result = recognize_tiles(hand, river, templates=templates, tile_bounds=tb)
compact = result.hand_compact
codes = [t.code for t in result.hand_tiles]

print("=" * 50)
print("当前牌面")
print("=" * 50)
if result.hand_tiles:
    names = " ".join(f"{t.code}({t.name})" for t in result.hand_tiles)
    print(f"手牌: {names}")
    print(f"编码: {compact}  共{len(result.hand_tiles)}张")
if result.river_compact:
    print(f"牌河: {result.river_compact}")

if result.errors:
    for e in result.errors:
        print(f"错误: {e}")

if not compact or len(compact) < 4:
    print("手牌不足，无法分析")
    sys.exit(0)

# 听牌检测
tenpai_now = is_tenpai(codes)
if tenpai_now:
    waits = tenpai_waiting_tiles(codes)
    print(f"\n当前听牌!  等待: {' '.join(waits)}")

# 完整分析
try:
    analysis = evaluate_hand(compact)
    ad = result_to_dict(analysis)

    if ad.get("mode") == "pre-draw" and ad.get("pre_draw"):
        print()
        print("=" * 50)
        print("摸牌分析 (模拟摸到每张牌后怎么打)")
        print("=" * 50)
        pre = ad["pre_draw"]
        shown = 0
        for pd in pre:
            if shown >= 8:
                break
            draw = pd.get("draw", "?")
            is_win = pd.get("is_win", False)
            was_tenpai = pd.get("was_tenpai", False)
            best_discard = pd.get("best_discard", "?")
            score = pd.get("score", 0)
            remaining = pd.get("remaining", 0)

            flag = ""
            if is_win:
                flag = " [自摸!]"
            elif was_tenpai:
                flag = " [听牌摸到]"
            print(f"摸 {draw}(剩{remaining}张) -> 打 {best_discard}  评分={score:.1f}{flag}")
            shown += 1

        # summary: most valuable discards when not tenpai
        if not tenpai_now:
            from collections import Counter
            discard_counter = Counter()
            for pd in pre[:20]:
                bd = pd.get("best_discard", "")
                if bd:
                    discard_counter[bd] += 1
            print()
            print("当前最佳出牌建议:")
            for tile, cnt in discard_counter.most_common(5):
                print(f"  打 {tile} (被 {cnt} 种摸牌选中)")

    else:
        print()
        print("=" * 50)
        print("出牌分析 (14张手牌)")
        print("=" * 50)
        for opt in ad.get("options", [])[:5]:
            tile = opt.get("discard", "?")
            score = opt.get("total_score", 0)
            eff = opt.get("effective_draws", 0)
            summary = opt.get("summary", "")
            print(f"  打 {tile}  评分={score:.1f}  有效进张={eff}")
            if summary:
                print(f"    {summary}")

except Exception as e:
    print(f"分析出错: {e}")
    import traceback; traceback.print_exc()
