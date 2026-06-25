"""
诊断脚本：逐张分析手牌匹配效果（使用 MatchEngine）
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from match_engine import MatchEngine, CODE_TO_NAME


def main():
    engine = MatchEngine()
    engine.load_all()
    print(f"模板数: {len(engine.templates)}")

    # 读取标定
    try:
        with open("calibration_config.json", "r", encoding="utf-8") as f:
            hand_region = json.load(f)["hand"]
        print(f"手牌区域: {hand_region}")
    except Exception:
        hand_region = None
        print("没有标定数据！")
        return

    # 打开 OBS 虚拟相机
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("打不开相机！")
        return

    # 跳帧稳定
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("读取帧失败")
        return

    h, w = frame.shape[:2]
    x1 = int(hand_region["left"] * w)
    y1 = int(hand_region["top"] * h)
    x2 = int((hand_region["left"] + hand_region["width"]) * w)
    y2 = int((hand_region["top"] + hand_region["height"]) * h)
    hand_crop = frame[y1:y2, x1:x2].copy()

    hh, ww = hand_crop.shape[:2]
    tile_w = ww // 14
    print(f"手牌区域: {ww}x{hh}, 每张宽: {tile_w}\n")

    # 逐张分析
    overlap = int(tile_w * 0.25)
    results = []
    for i in range(14):
        tx = max(0, i * tile_w - overlap)
        ex = min(ww, (i + 1) * tile_w + overlap)
        tile_img = hand_crop[:, tx:ex]
        top3 = engine.match_top3(tile_img)

        print(f"=== 位置 {i+1}/14 ===")
        for rank, (code, score) in enumerate(top3):
            name = CODE_TO_NAME.get(code, code)
            print(f"  {rank+1}. {name}({code}) 分数={score:.3f}")

        results.append(top3)

    # 汇总
    print("\n========== 汇总 ==========")
    best_set = []
    for i, cands in enumerate(results):
        if cands:
            code, score = cands[0]
            name = CODE_TO_NAME.get(code, code)
            print(f"位置{i+1}: {name} ({code}) 得分={score:.3f}")
            best_set.append(code)
        else:
            print(f"位置{i+1}: 未识别")
            best_set.append("?")

    # 整手匹配
    print("\n========== 整手匹配 (引擎) ==========")
    recognized_list, compact = engine.match_hand(hand_crop)
    print(f"逐位置: {' '.join(recognized_list)}")
    print(f"编码: {compact}")

    # 显示
    hand_display = None
    for i in range(14):
        tx = i * tile_w
        tile_img = hand_crop[:, tx:tx + tile_w].copy()
        if i < len(recognized_list) and recognized_list[i] != "?":
            label = recognized_list[i]
        elif i < len(results) and results[i]:
            label = results[i][0][0]
        else:
            label = "?"
        h = tile_img.shape[0]
        cv2.putText(tile_img, label, (5, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if hand_display is None:
            hand_display = tile_img
        else:
            hand_display = np.hstack([hand_display, tile_img])

    cv2.imshow("Diagnosis - Press any key", hand_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
