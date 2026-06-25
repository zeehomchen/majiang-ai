"""
简化版自动识别
直接用模板匹配，显示调试信息！
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

# 添加项目路径
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np
from majiang_ai import evaluate_hand


def load_templates_simple(template_dir):
    """简单加载模板"""
    templates = {}
    import re
    valid = re.compile(r'^[1-9][mpsz]$')
    for f in template_dir.iterdir():
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            code = f.stem
            if valid.match(code):
                img = cv2.imread(str(f))
                if img is not None:
                    templates[code] = img
    return templates


def load_calibration():
    try:
        with open("calibration_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg["hand"]
    except Exception:
        return None


def match_single(crop, templates):
    """匹配单张牌，返回最相似的"""
    best_code = None
    best_score = 0
    if crop is None or crop.size == 0:
        return best_code, best_score

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

    for code, tmpl in templates.items():
        # 调整模板大小和裁剪一致
        tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY) if len(tmpl.shape) == 3 else tmpl
        h, w = gray.shape
        if w > 0 and h > 0:
            tmpl_resized = cv2.resize(tmpl_gray, (w, h))
            # 匹配
            try:
                result = cv2.matchTemplate(gray, tmpl_resized, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                    best_code = code
            except Exception:
                pass
    return best_code, best_score


def main():
    print("=" * 60)
    print("简化版自动识别 - 调试版")
    print("=" * 60)

    # 加载模板
    template_dir = ROOT / "tile_templates"
    templates = load_templates_simple(template_dir)
    print(f"✅ 已加载模板: {list(templates.keys())}")

    if len(templates) == 0:
        print("\n❌ 没有找到模板！")
        return

    # 加载标定
    hand_region = load_calibration()
    if not hand_region:
        print("❌ 没有标定！")
        return

    # 直接打开相机
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 打不开相机！")
            return

    print("✅ 相机已打开！按 'q' 退出，按 's' 保存截图\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cv2.waitKey(30)
                continue

            display = frame.copy()
            h, w = display.shape[:2]

            # 画标定区域
            x1 = int(hand_region["left"] * w)
            y1 = int(hand_region["top"] * h)
            x2 = int((hand_region["left"] + hand_region["width"]) * w)
            y2 = int((hand_region["top"] + hand_region["height"]) * h)

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # 裁剪区域
            hand_crop = frame[y1:y2, x1:x2].copy()

            # 识别牌
            recognized_codes = []
            if hand_crop.size > 0:
                # 简单分割：假设牌是等宽的
                h_h, w_h = hand_crop.shape[:2]
                if len(templates) > 0:
                    num_tiles = 14  # 假设14张牌
                    tile_w = w_h // num_tiles
                    if tile_w > 10:
                        # 切分并识别
                        for i in range(num_tiles):
                            tx1 = i * tile_w
                            tx2 = (i+1)*tile_w
                            tile_crop = hand_crop[:, tx1:tx2].copy()
                            # 匹配
                            code, score = match_single(tile_crop, templates)
                            if code and score > 0.3:
                                recognized_codes.append(code)

                                # 画出来
                                cx = x1 + (i * tile_w) + (tile_w // 2)
                                cy = y1 + (h_h // 2)
                                cv2.putText(
                                    display,
                                    code,
                                    (cx - 10, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0),
                                    2
                                )
                            else:
                                # 画框
                                cx = x1 + (i * tile_w) + (tile_w // 2)
                                cy = y1 + (h_h // 2)
                                cv2.rectangle(
                                    display,
                                    (x1 + i * tile_w, y1),
                                    (x1 + (i+1)*tile_w, y2),
                                    (0, 0, 255),
                                    1
                                )

            # 整理编码
            if recognized_codes:
                # 分组并排序
                groups = {"m": [], "p": [], "s": [], "z": []}
                for code in recognized_codes:
                    num = code[0]
                    suit = code[1]
                    groups[suit].append(num)

                # 构建紧凑编码
                compact = ""
                for suit in ["m", "p", "s", "z"]:
                    if groups[suit]:
                        compact += "".join(sorted(groups[suit])) + suit

                # 算牌
                try:
                    eval_result = evaluate_hand(compact)
                    # 显示结果
                    y_off = 30
                    cv2.putText(
                        display,
                        f"Hand: {compact}",
                        (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )
                    y_off += 30

                    if eval_result.options:
                        best = eval_result.options[0]
                        cv2.putText(
                            display,
                            f"Discard: {best.discard}",
                            (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            3
                        )
                except Exception as e:
                    pass

            # 显示
            cv2.imshow("Auto Recognition - Simple", display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("auto_recognition.png", display)
                print("📸 截图已保存！")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
