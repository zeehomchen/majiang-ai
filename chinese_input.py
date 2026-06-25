"""
中文输入版本
支持输入: "一万二万三万 一条二条 红中 这样的
"""

from __future__ import annotations

import sys
import json
import threading
from pathlib import Path

# 添加项目路径
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
from majiang_ai.virtual_camera import VirtualCameraCapture, VirtualCameraConfig
from majiang_ai import evaluate_hand


# 中文名称 -> 编码
NAME_TO_CODE = {
    # 万子
    "一万": "1m", "二万": "2m", "三万": "3m",
    "四万": "4m", "五万": "5m", "六万": "6m",
    "七万": "7m", "八万": "8m", "九万": "9m",
    # 筒子
    "一筒": "1p", "二筒": "2p", "三筒": "3p",
    "四筒": "4p", "五筒": "5p", "六筒": "6p",
    "七筒": "7p", "八筒": "8p", "九筒": "9p",
    # 条子
    "一条": "1s", "二条": "2s", "三条": "3s",
    "四条": "4s", "五条": "5s", "六条": "6s",
    "七条": "7s", "八条": "8s", "九条": "9s",
    # 字牌
    "东": "1z", "南": "2z", "西": "3z", "北": "4z",
    "白板": "5z", "发财": "6z", "红中": "7z",
    # 别名
    "1万": "1m", "2万": "2m", "3万": "3m",
    "4万": "4m", "5万": "5m", "6万": "6m",
    "7万": "7m", "8万": "8m", "9万": "9m",
    "1筒": "1p", "2筒": "2p", "3筒": "3p",
    "4筒": "4p", "5筒": "5p", "6筒": "6p",
    "7筒": "7p", "8筒": "8p", "9筒": "9p",
    "1条": "1s", "2条": "2s", "3条": "3s",
    "4条": "4s", "5条": "5s", "6条": "6s",
    "7条": "7s", "8条": "8s", "9条": "9s",
    "东风": "1z", "南风": "2z", "西风": "3z", "北风": "4z",
    "白": "5z", "发": "6z", "中": "7z",
}


def chinese_to_compact(chinese_input):
    """将中文输入转换为紧凑编码"""
    # 先处理多字匹配（最长匹配
    codes = []
    i = 0
    s = chinese_input.replace(" ", "").replace("　", "")

    while i < len(s):
        # 尝试匹配最长的 (2个字符)
        matched = False
        for length in [2, 1]:
            if i + length <= len(s):
                substr = s[i:i+length]
                if substr in NAME_TO_CODE:
                    codes.append(NAME_TO_CODE[substr])
                    i += length
                    matched = True
                    break
        if not matched:
            i += 1

    # 整理格式: 按 m, p, s, z 分组
    groups = {"m": [], "p": [], "s": [], "z": []}
    for code in codes:
        num = code[0]
        suit = code[1]
        if suit in groups:
            groups[suit].append(num)

    # 排序并组合
    result = ""
    for suit in ["m", "p", "s", "z"]:
        if groups[suit]:
            # 数字排序
            nums = sorted(groups[suit], key=lambda x: int(x) if x.isdigit() else x)
            result += "".join(nums) + suit

    return result, len(codes)


def load_calibration():
    try:
        with open("calibration_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg["hand"]
    except Exception:
        return None


def main():
    print("=" * 60)
    print("中文输入版本")
    print("=" * 60)
    print("\n输入示例：")
    print("  一万二万三万 一条二条三条 红中")
    print("  东风南风 白板发财")
    print("\n操作说明：")
    print("  1. 在终端直接输入中文牌名")
    print("  2. 按 Enter 提交")
    print("  3. 输入 q 退出\n")

    # 加载标定
    hand_region = load_calibration()

    # 初始化相机
    config = VirtualCameraConfig(camera_index=1, auto_detect=False)
    vc = VirtualCameraCapture(config)
    if not vc.start():
        print("❌ 虚拟相机启动失败！")
        return
    print("✅ 虚拟相机启动成功！")
    print("请输入手牌:\n")

    running = True
    last_result = None
    last_input = ""
    lock = threading.Lock()

    # 输入线程
    def input_thread():
        nonlocal last_result, last_input, running
        while running:
            try:
                line = input("> ").strip()
                if line.lower() == 'q':
                    running = False
                    break
                if line:
                    compact, count = chinese_to_compact(line)
                    print(f"🔄 转换: {compact} (共 {count} 张)")

                    if count != 14:
                        print(f"⚠️  注意: 应该 14 张牌, 现在 {count} 张")

                    if compact:
                        try:
                            res = evaluate_hand(compact)
                            with lock:
                                last_result = res
                                last_input = compact
                            if res.options:
                                best = res.options[0]
                                print(f"✅ 最佳切牌: {best.discard} (分数: {best.total_score:.1f})")
                        except Exception as e:
                            print(f"❌ 分析失败: {e}")
            except EOFError:
                running = False
                break

    threading.Thread(target=input_thread, daemon=True).start()

    try:
        while running:
            frame = vc.capture_frame()
            if frame is None:
                cv2.waitKey(30)
                continue

            display = frame.copy()
            h, w = display.shape[:2]

            # 画标定区域
            if hand_region:
                x1 = int(hand_region["left"] * w)
                y1 = int(hand_region["top"] * h)
                x2 = int((hand_region["left"] + hand_region["width"]) * w)
                y2 = int((hand_region["top"] + hand_region["height"]) * h)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 画结果
            with lock:
                res = last_result
                inp = last_input

            if res and inp:
                y_off = 30
                cv2.putText(
                    display,
                    f"Hand: {inp}",
                    (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
                y_off += 35

                if res.options:
                    best = res.options[0]
                    cv2.putText(
                        display,
                        f"Discard: {best.discard}",
                        (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )

            cv2.imshow("Mahjong AI - Chinese", display)

            if cv2.getWindowProperty("Mahjong AI - Chinese", cv2.WND_PROP_VISIBLE) < 1:
                running = False
                break

            cv2.waitKey(10)

    except KeyboardInterrupt:
        running = False
    finally:
        running = False
        vc.stop()
        cv2.destroyAllWindows()
        print("\n✅ 已停止")


if __name__ == "__main__":
    main()
