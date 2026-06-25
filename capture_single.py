"""
单个牌面采集工具
遇到没采集过的牌时，用这个程序单独添加！
"""

from __future__ import annotations

import sys
import json
import os
from pathlib import Path

# 添加项目路径
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2


def load_calibration():
    try:
        with open("calibration_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg["hand"]
    except Exception:
        return None


def main():
    print("=" * 60)
    print("单张牌面采集工具")
    print("=" * 60)

    # 加载标定
    hand_region = load_calibration()
    if not hand_region:
        print("❌ 请先运行 calibrate.py!")
        return

    # 打开相机
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 打不开相机！")
            return

    output_dir = ROOT / "tile_templates"
    output_dir.mkdir(exist_ok=True)

    print("\n操作说明：")
    print("  1. 确保目标牌在屏幕上")
    print("  2. 用鼠标在目标牌上画一个框")
    print("  3. 在终端输入牌面编码（例如：5m, 8p, 1z）")
    print("  4. 按 's' 保存")
    print("  5. 按 'q' 退出\n")

    print("编码说明：")
    print("  1m~9m = 一万~九万")
    print("  1p~9p = 一筒~九筒")
    print("  1s~9s = 一条~九条")
    print("  1z=东, 2z=南, 3z=西, 4z=北, 5z=白, 6z=发, 7z=中\n")

    # 选择框状态
    selecting = False
    start_x, start_y = -1, -1
    end_x, end_y = -1, -1
    temp_clone = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, start_x, start_y, end_x, end_y, temp_clone

        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            start_x, start_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            end_x, end_y = x, y
            # 画临时框
            temp_clone = frame.copy()
            x1, x2 = sorted([start_x, x])
            y1, y2 = sorted([start_y, y])
            cv2.rectangle(temp_clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            end_x, end_y = x, y

    cv2.namedWindow("Single Tile Capture")
    cv2.setMouseCallback("Single Tile Capture", mouse_callback)

    frame = None
    try:
        while True:
            ret, read_frame = cap.read()
            if ret:
                frame = read_frame

            if frame is None:
                cv2.waitKey(30)
                continue

            display = temp_clone if temp_clone is not None and selecting else frame.copy()

            # 画标定区域
            h, w = display.shape[:2]
            x1 = int(hand_region["left"] * w)
            y1 = int(hand_region["top"] * h)
            x2 = int((hand_region["left"] + hand_region["width"]) * w)
            y2 = int((hand_region["top"] + hand_region["height"]) * h)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 1)

            # 显示提示
            cv2.putText(
                display,
                "Draw a box around the tile, then input code in terminal",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.imshow("Single Tile Capture", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n👋 退出")
                break
            elif key == ord('s'):
                if start_x != -1 and end_x != -1 and frame is not None:
                    # 裁剪
                    x1, x2 = sorted([start_x, end_x])
                    y1, y2 = sorted([start_y, end_y])
                    tile_crop = frame[y1:y2, x1:x2].copy()

                    if tile_crop.size == 0:
                        print("❌ 选择的区域为空！")
                        continue

                    # 输入编码
                    code = input("\n请输入牌面编码（例如 5m, 8p, 7z）: ").strip().lower()

                    # 验证
                    import re
                    valid = re.compile(r'^[1-9][mpsz]$')
                    if not valid.match(code):
                        print("❌ 无效编码！格式如 1m, 2p, 7z")
                        continue

                    # 保存
                    save_path = output_dir / f"{code}.png"
                    cv2.imwrite(str(save_path), tile_crop)
                    print(f"✅ 已保存: {save_path}")
                    print("继续采集下一张或按 q 退出！")

                    # 重置
                    start_x, start_y = -1, -1
                    end_x, end_y = -1, -1
                    temp_clone = None

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ 已停止")


if __name__ == "__main__":
    main()
