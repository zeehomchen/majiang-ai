"""
简化模板采集，带调试
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
    except Exception as e:
        print(f"加载标定失败: {e}")
        return None


def main():
    print("=" * 60)
    print("简化模板采集 - 调试版")
    print("=" * 60)

    # 加载标定
    hand_region = load_calibration()
    if not hand_region:
        print("\n❌ 请先运行 calibrate.py 标定区域！")
        return

    print(f"标定区域: {hand_region}\n")

    # 直接OpenCV打开相机，不依赖类
    print("尝试打开相机索引 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ 打不开相机！尝试索引 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 也打不开索引 0！")
            return

    print("✅ 相机已打开！\n")
    print("按 's' 键保存截图到 capture.png\n按 'q' 退出\n")

    output_dir = ROOT / "tile_templates"
    output_dir.mkdir(exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  读不到帧！")
                cv2.waitKey(100)
                continue

            display = frame.copy()
            h, w = display.shape[:2]

            # 画标定区域
            x1 = int(hand_region["left"] * w)
            y1 = int(hand_region["top"] * h)
            x2 = int((hand_region["left"] + hand_region["width"]) * w)
            y2 = int((hand_region["top"] + hand_region["height"]) * h)

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # 显示一些信息
            cv2.putText(
                display,
                f"Press 's' to save, 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.imshow("Simple Capture", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                print("\n👋 退出")
                break
            elif key == ord('s'):
                # 保存完整截图
                full_path = output_dir / "full_screenshot.png"
                cv2.imwrite(str(full_path), frame)
                print(f"✅ 完整截图已保存: {full_path}")

                # 保存手牌区域
                hand_crop = frame[y1:y2, x1:x2].copy()
                if hand_crop.size > 0:
                    hand_path = output_dir / "hand_region.png"
                    cv2.imwrite(str(hand_path), hand_crop)
                    print(f"✅ 手牌区域已保存: {hand_path}")
                    print("现在请查看这两个文件！")
                else:
                    print("❌ 手牌区域为空！")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ 已停止")


if __name__ == "__main__":
    main()
