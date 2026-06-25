"""
模板采集程序
让你采集自己游戏里的牌面模板
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
from majiang_ai.virtual_camera import VirtualCameraCapture, VirtualCameraConfig
from majiang_ai.vision import segment_hand_strip


def load_calibration():
    try:
        with open("calibration_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg["hand"]
    except Exception:
        return None


def main():
    print("=" * 60)
    print("牌面模板采集程序")
    print("=" * 60)
    print("\n请确保：")
    print("  1. OBS已打开，虚拟相机已启动")
    print("  2. 游戏手牌区域已标定")
    print("  3. 游戏里有完整的手牌（14张）\n")

    # 加载标定
    hand_region = load_calibration()
    if not hand_region:
        print("❌ 未找到标定文件！")
        return

    # 初始化相机
    config = VirtualCameraConfig(camera_index=1, auto_detect=False)
    vc = VirtualCameraCapture(config)
    if not vc.start():
        print("❌ 虚拟相机启动失败！")
        return
    print("✅ 虚拟相机已启动")

    print("\n按 'c' 键采集当前手牌，按 'q' 键退出\n")

    output_dir = ROOT / "tile_templates"
    output_dir.mkdir(exist_ok=True)

    try:
        while True:
            frame = vc.capture_frame()
            if frame is None:
                cv2.waitKey(30)
                continue

            display = frame.copy()
            h, w = display.shape[:2]

            # 画标定区域
            x1 = int(hand_region["left"] * w)
            y1 = int(hand_region["top"] * h)
            x2 = int((hand_region["left"] + hand_region["width"]) * w)
            y2 = int((hand_region["top"] + hand_region["height"]) * h)

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Template Capture - Press 'c' to capture", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                print("\n👋 退出")
                break
            elif key == ord('c'):
                # 采集！
                hand_crop = frame[y1:y2, x1:x2].copy()
                if hand_crop.size == 0:
                    print("❌ 手牌区域为空！")
                    continue

                # 分割牌
                crops = segment_hand_strip(hand_crop, expected_count=14)
                print(f"\n✅ 已分割出 {len(crops)} 张牌！")

                # 保存预览
                for i, crop in enumerate(crops):
                    preview_path = output_dir / f"slot_{i+1:02d}.png"
                    cv2.imwrite(str(preview_path), crop)
                    print(f"  保存: {preview_path}")

                print(f"\n请查看 {output_dir} 目录，确认图片后，")
                print("按顺序把 slot_01.png ~ slot_14.png 重命名为牌面编码！")
                print("例如: 1m.png, 2p.png, 7z.png 这样！")
                print("\n编码说明:")
                print("  1m~9m = 一万~九万")
                print("  1p~9p = 一筒~九筒")
                print("  1s~9s = 一条~九条")
                print("  1z=东, 2z=南, 3z=西, 4z=北, 5z=白, 6z=发, 7z=中\n")

    except KeyboardInterrupt:
        pass
    finally:
        vc.stop()
        cv2.destroyAllWindows()
        print("\n✅ 已停止")


if __name__ == "__main__":
    main()
