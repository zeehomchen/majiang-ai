"""
简易交互版本
- 显示OBS画面和标定区域
- 支持手动输入手牌来测试算牌
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
from majiang_ai.virtual_camera import VirtualCameraCapture, VirtualCameraConfig
from majiang_ai import evaluate_hand


def load_calibration():
    try:
        with open("calibration_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg["hand"]
    except Exception:
        return None


def main():
    print("=" * 60)
    print("简易交互测试")
    print("=" * 60)
    print("\n操作说明：")
    print("  1. 在画面中查看标定区域")
    print("  2. 按 'i' 键输入手牌 (格式: 123m456p789s11222z)")
    print("  3. 按 'q' 退出\n")

    # 加载标定
    hand_region = load_calibration()
    if hand_region:
        print(f"✅ 已加载标定区域:")
        print(f"   位置: ({hand_region['left']:.3f}, {hand_region['top']:.3f})")
        print(f"   尺寸: {hand_region['width']:.3f} x {hand_region['height']:.3f}\n")

    # 初始化相机
    config = VirtualCameraConfig(camera_index=1, auto_detect=False)
    vc = VirtualCameraCapture(config)
    if not vc.start():
        print("❌ 虚拟相机启动失败！")
        return
    print("✅ 虚拟相机启动成功！\n")

    last_result = None
    input_mode = False
    input_buffer = ""

    try:
        while True:
            frame = vc.capture_frame()
            if frame is None:
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
                cv2.putText(
                    display,
                    "Hand Region",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # 画算牌结果
            if last_result:
                y_offset = 30
                cv2.putText(
                    display,
                    f"Hand: {last_result.hand}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
                y_offset += 30

                if last_result.options:
                    best = last_result.options[0]
                    cv2.putText(
                        display,
                        f"Best Discard: {best.discard} (score: {best.total_score:.1f})",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

            # 输入模式提示
            if input_mode:
                cv2.putText(
                    display,
                    f"Input: {input_buffer}_",
                    (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

            cv2.imshow("Mahjong AI - Simple Mode", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n👋 退出")
                break

            elif key == ord('i'):
                input_mode = not input_mode
                if input_mode:
                    input_buffer = ""
                    print("\n输入手牌 (格式: 123m456p789s11222z), 按 Enter 确认")
                    print("输入中...: ", end="", flush=True)

            elif input_mode:
                if key == 13:  # Enter
                    input_mode = False
                    print(input_buffer)
                    if input_buffer.strip():
                        try:
                            last_result = evaluate_hand(input_buffer.strip())
                            print(f"\n✅ 分析完成！")
                            if last_result.options:
                                best = last_result.options[0]
                                print(f"最佳切牌: {best.discard}")
                        except Exception as e:
                            print(f"\n❌ 分析失败: {e}")
                            last_result = None
                elif key == 8:  # Backspace
                    input_buffer = input_buffer[:-1]
                elif 32 <= key <= 126:
                    input_buffer += chr(key)

    except KeyboardInterrupt:
        print("\n👋 收到停止信号")
    finally:
        vc.stop()
        cv2.destroyAllWindows()
        print("\n✅ 已停止")


if __name__ == "__main__":
    main()
