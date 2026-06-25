"""
区域标定工具 - 让你手动选择手牌区域
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目路径
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import json
from majiang_ai.virtual_camera import VirtualCameraCapture, VirtualCameraConfig


# 鼠标回调
drawing = False
start_x, start_y = -1, -1
end_x, end_y = -1, -1


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_x, end_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y


def main():
    global start_x, start_y, end_x, end_y

    print("=" * 60)
    print("区域标定工具")
    print("=" * 60)
    print("\n操作说明：")
    print("  1. 在画面上用鼠标拖动画框，选择手牌区域")
    print("  2. 按 's' 保存区域配置")
    print("  3. 按 'q' 退出\n")

    # 初始化相机
    config = VirtualCameraConfig(camera_index=1, auto_detect=False)
    vc = VirtualCameraCapture(config)
    if not vc.start():
        print("❌ 虚拟相机启动失败！")
        return
    print("✅ 虚拟相机启动成功！")

    # 窗口和鼠标回调
    window_name = "Calibration - Select Hand Region"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        while True:
            frame = vc.capture_frame()
            if frame is None:
                continue

            display = frame.copy()

            # 画当前选择
            if start_x != -1 and end_x != -1:
                cv2.rectangle(
                    display,
                    (start_x, start_y),
                    (end_x, end_y),
                    (0, 255, 0),
                    2
                )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 退出")
                break
            elif key == ord('s'):
                if start_x != -1 and end_x != -1:
                    # 保存配置
                    h, w = frame.shape[:2]
                    cfg = {
                        "hand": {
                            "left": min(start_x, end_x) / w,
                            "top": min(start_y, end_y) / h,
                            "width": abs(end_x - start_x) / w,
                            "height": abs(end_y - start_y) / h
                        }
                    }

                    with open("calibration_config.json", "w", encoding="utf-8") as f:
                        json.dump(cfg, f, indent=2, ensure_ascii=False)

                    print(f"\n✅ 配置已保存: calibration_config.json")
                    print(f"区域: ({cfg['hand']['left']:.3f}, {cfg['hand']['top']:.3f})")
                    print(f"尺寸: {cfg['hand']['width']:.3f} x {cfg['hand']['height']:.3f}")

    except KeyboardInterrupt:
        print("\n👋 收到停止信号")
    finally:
        vc.stop()
        cv2.destroyAllWindows()
        print("\n✅ 已停止")


if __name__ == "__main__":
    main()
