"""
自动图像识别版本！
自动从标定区域识别手牌，不需要手动输入！
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
from majiang_ai.vision import recognize_tiles, load_templates


def load_calibration():
    try:
        with open("calibration_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg["hand"]
    except Exception:
        return None


def main():
    print("=" * 60)
    print("自动图像识别版本")
    print("=" * 60)
    print("\n加载中...")

    # 加载模板
    template_dir = ROOT / "tile_templates"
    templates = load_templates(str(template_dir))
    print(f"✅ 已加载 {len(templates)} 张牌面模板")

    # 加载标定
    hand_region = load_calibration()
    if not hand_region:
        print("\n❌ 未找到标定文件 calibration_config.json")
        print("请先运行 calibrate.py 标定手牌区域！")
        return
    print(f"✅ 已加载标定区域\n")

    # 初始化相机
    config = VirtualCameraConfig(camera_index=1, auto_detect=False)
    vc = VirtualCameraCapture(config)
    if not vc.start():
        print("❌ 虚拟相机启动失败！")
        return
    print("✅ 虚拟相机启动成功！")
    print("\n按 'q' 退出, 按 's' 保存截图\n")

    last_result = None
    last_recognized = None
    last_frame = None

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

            # 截取手牌区域
            hand_crop = frame[y1:y2, x1:x2].copy()

            # 图像识别
            if hand_crop.size > 0:
                vis_result = recognize_tiles(hand_crop, templates=templates)

                if vis_result.hand_compact:
                    # 识别成功，算牌
                    try:
                        eval_result = evaluate_hand(vis_result.hand_compact)
                        last_recognized = vis_result.hand_compact
                        last_result = eval_result
                        last_frame = hand_crop
                    except Exception:
                        pass

            # 画识别结果
            y_off = 30
            if last_recognized:
                cv2.putText(
                    display,
                    f"Recognized: {last_recognized}",
                    (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
                y_off += 35

            if last_result and last_result.options:
                best = last_result.options[0]
                cv2.putText(
                    display,
                    f"Discard: {best.discard} (score: {best.total_score:.1f})",
                    (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    3
                )
                y_off += 40

            # 显示识别的手牌图在左上角
            if last_frame is not None:
                scale = 0.5
                small = cv2.resize(last_frame, (0, 0), fx=scale, fy=scale)
                sh, sw = small.shape[:2]
                display[10:10+sh, 10:10+sw] = small

            cv2.imshow("Mahjong AI - Auto Recognition", display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                print("\n👋 退出")
                break
            elif key == ord('s'):
                cv2.imwrite("auto_recognition.png", display)
                print("📸 截图已保存: auto_recognition.png")

    except KeyboardInterrupt:
        pass
    finally:
        vc.stop()
        cv2.destroyAllWindows()
        print("\n✅ 已停止")


if __name__ == "__main__":
    main()
