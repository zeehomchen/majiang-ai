"""
最简单的采集方案！
不需要画框回调！
1. 保存截图
2. 用你喜欢的图片工具（画图，PS等）裁剪
3. 重命名保存到 tile_templates 文件夹！
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


def load_calibration():
    try:
        with open("calibration_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg["hand"]
    except Exception:
        return None


def main():
    print("=" * 60)
    print("最简单采集方案！")
    print("=" * 60)
    print("\n操作方法：")
    print("  1. 程序显示画面")
    print("  2. 当目标牌在屏幕上时，按 's' 保存截图到 'to_crop.png'")
    print("  3. 按 'q' 退出程序")
    print("  4. 用你喜欢的工具（画图、Photoshop等）打开 to_crop.png")
    print("  5. 裁剪出目标牌，保存到 tile_templates/ 文件夹")
    print("  6. 文件名为牌面编码，例如：5m.png, 8p.png, 7z.png\n")

    # 打开相机
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 打不开相机！")
            return

    output_dir = ROOT / "tile_templates"
    output_dir.mkdir(exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cv2.waitKey(30)
                continue

            # 画提示
            display = frame.copy()
            cv2.putText(
                display,
                "Press 's' to save screenshot, 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )
            cv2.imshow("Screenshot Helper", display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = ROOT / "to_crop.png"
                cv2.imwrite(str(save_path), frame)
                print(f"\n✅ 截图已保存: {save_path}")
                print("现在按 'q' 退出，去裁剪图片！")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ 已停止")


if __name__ == "__main__":
    main()
