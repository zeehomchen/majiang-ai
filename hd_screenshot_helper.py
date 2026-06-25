"""
高清截图助手！
强制设置相机为最高分辨率！
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
    print("高清截图助手！")
    print("=" * 60)
    print("\n正在尝试打开相机并设置最高分辨率...")

    # 打开相机
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 用DirectShow
    if not cap.isOpened():
        print("索引1打不开，试索引0...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("❌ 都打不开！")
            return

    # 设置最高分辨率！
    print("\n尝试设置高分辨率...")
    # 先尝试常用的高分辨率
    common_resolutions = [
        (1920, 1080),
        (1280, 720),
        (1024, 768),
        (800, 600),
    ]

    best_w, best_h = 0, 0
    for w, h in common_resolutions:
        print(f"  尝试 {w}x{h}...")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # 读取一帧验证
        ret, test_frame = cap.read()
        if ret:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"    实际: {actual_w}x{actual_h}")
            if actual_w * actual_h > best_w * best_h:
                best_w, best_h = actual_w, actual_h

    # 确保用最高的
    if best_w > 0 and best_h > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_h)

    # 再查一下实际
    final_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    final_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n✅ 相机分辨率: {final_w}x{final_h}")

    # 提高帧率，关闭自动曝光（可选）
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # 有些相机需要这个

    output_dir = ROOT / "tile_templates"
    output_dir.mkdir(exist_ok=True)

    print("\n操作说明：")
    print("  1. 把目标牌放到屏幕上")
    print("  2. 按 's' 保存高清截图到 'to_crop.png'")
    print("  3. 按 'q' 退出\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cv2.waitKey(30)
                continue

            # 显示缩小的预览（因为原图太大可能显示不全）
            h, w = frame.shape[:2]
            scale = 0.5 if w > 1500 else 1.0
            preview = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            # 画提示
            cv2.putText(
                preview,
                "Press 's' to save, 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow("HD Capture Helper", preview)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存原图！
                save_path = ROOT / "to_crop.png"
                cv2.imwrite(str(save_path), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])  # 无损PNG
                print(f"\n✅ 高清截图已保存: {save_path} ({w}x{h})")
                print("现在用图片工具裁剪出单张牌，重命名为牌面编码放到 tile_templates/ 文件夹！")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ 已停止")


if __name__ == "__main__":
    main()
