"""
列出所有可用的相机索引
帮你找到OBS虚拟相机对应的索引
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
from majiang_ai.virtual_camera import VirtualCameraCapture


def main():
    print("=" * 60)
    print("查找可用的摄像头")
    print("=" * 60)

    # 列出所有相机
    cameras = VirtualCameraCapture.list_cameras()

    if not cameras:
        print("\n❌ 未找到任何可用的摄像头！")
        print("\n请检查：")
        print("  1. OBS是否已打开")
        print("  2. OBS虚拟相机是否已启动")
        print("  3. 是否有其他摄像头被占用")
        return

    print(f"\n✅ 找到 {len(cameras)} 个可用的摄像头:")
    for idx in cameras:
        print(f"  索引 {idx}")

    print("\n" + "=" * 60)
    print("尝试逐个打开，看看哪个是OBS虚拟相机")
    print("=" * 60)

    for idx in cameras:
        print(f"\n尝试索引 {idx}...")

        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"  ❌ 无法打开索引 {idx}")
            continue

        # 尝试读取几帧
        success = False
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                success = True
                print(f"  ✅ 成功打开索引 {idx}")
                print(f"  画面尺寸: {frame.shape[1]}x{frame.shape[0]}")

                # 显示一下
                cv2.imshow(f"Camera {idx} (press q to quit)", frame)
                print(f"\n查看窗口 'Camera {idx}'")
                print("按 'q' 关闭这个窗口，继续看下一个")

                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cv2.destroyWindow(f"Camera {idx}")
                        break
                break

        cap.release()

    print("\n" + "=" * 60)
    print("测试完成！")
    print("记住哪个索引对应OBS虚拟相机！")


if __name__ == "__main__":
    main()
