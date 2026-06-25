"""
简单相机测试脚本 - 手动选择索引
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


def main():
    print("=" * 60)
    print("OBS虚拟相机测试 - 手动选择索引")
    print("=" * 60)
    print("\n请输入你想尝试的相机索引 (0, 1, 2, ...):")

    while True:
        try:
            idx_input = input("\n索引 (输入 q 退出): ").strip().lower()
            if idx_input == 'q':
                break

            idx = int(idx_input)
            print(f"\n尝试索引 {idx}...")

            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                print(f"❌ 无法打开索引 {idx}")
                continue

            print(f"✅ 已打开索引 {idx}!")
            print("查看弹出的窗口，如果是OBS画面，就记住这个索引！")
            print("按 q 关闭窗口，继续尝试其他索引")

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                cv2.imshow(f"Camera Index {idx} (press q to close)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyWindow(f"Camera Index {idx} (press q to close)")
                    break

            cap.release()

        except ValueError:
            print("请输入数字索引或 q 退出")

    print("\n👋 测试完成！")
    print("记住哪个索引对应OBS画面！")


if __name__ == "__main__":
    main()
