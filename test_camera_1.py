"""
OBS虚拟相机测试 - 固定索引1
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
from majiang_ai.virtual_camera import VirtualCameraCapture, VirtualCameraConfig


def main():
    print("=" * 60)
    print("OBS虚拟相机测试 (索引 1)")
    print("=" * 60)
    print("\n按 'q' 键退出，按 's' 键保存截图\n")

    # 使用固定索引1
    config = VirtualCameraConfig(
        camera_index=1,
        auto_detect=False
    )

    vc = VirtualCameraCapture(config)
    if not vc.start():
        print("❌ 虚拟相机启动失败！")
        return

    print("✅ 虚拟相机启动成功！\n")

    try:
        while True:
            frame = vc.capture_frame()
            if frame is None:
                continue

            cv2.imshow("OBS Virtual Camera - Index 1 (press q to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 退出")
                break
            elif key == ord('s'):
                cv2.imwrite("screenshot.png", frame)
                print("📸 截图已保存: screenshot.png")

    except KeyboardInterrupt:
        print("\n👋 收到停止信号")
    finally:
        vc.stop()
        cv2.destroyAllWindows()
        print("✅ 已停止")


if __name__ == "__main__":
    main()
