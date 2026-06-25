"""
简单启动脚本 - 先测试虚拟相机采集
不需要悬浮窗，只测试采集功能
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
    print("OBS虚拟相机测试")
    print("=" * 60)
    print("\n请确保：")
    print("  1. OBS已打开")
    print("  2. 游戏源已添加")
    print("  3. 虚拟相机已启动\n")

    # 初始化虚拟相机
    vc = VirtualCameraCapture()
    if not vc.start():
        print("❌ 虚拟相机启动失败！")
        print("\n请检查：")
        print("  - OBS虚拟相机是否已启动")
        print("  - 相机索引是否正确")
        return

    print("✅ 虚拟相机启动成功！")
    print("按 'q' 键退出，按 's' 键保存截图\n")

    try:
        # 显示画面
        while True:
            frame = vc.capture_frame()
            if frame is None:
                print("⚠️  未能获取画面")
                continue

            # 显示
            cv2.imshow("OBS Virtual Camera (press q to quit)", frame)

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
