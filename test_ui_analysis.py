"""
整合测试 - 虚拟相机 + 动态UI识别
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
from majiang_ai.dynamic_ui import DynamicUIRecognizer


def main():
    print("=" * 60)
    print("虚拟相机 + 动态UI识别 测试")
    print("=" * 60)
    print("\n按 'q' 键退出，按 's' 键保存截图\n")

    # 初始化相机 (索引 1)
    config = VirtualCameraConfig(
        camera_index=1,
        auto_detect=False
    )
    vc = VirtualCameraCapture(config)
    if not vc.start():
        print("❌ 虚拟相机启动失败！")
        return
    print("✅ 虚拟相机启动成功！")

    # 初始化UI识别器
    ui_recognizer = DynamicUIRecognizer(debug=True)
    print("✅ UI识别器初始化成功！\n")

    try:
        while True:
            frame = vc.capture_frame()
            if frame is None:
                continue

            # 做一个副本，用于绘制
            display = frame.copy()

            # 分析底部区域
            ui_result = ui_recognizer.analyze_bottom_region(frame)

            # 绘制分析结果
            display = ui_recognizer.visualize_analysis(display, ui_result)

            # 显示
            cv2.imshow("Dynamic UI Analysis (press q to quit)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 退出")
                break
            elif key == ord('s'):
                cv2.imwrite("ui_analysis.png", display)
                print("📸 分析截图已保存: ui_analysis.png")

    except KeyboardInterrupt:
        print("\n👋 收到停止信号")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        vc.stop()
        cv2.destroyAllWindows()
        print("✅ 已停止")


if __name__ == "__main__":
    main()
