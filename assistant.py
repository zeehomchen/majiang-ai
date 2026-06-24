"""
广东推倒胡算牌器 - 主程序
整合所有功能模块
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List

# 添加项目路径
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from majiang_ai.capture import Capturer, CaptureMode, CaptureConfig
from majiang_ai.virtual_camera import VirtualCameraCapture, VirtualCameraConfig
from majiang_ai.dynamic_ui import DynamicUIRecognizer, DynamicUIAnalysis
from majiang_ai.enhanced_evaluator import evaluate_hand_enhanced, EnhancedMahjongEvaluator
from majiang_ai.overlay import (
    create_overlay,
    MahjongOverlay,
    OverlayConfig,
    TileHighlight,
    ActionHint,
    GlobalStatus,
)
from majiang_ai.yolo_detector import YOLO11Detector


class MajiangAssistant:
    """
    麻将助手主类
    """

    def __init__(
        self,
        use_virtual_camera: bool = False,
        use_yolo: bool = False,
        yolo_model_path: Optional[str] = None,
        use_overlay: bool = True,
        debug: bool = False,
    ):
        self.use_virtual_camera = use_virtual_camera
        self.use_yolo = use_yolo
        self.yolo_model_path = yolo_model_path
        self.use_overlay = use_overlay
        self.debug = debug

        # 初始化组件
        self._capturer: Optional[Capturer] = None
        self._virtual_camera: Optional[VirtualCameraCapture] = None
        self._ui_recognizer: Optional[DynamicUIRecognizer] = None
        self._yolo_detector: Optional[YOLO11Detector] = None
        self._overlay: Optional[MahjongOverlay] = None
        self._evaluator: Optional[EnhancedMahjongEvaluator] = None

        self._running: bool = False

    def initialize(self) -> bool:
        """
        初始化所有组件

        Returns:
            是否成功
        """
        print("=" * 50)
        print("广东推倒胡算牌器 - 初始化")
        print("=" * 50)

        try:
            # 初始化评估器
            self._evaluator = EnhancedMahjongEvaluator(
                speed_priority=0.6,
                safety_priority=0.3,
                fan_priority=0.1
            )
            print("✅ 评估器初始化成功")

            # 初始化UI识别器
            self._ui_recognizer = DynamicUIRecognizer(debug=self.debug)
            print("✅ UI识别器初始化成功")

            # 初始化YOLO检测器
            if self.use_yolo and self.yolo_model_path:
                self._yolo_detector = YOLO11Detector(
                    model_path=self.yolo_model_path,
                    conf_threshold=0.5,
                    use_onnx=True,
                    debug=self.debug
                )
                if self._yolo_detector.load_model():
                    print("✅ YOLO检测器初始化成功")

            # 初始化采集器
            if self.use_virtual_camera:
                print("🔄 使用虚拟相机模式")
                self._virtual_camera = VirtualCameraCapture()
                if not self._virtual_camera.start():
                    print("❌ 虚拟相机启动失败")
                    return False
            else:
                print("🔄 使用窗口采集模式")
                capture_mode = CaptureMode.PRINT_WINDOW
                self._capturer = Capturer(mode=capture_mode, debug=self.debug)
                if not self._capturer.start():
                    print("❌ 采集器启动失败")
                    return False

            # 初始化悬浮窗
            if self.use_overlay:
                overlay_config = OverlayConfig(
                    anti_screenshot=True,
                    auto_follow=True,
                    click_through=True,
                )
                self._overlay = create_overlay(overlay_config)
                if not self._overlay.initialize():
                    print("⚠️ 悬浮窗初始化失败，继续运行")
                    self.use_overlay = False
                else:
                    print("✅ 悬浮窗初始化成功")

            print("\n🎉 初始化完成！")
            return True

        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False

    def run(self) -> None:
        """运行主循环"""
        if not self.initialize():
            return

        self._running = True
        print("\n开始运行... (按 Ctrl+C 停止)")
        print("-" * 50)

        try:
            if self.use_overlay and self._overlay:
                # 使用悬浮窗模式
                self._overlay.run()
            else:
                # 命令行模式
                self._run_cli_mode()

        except KeyboardInterrupt:
            print("\n\n收到停止信号")
        finally:
            self.stop()

    def _run_cli_mode(self) -> None:
        """命令行模式运行"""
        frame_count = 0
        last_time = time.time()

        while self._running:
            try:
                # 捕获画面
                if self.use_virtual_camera and self._virtual_camera:
                    frame = self._virtual_camera.capture_frame()
                elif self._capturer:
                    frame = self._capturer.capture_region(0, 0, 1, 1)
                else:
                    time.sleep(0.1)
                    continue

                if frame is None:
                    time.sleep(0.1)
                    continue

                # 处理画面
                self._process_frame(frame)

                # 统计FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - last_time)
                    print(f"FPS: {fps:.1f}")
                    last_time = current_time

                time.sleep(0.05)

            except Exception as e:
                print(f"处理错误: {e}")
                time.sleep(0.5)

    def _process_frame(self, frame) -> None:
        """
        处理单帧画面

        Args:
            frame: 图像帧
        """
        # 1. 使用UI识别器分析
        if self._ui_recognizer:
            ui_result = self._ui_recognizer.analyze_bottom_region(frame)
            
            # 2. 如果有YOLO，使用YOLO检测
            if self._yolo_detector:
                yolo_result = self._yolo_detector.predict(frame)
                # 合并结果...

            # 3. 评估手牌 (这里需要将识别结果转化为牌编码)
            # 目前占位，需要完整实现图像识别
            hand_tiles = []  # 需要从图像识别得到
            if hand_tiles and len(hand_tiles) == 14:
                hand_str = "".join(hand_tiles)
                result = evaluate_hand_enhanced(hand_str)
                
                # 4. 显示结果
                self._display_result(result)

    def _display_result(self, result) -> None:
        """
        显示结果

        Args:
            result: 评估结果
        """
        if not result.options:
            return

        best_option = result.options[0]
        
        print(f"\r手牌: {result.hand} | 切牌: {best_option.discard} | 分数: {best_option.total_score:.1f}", end="")

    def stop(self) -> None:
        """停止"""
        self._running = False

        if self._virtual_camera:
            self._virtual_camera.stop()

        if self._capturer:
            self._capturer.stop()

        if self._overlay:
            self._overlay.stop()

        print("\n👋 已停止")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="广东推倒胡算牌器")

    parser.add_argument(
        "--virtual-camera",
        action="store_true",
        help="使用OBS虚拟相机模式"
    )

    parser.add_argument(
        "--yolo",
        action="store_true",
        help="使用YOLO检测"
    )

    parser.add_argument(
        "--yolo-model",
        type=str,
        default=None,
        help="YOLO模型路径"
    )

    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="不使用悬浮窗"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式"
    )

    args = parser.parse_args()

    assistant = MajiangAssistant(
        use_virtual_camera=args.virtual_camera,
        use_yolo=args.yolo,
        yolo_model_path=args.yolo_model,
        use_overlay=not args.no_overlay,
        debug=args.debug
    )

    assistant.run()


if __name__ == "__main__":
    main()
