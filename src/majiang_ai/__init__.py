"""Guangdong Tuidaohu prototype engine.

Provides:
- evaluator : hand evaluation and optimal discard
- enhanced_evaluator : EV model + speed first evaluation
- parser    : tile string encoding/decoding
- capture   : Multi-mode screen capture
- virtual_camera : OBS virtual camera support
- vision    : tile image recognition
- dynamic_ui : dynamic UI recognition
- yolo_detector : YOLO11 object detection
- overlay : desktop overlay window
- simulator : full game simulation engine
- mcts      : Monte Carlo Tree Search optimizer
- cli       : command-line interface
"""

from .evaluator import evaluate_hand
from .enhanced_evaluator import evaluate_hand_enhanced, EnhancedMahjongEvaluator
from .parser import parse_compact_tiles, parse_tile_list, tiles_to_compact
from .capture import Capturer, CaptureMode, CaptureConfig

# 暂时注释掉 overlay 相关导入，避免没有 PyQt5 时报错
# 如需使用悬浮窗功能，请手动安装 PyQt5 后取消注释
# try:
#     from .overlay import create_overlay, MahjongOverlay, OverlayConfig
#     _has_overlay = True
# except ImportError:
#     _has_overlay = False
#     create_overlay = None
#     MahjongOverlay = None
#     OverlayConfig = None

__all__ = [
    "evaluate_hand",
    "evaluate_hand_enhanced",
    "EnhancedMahjongEvaluator",
    "parse_compact_tiles",
    "parse_tile_list",
    "tiles_to_compact",
    "Capturer",
    "CaptureMode",
    "CaptureConfig",
]

# 只在有overlay时添加到__all__
# if _has_overlay:
#     __all__.extend([
#         "create_overlay",
#         "MahjongOverlay",
#         "OverlayConfig",
#     ])
