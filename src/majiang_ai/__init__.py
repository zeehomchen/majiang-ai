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
from .overlay import create_overlay, MahjongOverlay, OverlayConfig

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
    "create_overlay",
    "MahjongOverlay",
    "OverlayConfig",
]
