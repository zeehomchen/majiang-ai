"""Guangdong Tuidaohu prototype engine.

Provides:
- evaluator : hand evaluation and optimal discard
- parser    : tile string encoding/decoding
- capture   : DirectX screen capture (dxcam)
- vision    : tile image recognition
- cli       : command-line interface
"""

from .evaluator import evaluate_hand
from .parser import parse_compact_tiles, parse_tile_list, tiles_to_compact

__all__ = [
    "evaluate_hand",
    "parse_compact_tiles",
    "parse_tile_list",
    "tiles_to_compact",
]
