"""
简单测试脚本 - 测试核心功能
"""

from pathlib import Path
import sys

# 添加src目录到路径
root = Path(__file__).parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

print("=" * 60)
print("测试1: 基础算牌功能")
print("=" * 60)
try:
    from majiang_ai import evaluate_hand
    from majiang_ai import evaluate_hand_enhanced
    
    print("✅ evaluate_hand 导入成功")
    
    # 测试基本功能 - 14张手牌（摸牌后
    result = evaluate_hand("123m456p789s11222z")
    print(f"✅ 手牌评估成功")
    print(f"   手牌数量: {result.tile_count}")
    if result.options:
        print(f"   最佳切牌: {result.options[0].discard}")
        print(f"   分数: {result.options[0].total_score:.1f}")
    
    # 测试增强版
    result2 = evaluate_hand_enhanced("123m456p789s11222z")
    print(f"✅ 增强版评估成功")
    if result2.options:
        print(f"   最佳切牌: {result2.options[0].discard}")
        print(f"   分数: {result2.options[0].total_score:.1f}")
    
except Exception as e:
    print(f"❌ 算牌功能测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试2: 虚拟相机模块")
print("=" * 60)
try:
    from majiang_ai.virtual_camera import VirtualCameraCapture
    print("✅ virtual_camera 模块导入成功")
except Exception as e:
    print(f"❌ 虚拟相机模块导入失败: {e}")

print("\n" + "=" * 60)
print("测试3: 动态UI识别模块")
print("=" * 60)
try:
    from majiang_ai.dynamic_ui import DynamicUIRecognizer
    print("✅ dynamic_ui 模块导入成功")
except Exception as e:
    print(f"❌ 动态UI模块导入失败: {e}")

print("\n" + "=" * 60)
print("测试4: YOLO检测模块")
print("=" * 60)
try:
    from majiang_ai.yolo_detector import YOLO11Detector
    print("✅ yolo_detector 模块导入成功")
except Exception as e:
    print(f"❌ YOLO模块导入失败: {e}")

print("\n" + "=" * 60)
print("测试5: 采集模块")
print("=" * 60)
try:
    from majiang_ai.capture import Capturer, CaptureMode
    print("✅ capture 模块导入成功")
    print(f"   可用模式: {[m.value for m in CaptureMode]}")
except Exception as e:
    print(f"❌ 采集模块导入失败: {e}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)