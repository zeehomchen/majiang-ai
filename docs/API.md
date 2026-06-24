# API文档

## 快速调用示例

### 评估手牌

```python
from majiang_ai import evaluate_hand_enhanced

result = evaluate_hand_enhanced(
    hand="123m456p789s1122z",
    visible_tiles="1m2m3m",
    round_phase="mid"
)

# 查看最优切牌
best = result.options[0]
print(f"切牌: {best.discard}")
print(f"分数: {best.total_score}")
```

---

## 主要模块

### 1. enhanced_evaluator - 增强版评估器

```python
from majiang_ai.enhanced_evaluator import (
    EnhancedMahjongEvaluator,
    evaluate_hand_enhanced
)

# 创建自定义评估器
evaluator = EnhancedMahjongEvaluator(
    speed_priority=0.7,
    safety_priority=0.2,
    fan_priority=0.1
)

# 评估
result = evaluator.evaluate(
    hand="123m456p789s11z",
    round_phase="late"
)
```

### 2. capture - 画面采集

```python
from majiang_ai import Capturer, CaptureMode

# 使用PrintWindow模式
cap = Capturer(mode=CaptureMode.PRINT_WINDOW)
cap.start()

# 采集手牌区域
hand_frame = cap.capture_hand()

# 采集指定区域
frame = cap.capture_region(0, 0, 1, 1)

cap.stop()
```

### 3. virtual_camera - OBS虚拟相机

```python
from majiang_ai.virtual_camera import (
    VirtualCameraCapture,
    VirtualCameraConfig
)

vc = VirtualCameraCapture()
vc.start()

# 采集一帧
frame = vc.capture_frame()

# 连续采集
for frame in vc.capture_continuous():
    # 处理 frame
    pass

vc.stop()
```

### 4. dynamic_ui - 动态UI识别

```python
from majiang_ai.dynamic_ui import DynamicUIRecognizer

recognizer = DynamicUIRecognizer()
result = recognizer.analyze_bottom_region(image)

# 手牌列表
print(result.hand_tiles)
# 副露列表
print(result.meld_tiles)
# 分割点
print(result.split_x)
```

### 5. yolo_detector - YOLO检测

```python
from majiang_ai.yolo_detector import YOLO11Detector

detector = YOLO11Detector(model_path="./best.onnx")
detector.load_model()

result = detector.predict(image)

for tile in result.tiles:
    print(f"牌: {tile.tile_code}")
    print(f"位置: {tile.bbox}")
    print(f"状态: {tile.tile_state}")
```

### 6. overlay - 悬浮窗

```python
from majiang_ai import (
    create_overlay,
    TileHighlight,
    GlobalStatus
)

# 创建悬浮窗
overlay = create_overlay()
overlay.initialize()

# 更新高亮
highlights = [
    TileHighlight(
        bbox=(100, 200, 200, 300),
        is_best=True,
        score=85.5,
        tile_code="3m"
    )
]
overlay.update_tile_highlights(highlights)

# 更新状态
status = GlobalStatus(
    shanten=0,
    waiting_tiles=["2m", "5m"],
    danger_tiles=["1p"]
)
overlay.update_global_status(status)

# 运行
overlay.run()
```

---

## 数据结构

### AnalysisResult

评估结果对象：

```python
result.hand              # 手牌编码
result.tile_count        # 牌数量
result.options           # 切牌选项列表
result.visible_counts    # 可见牌计数
result.exposed_triplets  # 暴露的刻子
result.pre_draw          # 摸牌前分析（13张时）
```

### DiscardOption

单个切牌选项：

```python
option.discard           # 切牌编码
option.total_score       # 总分
option.base_score        # 基础分
option.effective_draws   # 有效进张数
option.improving_tiles   # 改良牌列表
option.route_scores      # 番型路线分数
option.summary           # 摘要
```

---

## 完整示例

### 从虚拟相机到决策完整流程

```python
import cv2
from majiang_ai.virtual_camera import VirtualCameraCapture
from majiang_ai.dynamic_ui import DynamicUIRecognizer
from majiang_ai import evaluate_hand_enhanced

# 1. 初始化
vc = VirtualCameraCapture()
vc.start()

recognizer = DynamicUIRecognizer()

try:
    while True:
        # 2. 采集
        frame = vc.capture_frame()
        if frame is None:
            continue
        
        # 3. 识别（这里需要完整的识别管道）
        ui_result = recognizer.analyze_bottom_region(frame)
        
        # 4. 假设我们已经得到了手牌编码
        hand_str = "123m456p789s1122z"
        
        # 5. 评估
        result = evaluate_hand_enhanced(hand_str)
        
        # 6. 输出
        best = result.options[0]
        print(f"最优切牌: {best.discard}")
        
        # 显示画面
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    vc.stop()
    cv2.destroyAllWindows()
```
