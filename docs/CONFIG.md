# 配置文件详解

## 配置位置

程序运行时会自动生成一些配置文件，也可以手动创建。

---

## 采集配置 (CaptureConfig)

在 `capture.py` 中定义，也可以通过 `calibration_config.json` 加载：

```json
{
  "hand": {
    "left": 0.05,
    "top": 0.4,
    "width": 0.9,
    "height": 0.16
  }
}
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| hand_left | 手牌区域左边界比例 | 0.05 |
| hand_top | 手牌区域上边界比例 | 0.4 |
| hand_width | 手牌区域宽度比例 | 0.9 |
| hand_height | 手牌区域高度比例 | 0.16 |

---

## 悬浮窗配置 (OverlayConfig)

在代码中可以这样配置：

```python
from majiang_ai.overlay import OverlayConfig

config = OverlayConfig(
    # 窗口属性
    always_on_top=True,           # 置顶
    transparent_background=True,   # 透明背景
    click_through=True,           # 鼠标穿透
    
    # 对齐跟随
    auto_follow=True,
    follow_interval_ms=500,
    
    # 视觉样式
    opacity=0.9,
    arrow_color=(0, 255, 0),      # 绿色箭头
    text_color=(255, 255, 255),
    high_score_color=(0, 255, 0),
    medium_score_color=(255, 200, 0),
    low_score_color=(150, 150, 150),
    
    # 安全
    anti_screenshot=True          # 反截图
)
```

---

## 算牌器配置 (EnhancedMahjongEvaluator)

调整算法偏好：

```python
from majiang_ai.enhanced_evaluator import EnhancedMahjongEvaluator

evaluator = EnhancedMahjongEvaluator(
    speed_priority=0.6,     # 速度优先权重
    safety_priority=0.3,     # 安全权重
    fan_priority=0.1,        # 番型权重
    
    base_win_points=1,       # 基础胡牌得分
    kang_points=2,           # 杠得分
    horse_points=1           # 买马得分
)
```

**调整建议：**
- 快速场：提高 `speed_priority` 到 0.8
- 高端局：提高 `safety_priority` 到 0.5
- 想做大牌：提高 `fan_priority` 到 0.3

---

## YOLO检测配置 (YOLO11Detector)

```python
from majiang_ai.yolo_detector import YOLO11Detector

detector = YOLO11Detector(
    model_path="path/to/model.onnx",
    conf_threshold=0.5,      # 置信度阈值
    iou_threshold=0.45,      # NMS IOU阈值
    use_onnx=True,           # 使用ONNX推理
    device="cpu"             # 或 "cuda"
)
```

---

## 虚拟相机配置 (VirtualCameraConfig)

```python
from majiang_ai.virtual_camera import VirtualCameraConfig

config = VirtualCameraConfig(
    camera_index=0,
    width=1920,
    height=1080,
    fps=30,
    auto_detect=True
)
```

---

## 动态UI识别配置 (DynamicUIRecognizer)

```python
from majiang_ai.dynamic_ui import DynamicUIRecognizer

recognizer = DynamicUIRecognizer(
    hand_height_threshold=0.7,    # 手牌高度阈值
    meld_height_threshold=0.5,    # 副露高度阈值
    hand_aspect_ratio=0.77,       # 手牌宽高比
    meld_aspect_ratio=1.3,        # 副露宽高比
    debug=False
)
```

---

## 命令行参数

运行 `assistant.py` 时可用的参数：

```bash
python assistant.py [选项]
```

| 选项 | 说明 |
|------|------|
| `--virtual-camera` | 使用OBS虚拟相机模式 |
| `--yolo` | 启用YOLO检测 |
| `--yolo-model PATH` | 指定YOLO模型路径 |
| `--no-overlay` | 不显示悬浮窗 |
| `--debug` | 调试模式 |

例如：
```bash
# 使用虚拟相机+YOLO+调试模式
python assistant.py --virtual-camera --yolo --yolo-model ./best.onnx --debug
```
