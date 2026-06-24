# YOLO模型训练指南

## 目录

1. [准备数据集](#准备数据集)
2. [标注数据](#标注数据)
3. [训练模型](#训练模型)
4. [导出ONNX](#导出onnx)
5. [使用模型](#使用模型)

---

## 准备数据集

### 数据采集

从游戏中采集截图，建议：
- 不同手牌组合
- 碰/杠后的不同状态
- 不同分辨率
- 至少500-1000张图片

保存到 `datasets/images/` 目录。

---

## 标注数据

### 标签格式

YOLO使用txt格式的标签文件，每个标签文件和图片同名。

标签格式：
```
class_id x_center y_center width height
```

坐标都是相对于图片的比例（0-1）。

### 类别定义

我们有68个类别：

| 前缀 | 说明 | 数量 |
|------|------|------|
| `hand_*` | 手牌状态 | 34 |
| `meld_*` | 副露状态 | 34 |

具体类别：
```
hand_1m, hand_2m, ..., hand_9m,
hand_1p, hand_2p, ..., hand_9p,
hand_1s, hand_2s, ..., hand_9s,
hand_1z, hand_2z, ..., hand_7z,
meld_1m, meld_2m, ..., meld_7z
```

可以使用 `yolo_detector.py` 生成类别名文件：

```python
from majiang_ai.yolo_detector import YOLO11Detector
YOLO11Detector.generate_class_names_file("class_names.txt")
```

### 标注工具

推荐使用：
- **LabelImg** - https://github.com/HumanSignal/labelImg
- **Roboflow** - https://roboflow.com/ (在线)

---

## 训练模型

### 1. 目录结构

```
datasets/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
dataset.yaml
```

### 2. 创建 dataset.yaml

```yaml
path: ./datasets
train: images/train
val: images/val

names:
  0: hand_1m
  1: hand_2m
  # ... 共68个类别
```

或者使用自动生成：

```python
from majiang_ai.yolo_detector import CLASS_NAMES

names_dict = {i: name for i, name in enumerate(CLASS_NAMES)}

import yaml
with open("dataset.yaml", "w") as f:
    yaml.dump({
        "path": "./datasets",
        "train": "images/train",
        "val": "images/val",
        "names": names_dict
    }, f)
```

### 3. 开始训练

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolo11n.pt")

# 训练
results = model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="0"  # 或 "cpu"
)
```

### 4. 验证

```python
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
```

---

## 导出ONNX

训练完成后，导出为ONNX格式用于推理：

```python
from majiang_ai.yolo_detector import YOLO11Detector

detector = YOLO11Detector()
onnx_path = detector.export_to_onnx(
    pt_model_path="runs/detect/train/weights/best.pt",
    imgsz=640,
    simplify=True
)
print(f"已导出: {onnx_path}")
```

或者使用ultralytics直接导出：

```python
model = YOLO("runs/detect/train/weights/best.pt")
model.export(format="onnx", imgsz=640, simplify=True)
```

---

## 使用模型

### 在代码中使用

```python
from majiang_ai.yolo_detector import YOLO11Detector

detector = YOLO11Detector(
    model_path="./best.onnx",
    conf_threshold=0.5,
    use_onnx=True
)
detector.load_model()

# 检测
result = detector.predict(image_array)

# 可视化
vis_image = detector.visualize(image_array, result)
```

### 在assistant.py中使用

```bash
python assistant.py --yolo --yolo-model ./best.onnx
```

---

## 训练技巧

### 数据增强
- 翻转、旋转、缩放
- 颜色抖动
- 马赛克增强

### 类别平衡
- 确保每个类别样本数相近
- 可以使用过采样/欠采样

### 模型选择
- yolo11n - 最快，精度稍低
- yolo11s - 平衡
- yolo11m - 更准，更慢

### 超参数
- 学习率：1e-3
- 批大小：16-32
- 训练轮数：50-200
