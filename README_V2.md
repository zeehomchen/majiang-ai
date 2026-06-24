# 广东推倒胡算牌器 - 增强版

基于 **OBS虚拟相机 + YOLO11 + EV模型 + 桌面悬浮窗** 的麻将决策系统，针对广东推倒胡规则优化。

---

## 📚 文档导航

| 文档 | 说明 |
|------|------|
| 🚀 [快速入门](./docs/QUICKSTART.md) | 5分钟上手教程 |
| 🎥 [OBS设置](./docs/OBS_SETUP.md) | 虚拟相机详细配置 |
| ⚙️ [配置详解](./docs/CONFIG.md) | 所有配置选项 |
| 📖 [API文档](./docs/API.md) | 开发者接口 |
| 🎓 [YOLO训练](./docs/TRAIN_YOLO.md) | 训练自己的模型 |
| ❓ [故障排除](./docs/TROUBLESHOOTING.md) | 常见问题解决 |
| 📂 [文档索引](./docs/README.md) | 全部文档导航 |

---

## 🌟 新功能

### 1. 安全的画面采集
- **OBS虚拟相机模式**：完全非侵入式，不挂钩游戏进程
- **PrintWindow模式**：后台窗口采集，不影响游戏
- **mss/dxcam降级**：多种采集方式自动切换

### 2. 动态UI识别
- 基于牌体尺寸特征自动区分手牌/副露
- 自动检测分割边界，不依赖固定坐标
- 支持碰杠后的手牌区域变化

### 3. YOLO11物体检测
- 68类检测（34种牌 × 手牌/副露状态）
- ONNX格式导出，CPU推理优化
- 目标延迟：<20ms

### 4. 增强版算牌算法
- **14张限制**：严格遵循摸牌后切牌规则
- **EV期望收益模型**：综合胡牌概率、得分期望
- **速度优先策略**：推倒胡以听牌速度为核心

### 5. 桌面悬浮窗
- **置顶显示**：WS_EX_TOPMOST属性
- **透明背景**：Alpha通道完全透明
- **鼠标穿透**：WS_EX_TRANSPARENT，不阻挡游戏操作
- **窗口跟随**：自动对齐游戏窗口
- **反截图**：SetWindowDisplayAffinity防截屏

## 安装

```bash
# 基础依赖
pip install opencv-python numpy pillow mss

# YOLO相关（可选）
pip install ultralytics onnxruntime

# 悬浮窗（推荐）
pip install PyQt5

# 完整依赖
pip install -r requirements.txt
```

## 快速开始

### OBS虚拟相机模式（推荐）

1. 打开OBS Studio
2. 添加游戏源或窗口捕获，捕获麻将游戏
3. 点击"启动虚拟相机"
4. 运行算牌器：

```bash
python assistant.py --virtual-camera
```

### 传统窗口模式

```bash
python assistant.py
```

### 命令行参数

```bash
--virtual-camera    # 使用OBS虚拟相机
--yolo              # 启用YOLO检测（需要模型）
--yolo-model PATH   # YOLO模型路径
--no-overlay        # 不使用悬浮窗
--debug             # 调试模式
```

## 使用说明

### 悬浮窗功能

- **绿色箭头**：指向最优切牌
- **分数标签**：每张牌的切牌分数
  - ≥80分：绿色（强烈推荐）
  - 40-79分：黄色（可选）
  - <40分：灰色（不推荐）
- **左上角状态**：向听数、听牌张、番型趋势、危险张
- **碰/杠提示**：黄色/红色圆点提示

### 安全提示

⚠️ 本工具仅供学习研究使用。使用第三方工具有封号风险，请自行承担责任。

## 项目结构

```
majiang-ai-prototype/
├── src/majiang_ai/
│   ├── capture.py           # 画面采集（增强版）
│   ├── virtual_camera.py    # OBS虚拟相机
│   ├── dynamic_ui.py        # 动态UI识别
│   ├── yolo_detector.py     # YOLO11检测
│   ├── enhanced_evaluator.py # EV模型评估
│   ├── overlay.py           # 桌面悬浮窗
│   ├── evaluator.py         # 原评估器
│   ├── vision.py            # 原视觉识别
│   ├── parser.py            # 牌编码解析
│   └── ...
├── assistant.py             # 新主程序
├── main.py                  # 原主程序
├── requirements.txt         # 依赖
└── README.md
```

## YOLO训练（可选）

如需训练自己的YOLO模型：

1. 准备数据集，标注格式：`hand_1m`, `meld_1m`等
2. 生成类别名文件：
```python
from majiang_ai.yolo_detector import create_training_template
create_training_template()
```
3. 使用ultralytics训练
4. 导出为ONNX：
```python
from majiang_ai.yolo_detector import YOLO11Detector
detector = YOLO11Detector()
detector.export_to_onnx("best.pt")
```

## 需求规约对照

✅ 采集安全：OBS虚拟相机非侵入式  
✅ 动态UI：基于牌体特征识别  
✅ YOLO检测：68类端到端检测  
✅ 算牌算法：14张限制+EV模型  
✅ 悬浮窗：置顶/透明/穿透/跟随  
✅ 反截图：SetWindowDisplayAffinity  

## 许可证

MIT
