# 广东推倒胡智能算牌器

基于 **OBS虚拟相机采集 + 多策略图像匹配 + EV期望收益模型** 的麻将实战AI助手，专为**广东推倒胡**规则深度优化。

---

## 快速开始

```bash
# 1. 安装依赖
pip install opencv-python pillow numpy tkinter

# 2. 启动统一GUI
python majiang_gui.py
```

**三步上手**：
1. 点击 **「标定区域」** → 在视频上画框选出手牌区域 → 点确定
2. 点击 **「采集模板」** → 选择缺失的牌 → 画框 → 保存（或点「保存到待确认」截图后手动重命名）
3. 点击 **「🎯 开始算牌」** → 实时识别 + 自动算牌 + 画面显示建议

---

## 功能总览

### 采集安全（非侵入式）

| 方式 | 说明 |
|------|------|
| **OBS虚拟相机** | 推荐。完全免注入、不碰游戏进程，直接读取OBS输出的虚拟摄像头画面 |
| DirectX截屏 | 备选。通过 dxcam/mss 截取窗口 |

### 图像识别

```
多策略融合匹配引擎 (match_engine.py)
├── 多尺度模板匹配 (0.8x / 1.0x / 1.2x)
├── Canny边缘匹配
├── HSV颜色直方图 (区分万/筒/条)
├── Sobel纹理特征 (区分字牌)
└── 空间平滑纠错
```

| 策略 | 权重(数牌) | 权重(字牌) | 解决什么 |
|------|-----------|-----------|---------|
| 模板匹配 | 30% | 25% | 整体相似度 |
| 边缘匹配 | 20% | 35% | 光照变化 |
| 颜色直方图 | 35% | 15% | 花色区分 |
| 纹理特征 | 15% | 25% | 字形差异 |

### 算牌引擎

- **EV期望收益模型**：综合胡牌概率、得分期望
- **14张严格限制**：摸牌后才推荐切牌
- **推倒胡规则**：不能吃、可碰杠、买马收益计入

### 输出示例

```
手牌: 123m456p789s11222z
========================================
🏆 最佳切牌: 1z (东)
   分数: 322.89
   有效进张: 28

Top 选项:
   1. 打 东 (1z) | 总分 322.89
   2. 打 南 (2z) | 总分 310.45
   3. 打 一萬 (1m) | 总分 298.12
```

---

## 所有程序一览

| 程序 | 功能 | 使用方式 |
|------|------|---------|
| **`majiang_gui.py`** | **统一GUI界面** | `python majiang_gui.py` |
| `match_engine.py` | 多策略匹配引擎 | 被 GUI 和诊断脚本调用 |
| `diagnose_match.py` | 逐张牌诊断匹配效果 | `python diagnose_match.py` |
| `chinese_input.py` | 中文输入手动算牌 | `python chinese_input.py` |
| `main.py` | 原始CLI算牌 | `python main.py --hand 123m456p789s11222z` |
| `calibrate.py` | 纯标定工具 | `python calibrate.py` |
| `simple_auto.py` | 简易自动识别 | `python simple_auto.py` |
| `test_camera_1.py` | 测试OBS相机 | `python test_camera_1.py` |
| `check_templates.py` | 检查模板覆盖情况 | `python check_templates.py` |
| `self_play.py` | 4-AI差异化自对弈 | `python self_play.py 50` |

---

## 项目结构

```
majiang-ai-prototype/
├── majiang_gui.py            # 统一GUI主程序
├── match_engine.py           # 多策略图像匹配引擎
├── diagnose_match.py         # 匹配效果诊断工具
├── main.py                   # 原始CLI入口
├── self_play.py              # 4-AI自对弈模拟
├── personality_library.py    # 30种人格库
├── chinese_input.py          # 中文输入算牌
├── calibrate.py              # 标定工具
├── simple_auto.py            # 简易自动识别
├── simple_capture.py         # 截图辅助
├── test_camera_1.py          # 相机测试
├── check_templates.py        # 模板检查
├── capture_single.py         # 单张采集
├── README.md
├── calibration_config.json   # 标定配置文件
├── tile_templates/           # 牌面模板图片 (36张)
│   ├── 1m.png ~ 9m.png       # 万子
│   ├── 1p.png ~ 9p.png       # 筒子
│   ├── 1s.png ~ 9s.png       # 条子
│   └── 1z.png ~ 7z.png       # 字牌(东南西北白板发财红中)
├── docs/                     # 文档
└── src/
    └── majiang_ai/
        ├── __init__.py
        ├── cli.py            # 命令行界面
        ├── parser.py         # 牌面编码解析
        ├── models.py         # 数据结构
        ├── evaluator.py      # 启发式路线评估
        ├── capture.py        # 截屏采集
        ├── vision.py         # 牌面图像识别
        ├── virtual_camera.py # OBS虚拟相机
        ├── dynamic_ui.py     # 动态UI识别
        ├── yolo_detector.py  # YOLO检测
        ├── overlay.py        # 桌面悬浮窗
        ├── enhanced_evaluator.py # 增强版算牌
        ├── simulator.py      # 对局模拟器
        └── mcts.py           # MCTS搜索
```

---

## 推倒胡关键规则（引擎内建）

| 规则 | 说明 |
|------|------|
| 不能吃牌 | 无法用上家的牌组成顺子 |
| 可以碰/杠 | 可碰（2张）、明杠/暗杠/补杠 |
| 多对子 ≠ 自摸 | 3对以上对子，摸到第4张不能直接和牌 |
| 买马 | 胡牌后取6张，按位置映射中马翻倍 |
| 13/14张区分 | 13张=摸牌前，14张=摸牌后切牌 |

---

## 4-AI 自对弈

```bash
python self_play.py 50          # 50局批量统计
python self_play.py --detail    # 单局详细回放
```

| AI | 策略 | 特点 |
|----|------|------|
| AI-1 | MCTS搜索 | 每步模拟200局 |
| AI-2 | 启发式+防守 | 路线评分+牌河防守 |
| AI-3 | 激进进攻型 | 优先听牌 |
| AI-4 | 保守防守型 | 避点炮 |

---

## 牌面编码对照

| 万子 | 筒子 | 条子 | 字牌 |
|------|------|------|------|
| 1m=一万 | 1p=一筒 | 1s=一条 | 1z=东 |
| 2m=二万 | 2p=二筒 | 2s=二条 | 2z=南 |
| 3m=三万 | 3p=三筒 | 3s=三条 | 3z=西 |
| 4m=四万 | 4p=四筒 | 4s=四条 | 4z=北 |
| 5m=五万 | 5p=五筒 | 5s=五条 | 5z=白板 |
| 6m=六万 | 6p=六筒 | 6s=六条 | 6z=发财 |
| 7m=七万 | 7p=七筒 | 7s=七条 | 7z=红中 |
| 8m=八万 | 8p=八筒 | 8s=八条 | |
| 9m=九万 | 9p=九筒 | 9s=九条 | |

---

MIT. Free forever.
