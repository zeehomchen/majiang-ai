# 快速入门指南

## 第一步：环境准备

### 系统要求
- Windows 10/11
- Python 3.10+

### 安装Python依赖

打开 PowerShell 或 Command Prompt，进入项目目录：

```bash
cd c:\Users\Administrator\Desktop\demo\majiang-ai-prototype
```

安装基础依赖：

```bash
pip install -r requirements.txt
```

如果上面命令很慢，可以使用国内镜像源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 第二步：选择采集模式

### 🎥 方案A：OBS虚拟相机（推荐，最安全）

#### 1. 下载安装 OBS Studio

https://obsproject.com/

#### 2. 配置OBS捕获麻将窗口

1. 打开OBS
2. 在「来源」面板点击 `+` → 选择「游戏捕获」或「窗口捕获」
3. 选择你的麻将游戏窗口
4. 调整画面大小，确保麻将窗口全屏显示

#### 3. 启动虚拟相机

在OBS主界面点击 `启动虚拟相机` 按钮

#### 4. 运行算牌器

```bash
python assistant.py --virtual-camera
```

---

### 🖥️ 方案B：直接窗口捕获（简单但有风险）

无需OBS，直接运行：

```bash
python assistant.py
```

程序会自动寻找腾讯麻将窗口。

---

## 第三步：开始使用

### 悬浮窗说明

当检测到游戏窗口后，你会看到：

| 元素 | 说明 |
|------|------|
| 🔝 绿色箭头 | 指向最优切牌 |
| 🔢 分数标签 | 每张牌的切牌价值 |
| 📊 左上角面板 | 向听数、听牌张、危险张 |
| 🟡 黄色圆 | 碰牌提示 |
| 🔴 红色圆 | 杠牌提示 |

### 分数颜色含义

- 🟢 **≥80分**：强烈推荐切这张
- 🟡 **40-79分**：可以考虑
- ⚪ **<40分**：不推荐

### 退出程序

在终端按 `Ctrl + C` 退出

---

## 常见问题

**Q: 找不到虚拟相机怎么办？**

A: 确保OBS的虚拟相机已启动，尝试在 `assistant.py` 中修改相机索引，或使用程序自动检测。

**Q: 悬浮窗不显示？**

A: 检查是否安装了 PyQt5：`pip install PyQt5`

**Q: 检测不准确？**

A: 这是因为当前还没有训练好的YOLO模型。需要标注数据集训练模型才能获得最佳效果。

---

## 下一步

- 查看 [OBS设置详细教程](./docs/OBS_SETUP.md)
- 了解 [配置文件详解](./docs/CONFIG.md)
- 学习 [如何训练YOLO模型](./docs/TRAIN_YOLO.md)
