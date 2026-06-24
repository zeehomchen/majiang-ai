# 故障排除指南

## 常见问题

---

### 安装问题

**Q: pip install 失败？**

A:
1. 升级 pip：`python -m pip install --upgrade pip`
2. 使用国内镜像：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
3. 网络问题配置代理或使用离线安装包

**Q: No module named 'xxx'**

A: 确保在正确的Python环境中，检查已安装包：
```bash
pip list
```

---

### 采集问题

**Q: 找不到麻将窗口？**

A:
- 确保游戏已打开
- 尝试管理员权限运行
- 检查窗口标题是否包含「麻将」「腾讯」等关键词
- 手动指定窗口标题：修改 `capture.py` 中的关键词

**Q: 虚拟相机无画面？**

A:
- 确认OBS虚拟相机已启动
- 尝试Windows「相机」应用测试虚拟相机是否正常
- 调整相机索引（修改 `virtual_camera.py` 的 `camera_index`）
- 检查OBS中的「来源」是否被「停用」

**Q: 画面很卡？**

A:
- 降低OBS输出分辨率到1280x720
- 降低FPS到30
- 关闭 `--debug` 模式

---

### 悬浮窗问题

**Q: 悬浮窗不显示？**

A:
1. 检查PyQt5是否安装：`pip install PyQt5`
2. 尝试不带悬浮窗运行：`python assistant.py --no-overlay`
3. 检查Windows版本，需要Win10/11

**Q: 悬浮窗挡住游戏操作？**

A: 鼠标穿透应该已启用，如果不工作：
- 确保 `OverlayConfig.click_through = True`
- 使用Windows Alt+Tab切换窗口测试

**Q: 悬浮窗没有对齐游戏窗口？**

A:
- 检查 `auto_follow` 是否开启
- 游戏窗口移动后需要等一会（默认500ms）
- 可以手动调整悬浮窗位置

**Q: 能看到悬浮窗，但录屏/截图里没有？**

A: 这是反截图功能在正常工作！
- 如需测试截图，设置 `anti_screenshot=False`

---

### 识别问题

**Q: 识别不准确？**

A:
- 当前版本需要YOLO模型才能准确识别
- 没有模型时使用的是传统视觉方法，效果有限
- 查看 [TRAIN_YOLO.md](./TRAIN_YOLO.md) 训练自己的模型

**Q: 手牌和副露分不清？**

A:
- 调整 `DynamicUIRecognizer` 的阈值
- 检查麻将牌大小是否有明显区别
- 使用YOLO模型会有更好的效果

---

### 算牌问题

**Q: 推荐的切牌感觉不对？**

A:
- 可以调整算法权重，提高速度或安全优先级
- 查看 `CONFIG.md` 了解如何配置
- 确认输入的手牌编码正确

**Q: 13张牌时不推荐切牌？**

A: 这是正常的！13张是摸牌前状态，需要等摸到第14张才推荐切牌。

---

### YOLO问题

**Q: onnxruntime 导入错误？**

A:
1. 确保安装正确：`pip install onnxruntime`
2. 对于GPU加速：`pip install onnxruntime-gpu`
3. 检查CUDA/cuDNN版本（GPU版）

**Q: 模型加载失败？**

A:
- 检查模型文件路径是否正确
- 确认文件是 .onnx 或 .pt 格式
- 查看模型是否损坏

---

### 性能问题

**Q: CPU占用很高？**

A:
- 降低FPS（虚拟相机或采集频率）
- 使用ONNX推理比PyTorch快
- 使用更小的模型（yolo11n.pt）

**Q: 延迟高？**

A:
- 减少中间调试输出
- 使用多线程/多进程处理
- 降低图像分辨率

---

## 获取帮助

如果上述都无法解决：

1. 启用调试模式运行：`python assistant.py --debug`
2. 查看终端错误信息
3. 检查是否有错误日志输出
4. 提供以下信息寻求帮助：
   - Windows版本
   - Python版本
   - 错误日志
   - 使用的采集方式

---

## 回退方案

如果新版本太复杂，可以使用原版本：

```bash
python main.py --live
```
