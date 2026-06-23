# Changelog

## v0.2.0 — 2026-06-23

### 核心改进：精准切图 + 模板匹配牌面识别

#### 新增功能
- **`analyze_now.py`**: 一键截图识别当前手牌并给出出牌建议
- **`build_final.py`**: 用对齐工具坐标系批量切出模板牌面
- **`gen_tuner_html.py`**: 生成交互式 HTML 逐牌对齐工具（可拖拽绿色竖线对齐每张牌的右边界，支持 Tab/←→ 微调）

#### 识别引擎 (`vision.py`)
- **`segment_hand_strip`** 新增三层优先级切割策略：
  1. `calibration_config.json` 中的精确比例坐标（窗口缩放自适应）
  2. `_scan_hand_by_templates` 全图模板匹配自扫描（多缩放因子 + NMS 去重）
  3. 列投影 + Otsu 自动检测（终极降级）
- `load_templates` 过滤 slot_NN 文件，只加载 `[1-9][mpsz]` 编码模板
- `recognize_tiles` 新增 `tile_bounds` 参数透传切割坐标
- 切割坐标支持绝对像素和相对比例（`right < 2.0` 自动识别为比例模式）

#### 配置持久化 (`calibration_config.json`)
- 手牌区域：`hand: {left, top, width, height}` 相对窗口比例
- 13 张牌边界：`tiles: [{left, right}]` 相对手牌宽度的比例
- CLI (`cli.py`) 启动时自动加载 `tiles` 配置传给识别流程

#### 模板库 (`tile_templates/`)
- 13 张编码模板：`3m 4m 6m 8m 1s 2s 6s 9s 2p 4p 5z 6z 7z`
- 字牌编码：1z=东 2z=南 3z=西 4z=北 5z=白 6z=发 7z=中

#### 其他修复
- PrintWindow 回退逻辑修复（`or` 短路问题 → `if full is None`）
- 错误文件恢复（SearchReplace 操作误覆盖后完整重写 vision.py）

---

## v0.1.0 — 初始版本（历史）

- Win32 PrintWindow API 后台窗口捕获
- 基础的列投影牌面分割
- 基于颜色直方图的牌面分类
- MCTS 出牌决策引擎
- 命令行 `--live` / `--calibrate` 模式
