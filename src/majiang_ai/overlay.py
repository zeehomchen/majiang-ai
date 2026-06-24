"""
桌面悬浮窗模块
实现：
1. 置顶显示
2. 透明背景
3. 鼠标穿透
4. 窗口对齐跟随
5. 反截图对抗
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Callable, List

# 尝试导入GUI库
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QLabel, QWidget,
        QVBoxLayout, QHBoxLayout
    )
    from PyQt5.QtCore import (
        Qt, QTimer, QPoint, QRect, pyqtSignal, QObject
    )
    from PyQt5.QtGui import (
        QPainter, QColor, QPen, QFont, QBrush, QPainterPath
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

try:
    import tkinter as tk
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False


class GUIType(Enum):
    """GUI类型"""
    PYQT = "pyqt"
    TKINTER = "tkinter"
    AUTO = "auto"


@dataclass
class OverlayConfig:
    """悬浮窗配置"""
    # 窗口属性
    always_on_top: bool = True
    transparent_background: bool = True
    click_through: bool = True
    
    # 对齐
    auto_follow: bool = True
    follow_interval_ms: int = 500
    
    # 视觉
    opacity: float = 0.9
    arrow_color: Tuple[int, int, int] = (0, 255, 0)  # 绿色
    text_color: Tuple[int, int, int] = (255, 255, 255)
    high_score_color: Tuple[int, int, int] = (0, 255, 0)
    medium_score_color: Tuple[int, int, int] = (255, 200, 0)
    low_score_color: Tuple[int, int, int] = (150, 150, 150)
    
    # 安全
    anti_screenshot: bool = True


@dataclass
class TileHighlight:
    """牌高亮信息"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    is_best: bool = False  # 是否是最优切牌
    score: float = 0.0
    tile_code: str = ""


@dataclass
class ActionHint:
    """动作提示"""
    action_type: str  # "peng", "gang", "pass"
    confidence: float
    position: Tuple[int, int]  # 提示位置
    message: str = ""


@dataclass
class GlobalStatus:
    """全局状态"""
    shanten: int = 13
    waiting_tiles: List[str] = None
    danger_tiles: List[str] = None
    fan_trend: str = ""


class OverlaySignal(QObject):
    """PyQt信号"""
    update_signal = pyqtSignal(object)
    hide_signal = pyqtSignal()
    show_signal = pyqtSignal()


class MahjongOverlay:
    """
    麻将悬浮窗基类
    """

    def __init__(
        self,
        config: Optional[OverlayConfig] = None,
        gui_type: GUIType = GUIType.AUTO
    ):
        self.config = config or OverlayConfig()
        self._select_gui_type(gui_type)
        
        self._window: Optional[QMainWindow] = None
        self._app: Optional[QApplication] = None
        self._timer: Optional[QTimer] = None
        
        self._target_window_rect: Optional[Tuple[int, int, int, int]] = None
        self._tile_highlights: List[TileHighlight] = []
        self._action_hints: List[ActionHint] = []
        self._global_status: GlobalStatus = GlobalStatus()
        self._is_visible: bool = True
        
        # 窗口发现回调
        self._window_found_callback: Optional[Callable] = None
        self._window_lost_callback: Optional[Callable] = None

    def _select_gui_type(self, gui_type: GUIType) -> None:
        """选择GUI类型"""
        if gui_type == GUIType.AUTO:
            if PYQT_AVAILABLE:
                self._gui_type = GUIType.PYQT
            elif TK_AVAILABLE:
                self._gui_type = GUIType.TKINTER
            else:
                raise ImportError("需要PyQt5或tkinter")
        else:
            self._gui_type = gui_type

    def initialize(self) -> bool:
        """
        初始化悬浮窗

        Returns:
            是否成功
        """
        if self._gui_type == GUIType.PYQT:
            return self._initialize_pyqt()
        else:
            return self._initialize_tkinter()

    def _initialize_pyqt(self) -> bool:
        """初始化PyQt版本"""
        if not PYQT_AVAILABLE:
            print("PyQt5不可用")
            return False

        # 创建应用
        self._app = QApplication.instance()
        if self._app is None:
            self._app = QApplication(sys.argv)

        # 创建主窗口
        self._window = OverlayWindowPyQt(self.config)
        
        # 连接信号
        self._window.signals.update_signal.connect(self._on_update)
        self._window.signals.hide_signal.connect(self._on_hide)
        self._window.signals.show_signal.connect(self._on_show)
        
        # 设置定时器
        if self.config.auto_follow:
            self._timer = QTimer()
            self._timer.timeout.connect(self._follow_window)
            self._timer.start(self.config.follow_interval_ms)

        print("✅ PyQt悬浮窗初始化成功")
        return True

    def _initialize_tkinter(self) -> bool:
        """初始化tkinter版本（简化版）"""
        if not TK_AVAILABLE:
            print("tkinter不可用")
            return False
        print("⚠️ tkinter版本功能有限，推荐使用PyQt5")
        return True

    def show(self) -> None:
        """显示悬浮窗"""
        if self._window:
            self._window.show()
            self._is_visible = True

    def hide(self) -> None:
        """隐藏悬浮窗"""
        if self._window:
            self._window.hide()
            self._is_visible = False

    def set_target_window_rect(
        self,
        rect: Tuple[int, int, int, int]  # x, y, width, height
    ) -> None:
        """
        设置目标窗口位置

        Args:
            rect: (x, y, width, height)
        """
        self._target_window_rect = rect
        if self._window and self._is_visible:
            self._window.set_geometry(rect)

    def update_tile_highlights(
        self,
        highlights: List[TileHighlight]
    ) -> None:
        """
        更新牌高亮

        Args:
            highlights: 牌高亮列表
        """
        self._tile_highlights = highlights
        if self._window:
            self._window.update_highlights(highlights)

    def update_action_hints(
        self,
        hints: List[ActionHint]
    ) -> None:
        """
        更新动作提示

        Args:
            hints: 动作提示列表
        """
        self._action_hints = hints
        if self._window:
            self._window.update_actions(hints)

    def update_global_status(
        self,
        status: GlobalStatus
    ) -> None:
        """
        更新全局状态

        Args:
            status: 全局状态
        """
        self._global_status = status
        if self._window:
            self._window.update_status(status)

    def run(self) -> None:
        """运行主循环"""
        if self._app:
            self.show()
            self._app.exec_()

    def stop(self) -> None:
        """停止"""
        if self._timer:
            self._timer.stop()
        if self._app:
            self._app.quit()

    def _follow_window(self) -> None:
        """跟随目标窗口（定时调用）"""
        if self._window_found_callback:
            self._window_found_callback()

    def _on_update(self, data: object) -> None:
        """更新回调"""
        pass

    def _on_hide(self) -> None:
        """隐藏回调"""
        pass

    def _on_show(self) -> None:
        """显示回调"""
        pass

    def set_window_callbacks(
        self,
        found_callback: Optional[Callable] = None,
        lost_callback: Optional[Callable] = None
    ) -> None:
        """设置窗口回调"""
        self._window_found_callback = found_callback
        self._window_lost_callback = lost_callback


class OverlayWindowPyQt(QMainWindow):
    """PyQt实现的悬浮窗"""

    def __init__(self, config: OverlayConfig):
        super().__init__()
        self.config = config
        self.signals = OverlaySignal()
        
        self._highlights: List[TileHighlight] = []
        self._hints: List[ActionHint] = []
        self._status: GlobalStatus = GlobalStatus()
        
        self._setup_window()

    def _setup_window(self) -> None:
        """设置窗口属性"""
        # 窗口标题
        self.setWindowTitle("麻将助手")
        
        # 无边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        # 置顶
        if self.config.always_on_top:
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        # 透明背景
        if self.config.transparent_background:
            self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 鼠标穿透
        if self.config.click_through:
            self.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        # 设置窗口大小
        self.resize(1920, 1080)
        
        # 反截图
        if self.config.anti_screenshot:
            self._set_anti_screenshot()

    def _set_anti_screenshot(self) -> None:
        """设置反截图（Windows）"""
        try:
            import ctypes
            from ctypes import wintypes
            
            # SetWindowDisplayAffinity
            WDA_EXCLUDEFROMCAPTURE = 0x00000011
            
            user32 = ctypes.windll.user32
            hwnd = int(self.winId())
            result = user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
            if result:
                print("✅ 反截图已启用")
        except Exception as e:
            print(f"反截图设置失败: {e}")

    def set_geometry(self, rect: Tuple[int, int, int, int]) -> None:
        """设置窗口位置和大小"""
        x, y, w, h = rect
        self.setGeometry(x, y, w, h)

    def update_highlights(self, highlights: List[TileHighlight]) -> None:
        """更新牌高亮"""
        self._highlights = highlights
        self.update()

    def update_actions(self, hints: List[ActionHint]) -> None:
        """更新动作提示"""
        self._hints = hints
        self.update()

    def update_status(self, status: GlobalStatus) -> None:
        """更新状态"""
        self._status = status
        self.update()

    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 清空背景
        painter.fillRect(self.rect(), Qt.transparent)
        
        # 绘制牌高亮
        self._paint_highlights(painter)
        
        # 绘制动作提示
        self._paint_actions(painter)
        
        # 绘制全局状态
        self._paint_status(painter)

    def _paint_highlights(self, painter: QPainter) -> None:
        """绘制牌高亮"""
        for highlight in self._highlights:
            x1, y1, x2, y2 = highlight.bbox
            
            # 选择颜色
            if highlight.score >= 80:
                color = QColor(*self.config.high_score_color)
            elif highlight.score >= 40:
                color = QColor(*self.config.medium_score_color)
            else:
                color = QColor(*self.config.low_score_color)
            
            # 绘制高亮框
            pen = QPen(color, 3)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            
            # 最优切牌绘制箭头
            if highlight.is_best:
                self._paint_arrow(painter, x1, y1, x2, y2, color)
            
            # 绘制分数
            if highlight.score > 0:
                painter.setPen(QColor(*self.config.text_color))
                font = QFont("Arial", 10, QFont.Bold)
                painter.setFont(font)
                painter.drawText(
                    x1, y1 - 5,
                    f"{highlight.tile_code} {highlight.score:.0f}"
                )

    def _paint_arrow(
        self,
        painter: QPainter,
        x1: int, y1: int, x2: int, y2: int,
        color: QColor
    ) -> None:
        """绘制向上的箭头"""
        cx = (x1 + x2) // 2
        cy = y1 - 20
        
        # 箭头形状
        path = QPainterPath()
        path.moveTo(cx, cy)
        path.lineTo(cx - 15, cy + 20)
        path.lineTo(cx + 15, cy + 20)
        path.closeSubpath()
        
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)

    def _paint_actions(self, painter: QPainter) -> None:
        """绘制动作提示"""
        for hint in self._hints:
            x, y = hint.position
            
            # 选择颜色
            if hint.action_type == "peng":
                color = QColor(255, 200, 0)  # 黄色
            elif hint.action_type == "gang":
                color = QColor(255, 0, 0)  # 红色
            else:
                color = QColor(128, 128, 128)  # 灰色
            
            # 绘制圆形提示
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(x - 30, y - 30, 60, 60)
            
            # 绘制文字
            painter.setPen(QColor(*self.config.text_color))
            font = QFont("Arial", 14, QFont.Bold)
            painter.setFont(font)
            text = {"peng": "碰", "gang": "杠", "pass": "过"}.get(hint.action_type, "?")
            painter.drawText(x - 10, y + 5, text)

    def _paint_status(self, painter: QPainter) -> None:
        """绘制全局状态"""
        # 在左上角绘制状态
        x, y = 20, 40
        line_height = 25
        
        painter.setPen(QColor(*self.config.text_color))
        font = QFont("Arial", 12)
        painter.setFont(font)
        
        # 向听数
        shanten_text = "听牌" if self._status.shanten == 0 else f"{self._status.shanten}向听"
        painter.drawText(x, y, shanten_text)
        y += line_height
        
        # 听牌时显示听牌张
        if self._status.shanten == 0 and self._status.waiting_tiles:
            waits_text = f"听: {' '.join(self._status.waiting_tiles[:5])}"
            painter.drawText(x, y, waits_text)
            y += line_height
        
        # 番型趋势
        if self._status.fan_trend:
            painter.drawText(x, y, f"趋势: {self._status.fan_trend}")
            y += line_height
        
        # 危险张
        if self._status.danger_tiles:
            painter.setPen(QColor(255, 100, 100))
            danger_text = f"危险: {' '.join(self._status.danger_tiles[:3])}"
            painter.drawText(x, y, danger_text)


def create_overlay(
    config: Optional[OverlayConfig] = None
) -> MahjongOverlay:
    """
    创建悬浮窗实例

    Args:
        config: 配置

    Returns:
        MahjongOverlay实例
    """
    return MahjongOverlay(config=config)
