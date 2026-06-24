"""
OBS虚拟相机非侵入式画面采集模块
完全免注入、非侵入式，通过OpenCV VideoCapture读取虚拟摄像头流
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class VirtualCameraConfig:
    """虚拟相机配置"""
    camera_index: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    auto_detect: bool = True


class VirtualCameraCapture:
    """
    OBS虚拟相机画面采集器
    通过OpenCV VideoCapture读取虚拟摄像头流
    """

    def __init__(self, config: Optional[VirtualCameraConfig] = None):
        self.config = config or VirtualCameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._started = False

    def start(self) -> bool:
        """
        启动虚拟相机采集

        Returns:
            启动是否成功
        """
        if not CV2_AVAILABLE:
            raise ImportError("需要安装 opencv-python: pip install opencv-python")
        
        # 尝试自动检测可用摄像头
        if self.config.auto_detect:
            camera_index = self._auto_detect_camera()
            if camera_index is None:
                print("警告: 未检测到OBS虚拟相机，尝试默认索引0")
                camera_index = self.config.camera_index
        else:
            camera_index = self.config.camera_index
        
        self._cap = cv2.VideoCapture(camera_index)
        
        if not self._cap.isOpened():
            print(f"错误: 无法打开摄像头索引 {camera_index}")
            return False
        
        # 设置分辨率
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        self._started = True
        print(f"✅ 虚拟相机已启动 (索引 {camera_index})")
        return True

    def stop(self) -> None:
        """停止采集"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._started = False
        print("✅ 虚拟相机已停止")

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        捕获一帧画面

        Returns:
            BGR格式的numpy数组，失败返回None
        """
        if not self._started or self._cap is None:
            return None
        
        ret, frame = self._cap.read()
        if ret:
            return frame
        return None

    def capture_continuous(self, interval: float = 0.033):
        """
        连续捕获画面的生成器

        Args:
            interval: 捕获间隔(秒)

        Yields:
            BGR格式的numpy数组
        """
        while self._started:
            frame = self.capture_frame()
            if frame is not None:
                yield frame
            time.sleep(interval)

    @property
    def is_started(self) -> bool:
        return self._started

    @staticmethod
    def _auto_detect_camera(self, max_index: int = 10) -> Optional[int]:
        """
        自动检测可用的OBS虚拟相机

        Args:
            max_index: 最大尝试索引

        Returns:
            找到的摄像头索引，或None
        """
        if not CV2_AVAILABLE:
            return None
            
        for i in range(max_index):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # 尝试读取一帧确认
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        print(f"检测到可用摄像头: 索引 {i}")
                        return i
            except Exception:
                continue
        return None

    @staticmethod
    def list_cameras(self, max_index: int = 10) -> list[int]:
        """
        列出所有可用摄像头索引

        Args:
            max_index: 最大尝试索引

        Returns:
            可用摄像头索引列表
        """
        if not CV2_AVAILABLE:
            return []
            
        available = []
        for i in range(max_index):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        available.append(i)
            except Exception:
                continue
        return available
