"""
屏幕采集模块 - 支持多种非侵入式采集方式
1. OBS虚拟相机 (VirtualCameraCapture)
2. PrintWindow API (后台窗口捕获)
3. mss (屏幕截取)
4. dxcam (高性能DXGI捕获)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np

# 导入虚拟相机模块
try:
    from .virtual_camera import VirtualCameraCapture, VirtualCameraConfig
    VIRTUAL_CAMERA_AVAILABLE = True
except ImportError:
    VIRTUAL_CAMERA_AVAILABLE = False


class CaptureMode(Enum):
    """采集模式枚举"""
    VIRTUAL_CAMERA = "virtual_camera"
    PRINT_WINDOW = "print_window"
    MSS = "mss"
    DX_CAM = "dxcam"
    AUTO = "auto"  # 自动选择最优方式


# ---------------------------------------------------------------------------
# 依赖检测
# ---------------------------------------------------------------------------
_MSS_AVAILABLE = False
_DXCAM_AVAILABLE = False

try:
    import mss  # type: ignore[import-untyped]

    _MSS_AVAILABLE = True
except ImportError:
    pass

try:
    import dxcam  # type: ignore[import-untyped]

    _DXCAM_AVAILABLE = True
except ImportError:
    pass

# 优先使用 dxcam (低延迟), 不可用时降级为 mss
_USE_DXCAM = _DXCAM_AVAILABLE


# ---------------------------------------------------------------------------
# 区域坐标配置
# ---------------------------------------------------------------------------


@dataclass
class Region:
    left: float
    top: float
    width: float
    height: float
    label: str = ""


@dataclass
class CaptureConfig:
    """腾讯麻将窗口内各区域的坐标比例 (0.0-1.0)。

    默认值是估算值，首次使用建议运行 calibrate() 校准。
    """

    # 基于方差扫描自动探测后的默认值
    # 游戏窗口 812×499，内容集中在 0%-70% 高度
    hand_left: float = 0.05
    hand_top: float = 0.40
    hand_width: float = 0.90
    hand_height: float = 0.16

    river_left: float = 0.10
    river_top: float = 0.22
    river_width: float = 0.80
    river_height: float = 0.36

    upper_hand_left: float = 0.30
    upper_hand_top: float = 0.02
    upper_hand_width: float = 0.40
    upper_hand_height: float = 0.10

    upper_river_left: float = 0.22
    upper_river_top: float = 0.12
    upper_river_width: float = 0.56
    upper_river_height: float = 0.10

    opposite_hand_left: float = 0.88
    opposite_hand_top: float = 0.20
    opposite_hand_width: float = 0.10
    opposite_hand_height: float = 0.60

    opposite_river_left: float = 0.78
    opposite_river_top: float = 0.22
    opposite_river_width: float = 0.10
    opposite_river_height: float = 0.36

    lower_hand_left: float = 0.02
    lower_hand_top: float = 0.20
    lower_hand_width: float = 0.10
    lower_hand_height: float = 0.60

    lower_river_left: float = 0.12
    lower_river_top: float = 0.22
    lower_river_width: float = 0.10
    lower_river_height: float = 0.36

    discard_indicator_left: float = 0.30
    discard_indicator_top: float = 0.58
    discard_indicator_width: float = 0.40
    discard_indicator_height: float = 0.06

    action_left: float = 0.20
    action_top: float = 0.58
    action_width: float = 0.60
    action_height: float = 0.10


DEFAULT_CONFIG = CaptureConfig()
CONFIG_FILE = "calibration_config.json"


def _load_saved_config() -> Optional[CaptureConfig]:
    """加载上次校准保存的配置。"""
    path = Path(CONFIG_FILE)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        hand = data.get("hand", {})
        return CaptureConfig(
            hand_left=hand.get("left", DEFAULT_CONFIG.hand_left),
            hand_top=hand.get("top", DEFAULT_CONFIG.hand_top),
            hand_width=hand.get("width", DEFAULT_CONFIG.hand_width),
            hand_height=hand.get("height", DEFAULT_CONFIG.hand_height),
        )
    except Exception:
        return None


def _save_config(cfg: CaptureConfig) -> None:
    """保存校准配置到 JSON 文件。"""
    data = {
        "hand": {
            "left": cfg.hand_left,
            "top": cfg.hand_top,
            "width": cfg.hand_width,
            "height": cfg.hand_height,
        }
    }
    Path(CONFIG_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 窗口查找
# ---------------------------------------------------------------------------

def find_mahjong_window(title_keywords: Optional[list[str]] = None) -> Optional[tuple[int, int, int, int, int]]:
    """扫描桌面窗口，找到腾讯麻将窗口。Returns (hwnd, left, top, right, bottom) 或 None。"""
    if title_keywords is None:
        title_keywords = ["腾讯", "麻将", "QQGame", "Mahjong", "Tencent"]

    try:
        from ctypes import byref, create_unicode_buffer, windll, WINFUNCTYPE
        from ctypes.wintypes import BOOL, HWND, LPARAM, RECT

        user32 = windll.user32
    except Exception:
        return None

    result = None
    WNDENUMPROC = WINFUNCTYPE(BOOL, HWND, LPARAM)

    def enum_proc(hwnd, _lparam):
        nonlocal result
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value or ""
        if any(kw.lower() in title.lower() for kw in title_keywords):
            rect = RECT()
            if user32.GetWindowRect(hwnd, byref(rect)):
                result = (hwnd, rect.left, rect.top, rect.right, rect.bottom)
                return False
        return True

    user32.EnumWindows(WNDENUMPROC(enum_proc), 0)
    return result


def get_window_rect_from_hwnd(hwnd: int) -> Optional[tuple[int, int, int, int]]:
    try:
        from ctypes import byref
        from ctypes.wintypes import RECT

        user32 = __import__("ctypes").windll.user32
        rect = RECT()
        if user32.GetWindowRect(hwnd, byref(rect)):
            return rect.left, rect.top, rect.right, rect.bottom
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 捕获器
# ---------------------------------------------------------------------------

class CaptureError(Exception):
    """捕获失败统一异常。"""


class Capturer:
    """
    通用屏幕采集器
    支持多种非侵入式采集方式，优先使用OBS虚拟相机
    """

    def __init__(
        self,
        window_title: Optional[str] = None,
        title_keywords: Optional[list[str]] = None,
        config: Optional[CaptureConfig] = None,
        mode: CaptureMode = CaptureMode.AUTO,
        virtual_camera_config: Optional[VirtualCameraConfig] = None,
        debug: bool = False,
    ):
        self._title_keywords = title_keywords
        self._config = config or _load_saved_config() or DEFAULT_CONFIG
        self._mode = mode
        self._virtual_camera_config = virtual_camera_config
        self._debug = debug

        self._virtual_camera: Optional[VirtualCameraCapture] = None
        self._camera: Any = None  # dxcam
        self._sct: Any = None  # mss
        self._hwnd: Optional[int] = None
        self._window_rect: Optional[tuple[int, int, int, int]] = None
        self._monitor_offset: tuple[int, int] = (0, 0)
        self._started = False
        self._engine: str = ""
        self._print_window_frame: Optional[np.ndarray] = None


    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """
        启动采集器

        根据选择的模式启动对应的采集方式

        Returns:
            启动是否成功
        """
        # 模式1: 虚拟相机
        if self._mode in (CaptureMode.VIRTUAL_CAMERA, CaptureMode.AUTO):
            if VIRTUAL_CAMERA_AVAILABLE:
                try:
                    self._virtual_camera = VirtualCameraCapture(self._virtual_camera_config)
                    if self._virtual_camera.start():
                        self._engine = "virtual_camera"
                        self._started = True
                        return True
                except Exception as e:
                    if self._debug:
                        print(f"虚拟相机启动失败: {e}")
                
                # 如果是AUTO模式且虚拟相机失败，继续尝试其他方式
                if self._mode == CaptureMode.VIRTUAL_CAMERA:
                    raise CaptureError("虚拟相机启动失败")
        
        # 模式2: PrintWindow / mss / dxcam（需要找到窗口）
        keywords = self._title_keywords or ["腾讯", "麻将", "QQGame", "Mahjong", "Tencent"]
        found = find_mahjong_window(keywords)
        if found is None:
            raise CaptureError(f"未找到麻将游戏窗口。搜索关键词: {keywords}")

        self._hwnd = found[0]
        self._window_rect = (found[1], found[2], found[3], found[4])
        
        # 尝试PrintWindow
        if self._mode in (CaptureMode.PRINT_WINDOW, CaptureMode.AUTO):
            test_frame = self._grab_printwindow()
            if test_frame is not None:
                self._engine = "print_window"
                self._started = True
                print(f"✅ PrintWindow采集模式已启动")
                return True
        
        # 初始化mss作为降级方案
        if _MSS_AVAILABLE:
            self._sct = mss.mss()
        
        # 初始化dxcam
        use_dxcam = _DXCAM_AVAILABLE and self._mode in (CaptureMode.DX_CAM, CaptureMode.AUTO)
        if use_dxcam:
            try:
                self._monitor_offset = self._find_monitor_for_window(found[1:])
                self._camera = dxcam.create(region=self._monitor_region(), output_color="BGR")
                self._engine = "dxcam"
            except Exception as exc:
                if self._debug:
                    print(f"dxcam初始化失败: {exc}，降级为mss")
                self._camera = None

        if self._camera is None and self._sct is not None:
            self._engine = "mss"
        elif self._camera is None:
            raise CaptureError("没有可用的截屏引擎。请安装: pip install mss dxcam opencv-python")

        self._started = True
        print(f"✅ 采集引擎启动: {self._engine}")
        return True


    def stop(self) -> None:
        """停止采集器"""
        self._started = False
        # 停止虚拟相机
        if self._virtual_camera is not None:
            self._virtual_camera.stop()
            self._virtual_camera = None
        self._camera = None
        self._sct = None
        self._print_window_frame = None


    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def window_rect(self) -> Optional[tuple[int, int, int, int]]:
        return self._window_rect

    @property
    def engine(self) -> str:
        return self._engine

    def _refresh_window_rect(self) -> None:
        if self._hwnd is not None:
            fresh = get_window_rect_from_hwnd(self._hwnd)
            if fresh:
                self._window_rect = fresh

    def _find_monitor_for_window(self, win_rect: tuple[int, int, int, int]) -> tuple[int, int]:
        """找到窗口所在的显示器左上角偏移量。"""
        wl, wt, wr, wb = win_rect
        cx, cy = (wl + wr) // 2, (wt + wb) // 2
        try:
            from ctypes import byref, windll
            from ctypes.wintypes import HMONITOR, RECT

            user32 = windll.user32

            def monitor_enum(hmon, hdc, prect, _lparam):
                rect = RECT.from_address(prect)
                if rect.left <= cx <= rect.right and rect.top <= cy <= rect.bottom:
                    result = (rect.left, rect.top)
                    return False
                return True

            MONITORENUMPROC = __import__("ctypes").WINFUNCTYPE(
                __import__("ctypes").c_bool,
                HMONITOR,
                __import__("ctypes").c_void_p,
                __import__("ctypes").c_void_p,
                __import__("ctypes").c_long,
            )
            result = [(0, 0)]

            def cb(hmon, hdc, prect, lp):
                rect = RECT.from_address(prect)
                if rect.left <= cx <= rect.right and rect.top <= cy <= rect.bottom:
                    result[0] = (rect.left, rect.top)
                    return False
                return True

            user32.EnumDisplayMonitors(0, 0, MONITORENUMPROC(cb), 0)
            return result[0]
        except Exception:
            return (0, 0)

    def _monitor_region(self) -> Optional[tuple[int, int, int, int]]:
        """返回窗口所在显示器的区域。"""
        mx, my = self._monitor_offset
        if self._window_rect is None:
            return None
        wl, wt, wr, wb = self._window_rect
        # 构造包含窗口的显示器区域 (给 dxcam 初始化用)
        return (mx, my, mx + 2560, my + 1440)

    # ------------------------------------------------------------------
    # 核心抓取
    # ------------------------------------------------------------------

    def _grab_printwindow(self) -> Optional[np.ndarray]:
        """通过 PrintWindow API 捕获窗口完整客户区（即使最小化/被遮挡也有效）。"""
        if self._hwnd is None:
            return None
        try:
            from ctypes import wintypes
            import ctypes

            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32

            # 获取窗口客户区尺寸
            class RECT(ctypes.Structure):
                _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                            ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
            rc = RECT()
            user32.GetClientRect(self._hwnd, ctypes.byref(rc))
            cw, ch = rc.right - rc.left, rc.bottom - rc.top
            if cw <= 0 or ch <= 0:
                return None

            # 创建内存 DC 和位图
            hdc_screen = user32.GetDC(self._hwnd)
            hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
            hbm = gdi32.CreateCompatibleBitmap(hdc_screen, cw, ch)
            gdi32.SelectObject(hdc_mem, hbm)

            # PrintWindow: PW_RENDERFULLCONTENT = 2 (Win 8.1+), 回退到 PW_CLIENTONLY = 1
            PW_CLIENTONLY = 1
            PW_RENDERFULLCONTENT = 2
            ok = user32.PrintWindow(self._hwnd, hdc_mem, PW_RENDERFULLCONTENT)
            if not ok:
                ok = user32.PrintWindow(self._hwnd, hdc_mem, PW_CLIENTONLY)

            if not ok:
                gdi32.DeleteObject(hbm)
                gdi32.DeleteDC(hdc_mem)
                user32.ReleaseDC(self._hwnd, hdc_screen)
                return None

            # 读取位图数据
            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
                    ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
                    ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                    ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                    ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
                    ("biClrImportant", ctypes.c_uint32),
                ]
            bi = BITMAPINFOHEADER()
            bi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bi.biWidth = cw
            bi.biHeight = -ch  # 负值 = top-down
            bi.biPlanes = 1
            bi.biBitCount = 32
            bi.biCompression = 0  # BI_RGB

            buf = (ctypes.c_ubyte * (cw * ch * 4))()
            gdi32.GetDIBits(hdc_mem, hbm, 0, ch, buf, ctypes.byref(bi), 0)

            gdi32.DeleteObject(hbm)
            gdi32.DeleteDC(hdc_mem)
            user32.ReleaseDC(self._hwnd, hdc_screen)

            # BGRA -> BGR numpy
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(ch, cw, 4)
            return arr[:, :, :3].copy()

        except Exception:
            return None

    def capture_region(self, left: float, top: float, width: float, height: float) -> np.ndarray:
        """
        抓取窗口内相对坐标区域 (0.0-1.0) 的截图，返回 BGR numpy 数组

        支持多种采集方式：
        - 虚拟相机 (OBS)
        - PrintWindow (后台捕获)
        - dxcam / mss (前台截屏)
        """
        if not self._started:
            raise CaptureError("Capturer未启动，请先调用start()。")

        # 方式1: 虚拟相机
        if self._engine == "virtual_camera" and self._virtual_camera is not None:
            frame = self._virtual_camera.capture_frame()
            if frame is None:
                raise CaptureError("虚拟相机捕获失败")
            h, w = frame.shape[:2]
            x1 = int(left * w)
            y1 = int(top * h)
            x2 = int((left + width) * w)
            y2 = int((top + height) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            return frame[y1:y2, x1:x2].copy()

        # 方式2: PrintWindow
        if self._engine == "print_window":
            full = self._grab_printwindow()
            if full is not None and full.size > 0:
                fh, fw = full.shape[:2]
                x1 = int(left * fw)
                y1 = int(top * fh)
                x2 = int((left + width) * fw)
                y2 = int((top + height) * fh)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(fw, x2), min(fh, y2)
                return full[y1:y2, x1:x2].copy()
            raise CaptureError("PrintWindow捕获失败")

        # 方式3: dxcam/mss
        self._refresh_window_rect()
        if self._window_rect is None:
            raise CaptureError("无法获取窗口坐标。")

        wl, wt, wr, wb = self._window_rect
        ww = wr - wl
        wh = wb - wt

        sx = int(wl + left * ww)
        sy = int(wt + top * wh)
        sw = int(width * ww)
        sh = int(height * wh)
        sw = max(1, sw)
        sh = max(1, sh)

        if self._engine == "dxcam" and self._camera is not None:
            frame = self._grab_dxcam(sx, sy, sw, sh)
            if frame is not None:
                return frame

        if self._sct is not None:
            return self._grab_mss(sx, sy, sw, sh)

        raise CaptureError("所有截屏引擎均失败。")

    def _grab_dxcam(self, sx: int, sy: int, sw: int, sh: int) -> Optional[np.ndarray]:
        """dxcam 抓取 (显示器相对坐标)。"""
        mx, my = self._monitor_offset
        rx, ry = sx - mx, sy - my
        region = (rx, ry, rx + sw, ry + sh)
        try:
            frame = self._camera.grab(region=region)
            if frame is not None:
                return frame
        except Exception:
            pass
        return None

    def _grab_mss(self, sx: int, sy: int, sw: int, sh: int) -> np.ndarray:
        """mss 抓取 (屏幕绝对坐标)。"""
        monitor = {"left": sx, "top": sy, "width": sw, "height": sh}
        sct_img = self._sct.grab(monitor)
        # mss 返回 BGRA, 转换为 BGR numpy, 并复制为连续内存
        frame = np.array(sct_img, dtype=np.uint8)
        frame = frame[:, :, :3].copy()  # 去掉 alpha, 复制保证连续性
        return frame

    # ------------------------------------------------------------------
    # 便捷区域抓取
    # ------------------------------------------------------------------

    def capture_hand(self) -> np.ndarray:
        cfg = self._config
        return self.capture_region(cfg.hand_left, cfg.hand_top, cfg.hand_width, cfg.hand_height)

    def capture_river(self) -> np.ndarray:
        cfg = self._config
        return self.capture_region(cfg.river_left, cfg.river_top, cfg.river_width, cfg.river_height)

    def capture_discard_indicator(self) -> np.ndarray:
        cfg = self._config
        return self.capture_region(cfg.discard_indicator_left, cfg.discard_indicator_top, cfg.discard_indicator_width, cfg.discard_indicator_height)

    def capture_action_area(self) -> np.ndarray:
        cfg = self._config
        return self.capture_region(cfg.action_left, cfg.action_top, cfg.action_width, cfg.action_height)

    def capture_opponent_areas(self, player: str) -> dict[str, np.ndarray]:
        cfg = self._config
        mapping = {
            "上": {
                "hand": (cfg.upper_hand_left, cfg.upper_hand_top, cfg.upper_hand_width, cfg.upper_hand_height),
                "river": (cfg.upper_river_left, cfg.upper_river_top, cfg.upper_river_width, cfg.upper_river_height),
            },
            "对": {
                "hand": (cfg.opposite_hand_left, cfg.opposite_hand_top, cfg.opposite_hand_width, cfg.opposite_hand_height),
                "river": (cfg.opposite_river_left, cfg.opposite_river_top, cfg.opposite_river_width, cfg.opposite_river_height),
            },
            "下": {
                "hand": (cfg.lower_hand_left, cfg.lower_hand_top, cfg.lower_hand_width, cfg.lower_hand_height),
                "river": (cfg.lower_river_left, cfg.lower_river_top, cfg.lower_river_width, cfg.lower_river_height),
            },
        }
        entry = mapping.get(player)
        if entry is None:
            raise ValueError(f"未知对手: {player}")
        frames: dict[str, np.ndarray] = {}
        for area_name, (l, t, w, h) in entry.items():
            try:
                frames[area_name] = self.capture_region(l, t, w, h)
            except CaptureError:
                frames[area_name] = np.zeros((1, 1, 3), dtype=np.uint8)
        return frames


# ---------------------------------------------------------------------------
# 校准工具
# ---------------------------------------------------------------------------

def _auto_detect_hand_region(full: np.ndarray) -> tuple[float, float, float, float]:
    """自动检测手牌区域在窗口中的相对坐标。

    策略:
      1. 用较宽的扫描条带 (12% 窗高) 在窗口下半部找列方差最大的行
      2. 精确找上边界
      3. 从下往上扫找下边界（牌面纹理消失处）
      4. 左右边界按暗像素列确定
      5. 高度兜底 >= 8% 窗高

    Returns (left, top, width, height) as fractions of window size.
    """
    import cv2

    h, w = full.shape[:2]
    gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)

    # ---- 第 1 遍: 粗扫找手牌所在的 y ----
    scan_height = int(h * 0.12)  # 12% 窗高, 足够覆盖整张牌
    step = max(4, scan_height // 6)
    best_score = 0.0
    best_y = int(h * 0.50)
    margin_x = int(w * 0.05)

    for y_start in range(int(h * 0.30), int(h * 0.80), step):
        y_end = min(h, y_start + scan_height)
        strip = gray[y_start:y_end, margin_x:w - margin_x]

        # 列方向的亮度方差 (牌面间隔 → 方差大)
        col_means = strip.mean(axis=0)
        col_var = col_means.var()

        # 暗像素密度
        dark_ratio = (strip < 100).mean()

        score = col_var * dark_ratio
        if score > best_score:
            best_score = score
            best_y = y_start

    # ---- 第 2 遍: 从上往下精确定位手牌上边界 ----
    top_y = best_y
    for y in range(best_y, max(0, best_y - int(h * 0.10)), -1):
        row = gray[y, margin_x:w - margin_x]
        dark_ratio = (row < 90).mean()
        # 方差大 = 牌面纹理存在
        col_var = row[::3].var()  # 采样加速
        if dark_ratio < 0.015 and col_var < 20:
            top_y = y + 1
            break

    # ---- 第 3 遍: 从下往上精确定位手牌下边界 ----
    # 从 best_y + scan_height 往上扫, 找到纹理突然出现的行
    bot_y = best_y + scan_height
    search_start = min(h - 5, best_y + scan_height + int(h * 0.05))
    for y in range(search_start, top_y + int(h * 0.05), -1):
        row = gray[y, margin_x:w - margin_x]
        dark_ratio = (row < 90).mean()
        col_var = row[::3].var()
        if dark_ratio > 0.03 or col_var > 30:
            bot_y = y + 1
            break

    # ---- 兜底 ----
    detected_h = bot_y - top_y
    min_h = int(h * 0.08)
    if detected_h < min_h:
        bot_y = min(h, top_y + min_h)

    # ---- 左右边界 ----
    left_x = margin_x
    right_x = w - margin_x
    for x in range(margin_x, int(w * 0.25)):
        col = gray[top_y:bot_y, x]
        if col.mean() < 80:
            left_x = max(0, x - 5)
            break
    for x in range(w - margin_x, int(w * 0.65), -1):
        col = gray[top_y:bot_y, x]
        if col.mean() < 80:
            right_x = min(w, x + 5)
            break

    left = max(0.0, left_x / w)
    top = max(0.0, top_y / h)
    width = min(1.0 - left, (right_x - left_x) / w)
    height = min(1.0 - top, (bot_y - top_y) / h)

    return (left, top, width, height)


def calibrate() -> CaptureConfig:
    """使用 PrintWindow 捕获窗口并自动检测牌面区域坐标（最小化也可用）。"""
    try:
        import cv2
    except ImportError:
        print("警告: opencv-python 未安装，无法显示校准截图。")
        return DEFAULT_CONFIG

    cap = Capturer()
    cap.start()

    # 优先 PrintWindow
    full = cap._grab_printwindow()
    if full is not None:
        engine = "printwindow"
    else:
        engine = cap.engine
        full = cap.capture_region(0.0, 0.0, 1.0, 1.0)

    full = full.copy()
    h, w = full.shape[:2]
    print(f"捕获引擎: {engine}")
    print(f"窗口客户区尺寸: {w} x {h}")

    cv2.imwrite("calibration_full.png", full)

    # 自动检测手牌区域
    hand_left, hand_top, hand_width, hand_height = _auto_detect_hand_region(full)
    print(f"自动检测手牌区域: left={hand_left:.3f} top={hand_top:.3f} w={hand_width:.3f} h={hand_height:.3f}")

    # 保存手牌区域截图供验证
    x1 = int(hand_left * w)
    y1 = int(hand_top * h)
    x2 = int((hand_left + hand_width) * w)
    y2 = int((hand_top + hand_height) * h)
    hand_crop = full[y1:y2, x1:x2]
    cv2.imwrite("debug_hand_detected.png", hand_crop)
    print("手牌区域截图: debug_hand_detected.png (请确认是否正确)")

    # 构建配置
    cfg = CaptureConfig(
        hand_left=hand_left,
        hand_top=hand_top,
        hand_width=hand_width,
        hand_height=hand_height,
    )

    # 画叠加图
    regions = [
        ("手牌", cfg.hand_left, cfg.hand_top, cfg.hand_width, cfg.hand_height, (0, 255, 0)),
    ]
    for name, l, t, rw, rh, color in regions:
        x1 = int(l * w)
        y1 = int(t * h)
        x2 = int((l + rw) * w)
        y2 = int((t + rh) * h)
        cv2.rectangle(full, (x1, y1), (x2, y2), color, 2)
        cv2.putText(full, name, (x1 + 4, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imwrite("calibration_overlay.png", full)
    print("区域叠加图: calibration_overlay.png")

    _save_config(cfg)
    print(f"配置已保存到 {CONFIG_FILE}")

    cap.stop()
    return cfg
