"""屏幕捕获模块。

优先使用 mss (屏幕绝对坐标, 兼容性最好)，
dxcam (DXGI, 高性能) 作为可选加速器。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

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
    """屏幕捕获器。

    优先 dxcam (高性能)，不可用时自动降级为 mss。
    两者失败则报错。

    用法:
        cap = Capturer()
        cap.start()
        frame = cap.capture_hand()
        cap.stop()
    """

    def __init__(
        self,
        window_title: Optional[str] = None,
        title_keywords: Optional[list[str]] = None,
        config: Optional[CaptureConfig] = None,
        force_mss: bool = False,
        debug: bool = False,
    ):
        self._title_keywords = title_keywords
        self._config = config or DEFAULT_CONFIG
        self._force_mss = force_mss
        self._debug = debug

        self._camera: Any = None
        self._sct: Any = None  # mss instance
        self._hwnd: Optional[int] = None
        self._window_rect: Optional[tuple[int, int, int, int]] = None
        self._monitor_offset: tuple[int, int] = (0, 0)  # dxcam 用
        self._started = False
        self._engine: str = ""

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """初始化捕获引擎并定位麻将窗口。"""
        keywords = self._title_keywords or ["腾讯", "麻将", "QQGame", "Mahjong", "Tencent"]
        found = find_mahjong_window(keywords)
        if found is None:
            raise CaptureError(f"未找到麻将游戏窗口。搜索关键词: {keywords}")

        self._hwnd = found[0]
        self._window_rect = (found[1], found[2], found[3], found[4])

        # 初始化 mss (始终可用作降级)
        if _MSS_AVAILABLE:
            self._sct = mss.mss()

        # 初始化 dxcam
        use_dxcam = _USE_DXCAM and not self._force_mss
        if use_dxcam:
            try:
                # 找到窗口所在的显示器偏移
                self._monitor_offset = self._find_monitor_for_window(found[1:])
                self._camera = dxcam.create(region=self._monitor_region(), output_color="BGR")
                self._engine = "dxcam"
            except Exception as exc:
                if self._debug:
                    print(f"[capture] dxcam 初始化失败: {exc}，降级为 mss")
                self._camera = None

        if self._camera is None and self._sct is not None:
            self._engine = "mss"
        elif self._camera is None:
            raise CaptureError("没有可用的截屏引擎。请安装: pip install mss dxcam")

        self._started = True
        return True

    def stop(self) -> None:
        self._started = False
        self._camera = None
        self._sct = None

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

    def capture_region(self, left: float, top: float, width: float, height: float) -> np.ndarray:
        """抓取窗口内相对坐标区域 (0.0-1.0) 的截图，返回 BGR numpy 数组。"""
        if not self._started:
            raise CaptureError("Capturer 未启动，请先调用 start()。")

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

        # 边界保护
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

def calibrate() -> CaptureConfig:
    """截取窗口全图并标注各区域，用于手动校准坐标。"""
    try:
        import cv2
    except ImportError:
        print("警告: opencv-python 未安装，无法显示校准截图。")
        return DEFAULT_CONFIG

    cap = Capturer()
    cap.start()
    print(f"捕获引擎: {cap.engine}")

    full = cap.capture_region(0.0, 0.0, 1.0, 1.0)
    # 确保是连续内存的副本，cv2 画图需要
    full = full.copy()
    h, w = full.shape[:2]
    print(f"游戏窗口尺寸: {w} x {h}")

    cv2.imwrite("calibration_full.png", full)
    print("全图已保存为 calibration_full.png")

    cfg = DEFAULT_CONFIG
    regions = [
        ("手牌", cfg.hand_left, cfg.hand_top, cfg.hand_width, cfg.hand_height, (0, 255, 0)),
        ("牌河", cfg.river_left, cfg.river_top, cfg.river_width, cfg.river_height, (255, 0, 0)),
        ("出牌", cfg.discard_indicator_left, cfg.discard_indicator_top, cfg.discard_indicator_width, cfg.discard_indicator_height, (0, 0, 255)),
        ("动作", cfg.action_left, cfg.action_top, cfg.action_width, cfg.action_height, (255, 255, 0)),
    ]
    for name, l, t, rw, rh, color in regions:
        x1 = int(l * w)
        y1 = int(t * h)
        x2 = int((l + rw) * w)
        y2 = int((t + rh) * h)
        cv2.rectangle(full, (x1, y1), (x2, y2), color, 2)
        cv2.putText(full, name, (x1 + 4, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imwrite("calibration_overlay.png", full)
    print("区域叠加图已保存为 calibration_overlay.png，请对照检查区域是否准确。")

    cap.stop()
    return cfg
