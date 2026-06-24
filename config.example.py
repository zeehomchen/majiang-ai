# 示例配置文件
# 复制为 config.py 并修改

# ======================================
# 采集配置
# ======================================
CAPTURE_MODE = "auto"  # "virtual_camera", "print_window", "mss", "auto"
VIRTUAL_CAMERA_INDEX = 0

# ======================================
# 悬浮窗配置
# ======================================
OVERLAY_ENABLED = True
ALWAYS_ON_TOP = True
TRANSPARENT_BACKGROUND = True
CLICK_THROUGH = True
ANTI_SCREENSHOT = True
FOLLOW_INTERVAL_MS = 500

# ======================================
# 算牌算法配置
# ======================================
SPEED_PRIORITY = 0.6
SAFETY_PRIORITY = 0.3
FAN_PRIORITY = 0.1

# ======================================
# YOLO配置
# ======================================
YOLO_ENABLED = False
YOLO_MODEL_PATH = None
YOLO_CONF_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45

# ======================================
# 调试配置
# ======================================
DEBUG_MODE = False
