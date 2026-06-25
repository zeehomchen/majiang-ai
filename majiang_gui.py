"""
广东推倒胡算牌器 - 美观版图形界面！
美观的配色、现代化设计！
"""

from __future__ import annotations

import sys
import json
import re
import threading
from pathlib import Path
from tkinter import *
from tkinter import ttk, messagebox
from tkinter.font import Font

# 添加项目路径
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np
from PIL import Image, ImageTk
from majiang_ai import evaluate_hand
from match_engine import MatchEngine


# 颜色主题
COLORS = {
    "bg": "#f0f2f5",
    "card_bg": "#ffffff",
    "primary": "#2563eb",
    "primary_hover": "#1d4ed8",
    "success": "#10b981",
    "success_hover": "#059669",
    "danger": "#ef4444",
    "danger_hover": "#dc2626",
    "warning": "#f59e0b",
    "text": "#1f2937",
    "text_secondary": "#6b7280",
    "border": "#e5e7eb",
}


# 中文牌名映射
CODE_TO_NAME = {
    "1m": "一万", "2m": "二万", "3m": "三万", "4m": "四万", "5m": "五万",
    "6m": "六万", "7m": "七万", "8m": "八万", "9m": "九万",
    "1p": "一筒", "2p": "二筒", "3p": "三筒", "4p": "四筒", "5p": "五筒",
    "6p": "六筒", "7p": "七筒", "8p": "八筒", "9p": "九筒",
    "1s": "一条", "2s": "二条", "3s": "三条", "4s": "四条", "5s": "五条",
    "6s": "六条", "7s": "七条", "8s": "八条", "9s": "九条",
    "1z": "东", "2z": "南", "3z": "西", "4z": "北",
    "5z": "白板", "6z": "发财", "7z": "红中",
}

NAME_TO_CODE = {v: k for k, v in CODE_TO_NAME.items()}


# 所有牌
ALL_TILES = list(CODE_TO_NAME.keys())


class ModernButton(Button):
    """自定义美观按钮"""
    def __init__(self, master, text="", bg_color=COLORS["primary"], fg_color="white", hover_color=COLORS["primary_hover"], padx=15, pady=8, **kwargs):
        self.default_bg = bg_color
        self.hover_bg = hover_color
        # 提取可能冲突的参数
        button_kwargs = kwargs.copy()
        font_val = button_kwargs.pop('font', ("Microsoft YaHei", 11, "bold"))
        
        super().__init__(
            master,
            text=text,
            bg=bg_color,
            fg=fg_color,
            activebackground=hover_color,
            activeforeground=fg_color,
            relief=FLAT,
            padx=padx,
            pady=pady,
            font=font_val,
            cursor="hand2",
            **button_kwargs
        )
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self.config(bg=self.hover_bg)

    def on_leave(self, e):
        self.config(bg=self.default_bg)


class MahjongApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🀄 广东推倒胡算牌器")
        self.root.geometry("1400x900")
        self.root.configure(bg=COLORS["bg"])
        self.root.minsize(1200, 800)

        # 状态
        self.cap = None
        self.hand_region = None
        self.templates = {}
        self.engine = MatchEngine()  # 新匹配引擎
        self.running = False
        self.video_thread = None
        self.last_frame = None

        # 画框状态
        self.drawing = False
        self.start_x, self.start_y = 0, 0
        self.current_box = None

        # 模式
        self.mode = "idle"  # idle, calibrate, capture, auto, manual

        # 图片显示参数，用于坐标转换
        self.img_x_offset = 0
        self.img_y_offset = 0
        self.img_scale = 1.0

        # 主线程调度
        self._after_id = None

        # 加载配置
        self.load_calibration()
        self.load_templates()

        # 界面布局
        self.create_widgets()

    def create_widgets(self):
        # 主容器
        main_container = Frame(self.root, bg=COLORS["bg"])
        main_container.pack(fill=BOTH, expand=True, padx=20, pady=20)

        # ========== 顶部标题栏 ==========
        header_frame = Frame(main_container, bg=COLORS["bg"])
        header_frame.pack(fill=X, pady=(0, 20))

        title_label = Label(
            header_frame,
            text="🀄 广东推倒胡算牌器",
            font=("Microsoft YaHei", 24, "bold"),
            bg=COLORS["bg"],
            fg=COLORS["text"]
        )
        title_label.pack(side=LEFT)

        # ========== 按钮栏 ==========
        toolbar = Frame(main_container, bg=COLORS["bg"])
        toolbar.pack(fill=X, pady=(0, 20))

        # 开始算牌 - 大按钮
        self.btn_play = ModernButton(
            toolbar,
            text="🎯  开始算牌",
            bg_color="#059669",
            hover_color="#047857",
            padx=24,
            pady=12,
            font=("Microsoft YaHei", 14, "bold"),
            command=self.start_auto
        )
        self.btn_play.pack(side=LEFT, padx=(0, 20))

        # 分隔线
        sep = Frame(toolbar, bg=COLORS["border"], width=2, height=36)
        sep.pack(side=LEFT, padx=(0, 20))
        sep.pack_propagate(False)

        # 按钮组
        self.btn_calibrate = ModernButton(
            toolbar,
            text="📐  标定区域",
            bg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            command=self.start_calibrate
        )
        self.btn_calibrate.pack(side=LEFT, padx=(0, 10))

        self.btn_capture = ModernButton(
            toolbar,
            text="📷  采集模板",
            bg_color="#7c3aed",
            hover_color="#6d28d9",
            command=self.start_capture
        )
        self.btn_capture.pack(side=LEFT, padx=10)

        self.btn_auto = ModernButton(
            toolbar,
            text="🤖  自动识别",
            bg_color=COLORS["success"],
            hover_color=COLORS["success_hover"],
            command=self.start_auto
        )
        self.btn_auto.pack(side=LEFT, padx=10)

        self.btn_manual = ModernButton(
            toolbar,
            text="✏️  手动输入",
            bg_color=COLORS["warning"],
            hover_color="#d97706",
            command=self.start_manual
        )
        self.btn_manual.pack(side=LEFT, padx=10)

        self.btn_stop = ModernButton(
            toolbar,
            text="⏹️  停止",
            bg_color=COLORS["danger"],
            hover_color=COLORS["danger_hover"],
            command=self.stop_current
        )
        self.btn_stop.pack(side=LEFT, padx=10)

        # ========== 内容区域 ==========
        content_frame = Frame(main_container, bg=COLORS["bg"])
        content_frame.pack(fill=BOTH, expand=True)

        # 左边视频区域
        left_frame = Frame(content_frame, bg=COLORS["bg"])
        left_frame.pack(side=LEFT, fill=BOTH, expand=True)

        # 视频卡片
        video_card = Frame(left_frame, bg=COLORS["card_bg"], relief=SOLID, bd=0)
        video_card.pack(fill=BOTH, expand=True)
        video_card.config(highlightbackground=COLORS["border"], highlightcolor=COLORS["border"], highlightthickness=1)

        # 视频容器用Canvas，更稳定
        self.video_canvas = Canvas(
            video_card,
            bg="#1f2937",
            highlightthickness=0
        )
        self.video_canvas.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # 初始文字
        self.video_canvas.create_text(
            400, 300,
            text="点击上方按钮开始",
            fill="#9ca3af",
            font=("Microsoft YaHei", 14)
        )

        # 右边控制面板 - 添加滚动条
        right_frame = Frame(content_frame, bg=COLORS["bg"], width=360)
        right_frame.pack(side=RIGHT, fill=Y, padx=(20, 0))
        
        # 滚动画布
        self.right_canvas = Canvas(right_frame, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = Scrollbar(right_frame, orient=VERTICAL, command=self.right_canvas.yview)
        self.scrollable_right_frame = Frame(self.right_canvas, bg=COLORS["bg"])
        
        self.scrollable_right_frame.bind(
            "<Configure>",
            lambda e: self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))
        )

        self.right_canvas.bind(
            "<Configure>",
            lambda e: self.right_canvas.itemconfigure(self.right_canvas_window, width=e.width)
        )

        self.right_canvas_window = self.right_canvas.create_window(
            (0, 0),
            window=self.scrollable_right_frame,
            anchor="nw",
        )
        self.right_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.right_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # 绑定鼠标滚轮
        self.right_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # ========== 模板状态卡片 ==========
        tpl_card = Frame(self.scrollable_right_frame, bg=COLORS["card_bg"], relief=SOLID, bd=0)
        tpl_card.pack(fill=X, pady=(0, 15))
        tpl_card.config(highlightbackground=COLORS["border"], highlightcolor=COLORS["border"], highlightthickness=1)

        # 卡片标题
        tpl_header = Frame(tpl_card, bg=COLORS["card_bg"])
        tpl_header.pack(fill=X, padx=15, pady=(15, 5))

        Label(
            tpl_header,
            text="📦  模板状态",
            font=("Microsoft YaHei", 13, "bold"),
            bg=COLORS["card_bg"],
            fg=COLORS["text"]
        ).pack(side=LEFT)

        self.btn_check = ModernButton(
            tpl_header,
            text="刷新",
            bg_color=COLORS["border"],
            fg_color=COLORS["text"],
            hover_color="#d1d5db",
            padx=10,
            pady=3,
            font=("Microsoft YaHei", 9),
            command=self.update_template_status
        )
        self.btn_check.pack(side=RIGHT)

        # 模板内容
        self.tpl_text = Text(
            tpl_card,
            height=12,
            wrap=WORD,
            bg=COLORS["bg"],
            fg=COLORS["text"],
            relief=FLAT,
            bd=0,
            padx=15,
            pady=15,
            font=("Microsoft YaHei", 10)
        )
        self.tpl_text.pack(fill=X, padx=15, pady=(5, 15))

        # ========== 当前状态卡片 ==========
        status_card = Frame(self.scrollable_right_frame, bg=COLORS["card_bg"], relief=SOLID, bd=0)
        status_card.pack(fill=X, pady=(0, 15))
        status_card.config(highlightbackground=COLORS["border"], highlightcolor=COLORS["border"], highlightthickness=1)

        Label(
            status_card,
            text="📊  当前状态",
            font=("Microsoft YaHei", 13, "bold"),
            bg=COLORS["card_bg"],
            fg=COLORS["text"]
        ).pack(anchor=W, padx=15, pady=(15, 5))

        self.status_label = Label(
            status_card,
            text="等待开始...",
            font=("Microsoft YaHei", 11),
            bg=COLORS["card_bg"],
            fg=COLORS["text_secondary"],
            wraplength=300,
            justify=LEFT
        )
        self.status_label.pack(anchor=W, padx=15, pady=(5, 15))

        # ========== 动态面板容器 ==========
        self.dynamic_panel = Frame(self.scrollable_right_frame, bg=COLORS["bg"])
        self.dynamic_panel.pack(fill=X, pady=(0, 10))
        
        # ========== 操作按钮区 ==========
        self.action_panel = Frame(self.scrollable_right_frame, bg=COLORS["bg"])
        self.action_panel.pack(fill=X, pady=(0, 10))
        
        # 确定按钮
        self.btn_ok = ModernButton(
            self.action_panel,
            text="✓ 确定",
            bg_color=COLORS["success"],
            hover_color=COLORS["success_hover"],
            command=self.on_ok
        )
        self.btn_ok.pack(side=LEFT, fill=X, expand=True, padx=(0, 5))
        self.btn_ok.pack_forget()  # 默认隐藏
        
        # 取消按钮
        self.btn_cancel = ModernButton(
            self.action_panel,
            text="✕ 取消",
            bg_color=COLORS["danger"],
            hover_color=COLORS["danger_hover"],
            command=self.on_cancel
        )
        self.btn_cancel.pack(side=RIGHT, fill=X, expand=True, padx=(5, 0))
        self.btn_cancel.pack_forget()  # 默认隐藏

        # ========== 算牌结果卡片 ==========
        result_card = Frame(self.scrollable_right_frame, bg=COLORS["card_bg"], relief=SOLID, bd=0)
        result_card.pack(fill=BOTH, expand=True)
        result_card.config(highlightbackground=COLORS["border"], highlightcolor=COLORS["border"], highlightthickness=1)

        Label(
            result_card,
            text="🎯  算牌结果",
            font=("Microsoft YaHei", 13, "bold"),
            bg=COLORS["card_bg"],
            fg=COLORS["text"]
        ).pack(anchor=W, padx=15, pady=(15, 5))

        self.result_text = Text(
            result_card,
            wrap=WORD,
            bg=COLORS["bg"],
            fg=COLORS["text"],
            relief=FLAT,
            bd=0,
            padx=15,
            pady=15,
            font=("Microsoft YaHei", 11)
        )
        self.result_text.pack(fill=BOTH, expand=True, padx=15, pady=(5, 15))

        # 初始状态
        self.update_template_status()

    def load_calibration(self):
        try:
            with open("calibration_config.json", "r", encoding="utf-8") as f:
                cfg = json.load(f)
                self.hand_region = cfg["hand"]
        except Exception:
            self.hand_region = None

    def save_calibration(self):
        if self.hand_region:
            with open("calibration_config.json", "w", encoding="utf-8") as f:
                json.dump({"hand": self.hand_region}, f, ensure_ascii=False, indent=2)

    def _on_mousewheel(self, event):
        # 处理鼠标滚轮
        self.right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def load_templates(self):
        self.templates = {}
        self._tpl_small = {}  # 预缩放的灰度模板，用于快速匹配
        template_dir = ROOT / "tile_templates"
        valid = re.compile(r'^[1-9][mpsz]$')
        if template_dir.exists():
            for f in template_dir.iterdir():
                if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    code = f.stem
                    if valid.match(code):
                        img = cv2.imread(str(f))
                        if img is not None:
                            self.templates[code] = img
                            # 转灰度并缩放到统一小尺寸 (40x60) 用于快速匹配
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            small = cv2.resize(gray, (40, 60))
                            self._tpl_small[code] = small
        # 同步到引擎
        self.engine.load_all()

    def update_template_status(self):
        self.load_templates()
        self.tpl_text.delete(1.0, END)

        categories = [
            ("万子", "m"),
            ("筒子", "p"),
            ("条子", "s"),
            ("字牌", "z"),
        ]

        for name, suit in categories:
            self.tpl_text.insert(END, f"【{name}】\n")
            for i in range(1, 10 if suit != 'z' else 8):
                if suit == 'z' and i == 0:
                    continue
                code = f"{i}{suit}"
                if code in self.templates:
                    self.tpl_text.insert(END, f"  ✅ {CODE_TO_NAME.get(code, code)}\n", "success")
                else:
                    self.tpl_text.insert(END, f"  ❌ {CODE_TO_NAME.get(code, code)}\n", "danger")
            self.tpl_text.insert(END, "\n")

        count = len(self.templates)
        self.tpl_text.insert(END, f"总计: {count}/34 张")

        # 颜色标签
        self.tpl_text.tag_config("success", foreground=COLORS["success"])
        self.tpl_text.tag_config("danger", foreground=COLORS["danger"])

    def clear_dynamic_panel(self):
        for widget in self.dynamic_panel.winfo_children():
            widget.destroy()

    def refresh_capture_list(self):
        if not hasattr(self, "capture_listbox"):
            return
        self.capture_listbox.delete(0, END)
        categories = [
            ("万子", "m", 1, 9),
            ("筒子", "p", 1, 9),
            ("条子", "s", 1, 9),
            ("字牌", "z", 1, 7),
        ]
        for cat_name, suit, start, end in categories:
            self.capture_listbox.insert(END, f"--- {cat_name} ---")
            for i in range(start, end + 1):
                code = f"{i}{suit}"
                name = CODE_TO_NAME.get(code, code)
                status = "✅" if code in self.templates else "❌"
                self.capture_listbox.insert(END, f"{code} - {name} {status}")

    def start_calibrate(self):
        self.stop_current()
        self.clear_dynamic_panel()
        self.mode = "calibrate"
        self.current_box = None
        self.status_label.config(text="标定模式：用鼠标画框选出手牌区域，点击「确定」保存", fg=COLORS["primary"])
        self.setup_video()
        self.bind_mouse()
        # 显示确定和取消按钮
        self.btn_ok.pack(side=LEFT, fill=X, expand=True, padx=(0, 5))
        self.btn_cancel.pack(side=RIGHT, fill=X, expand=True, padx=(5, 0))
        # 显示标定说明
        self.show_calibrate_panel()

    def show_calibrate_panel(self):
        self.clear_dynamic_panel()
        # 标定面板
        card = Frame(self.dynamic_panel, bg=COLORS["card_bg"], relief=SOLID, bd=0)
        card.pack(fill=BOTH, expand=True)
        card.config(highlightbackground=COLORS["border"], highlightcolor=COLORS["border"], highlightthickness=1)

        Label(
            card,
            text="📐 标定手牌区域",
            font=("Microsoft YaHei", 13, "bold"),
            bg=COLORS["card_bg"],
            fg=COLORS["text"]
        ).pack(anchor=W, padx=15, pady=(15, 10))

        instructions = Label(
            card,
            text="操作说明：\n\n1. 在左侧视频画面上\n2. 按住鼠标左键拖动画框\n3. 框选整个手牌区域\n4. 点击「确定」按钮保存\n\n如果OBS窗口大小变了，重新标定即可！",
            font=("Microsoft YaHei", 10),
            bg=COLORS["bg"],
            fg=COLORS["text_secondary"],
            justify=LEFT,
            anchor=W,
            padx=10,
            pady=10
        )
        instructions.pack(fill=X, padx=15, pady=(0, 10))

    def start_capture(self):
        self.stop_current()
        self.mode = "capture"
        self.current_box = None
        self.status_label.config(text="采集模式：选择牌，画框选中，点击保存", fg="#7c3aed")
        self.setup_video()
        self.bind_mouse()
        # 显示确定和取消按钮（采集模式下确定就是保存，取消就是退出）
        self.btn_ok.pack(side=LEFT, fill=X, expand=True, padx=(0, 5))
        self.btn_cancel.pack(side=RIGHT, fill=X, expand=True, padx=(5, 0))
        self.show_capture_panel()

    def show_capture_panel(self):
        self.clear_dynamic_panel()
        # 采集面板卡片
        card = Frame(self.dynamic_panel, bg=COLORS["card_bg"], relief=SOLID, bd=0)
        card.pack(fill=BOTH, expand=True)
        card.config(highlightbackground=COLORS["border"], highlightcolor=COLORS["border"], highlightthickness=1)

        Label(
            card,
            text="🎴 采集牌面模板",
            font=("Microsoft YaHei", 13, "bold"),
            bg=COLORS["card_bg"],
            fg=COLORS["text"]
        ).pack(anchor=W, padx=15, pady=(15, 10))

        # 使用说明
        instructions = Label(
            card,
            text="操作步骤：\n1. 已知缺哪张牌：先在列表里选牌，再画框，点「确定」保存\n2. 只知道这张牌还没录入：直接画框，点下面「保存到待确认」\n3. 保存后的图片会进入 tile_templates 或 tile_templates\\pending\n\n可以连续采集多张牌！",
            font=("Microsoft YaHei", 10),
            bg=COLORS["bg"],
            fg=COLORS["text_secondary"],
            justify=LEFT,
            anchor=W,
            padx=10,
            pady=10
        )
        instructions.pack(fill=X, padx=15, pady=(0, 10))

        # 滚动列表
        list_frame = Frame(card, bg=COLORS["bg"])
        list_frame.pack(fill=BOTH, expand=True, padx=15, pady=(0, 15))

        scrollbar = Scrollbar(list_frame)
        scrollbar.pack(side=RIGHT, fill=Y)

        self.capture_listbox = Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("Microsoft YaHei", 11),
            bg=COLORS["bg"],
            fg=COLORS["text"],
            selectbackground=COLORS["primary"],
            selectforeground="white",
            relief=FLAT,
            bd=0,
            height=10
        )
        self.capture_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=self.capture_listbox.yview)
        self.refresh_capture_list()

        btn_pending = ModernButton(
            card,
            text="📸 保存到待确认",
            bg_color=COLORS["warning"],
            hover_color="#d97706",
            command=self.save_pending_capture,
        )
        btn_pending.pack(fill=X, padx=15, pady=(0, 15))

    def start_auto(self):
        self.stop_current()
        self.clear_dynamic_panel()
        self.mode = "auto"
        self.last_recognized = ""
        self.status_label.config(text="算牌中：正在识别手牌...", fg=COLORS["success"])
        self.setup_video()
        # 显示自动识别状态面板
        self.show_auto_panel()

    def show_auto_panel(self):
        self.clear_dynamic_panel()
        card = Frame(self.dynamic_panel, bg=COLORS["card_bg"], relief=SOLID, bd=0)
        card.pack(fill=BOTH, expand=True)
        card.config(highlightbackground=COLORS["border"], highlightcolor=COLORS["border"], highlightthickness=1)

        Label(
            card,
            text="🤖 自动算牌中",
            font=("Microsoft YaHei", 13, "bold"),
            bg=COLORS["card_bg"],
            fg=COLORS["success"]
        ).pack(anchor=W, padx=15, pady=(15, 10))

        # 当前识别结果
        self.auto_recognized_label = Label(
            card,
            text="识别中...",
            font=("Microsoft YaHei", 12),
            bg=COLORS["card_bg"],
            fg=COLORS["text"],
            justify=LEFT,
            anchor=W
        )
        self.auto_recognized_label.pack(fill=X, padx=15, pady=(5, 5))

        Label(
            card,
            text="提示：等待几秒让识别稳定\n切换牌面后会自动更新建议",
            font=("Microsoft YaHei", 9),
            bg=COLORS["bg"],
            fg=COLORS["text_secondary"],
            justify=LEFT,
            anchor=W,
            padx=10,
            pady=10
        ).pack(fill=X, padx=15, pady=(0, 10))

    def start_manual(self):
        self.stop_current()
        self.clear_dynamic_panel()
        self.mode = "manual"
        self.status_label.config(text="手动输入模式：输入手牌（中文）", fg=COLORS["warning"])
        self.show_manual_panel()

    def show_manual_panel(self):
        card = Frame(self.dynamic_panel, bg=COLORS["card_bg"], relief=SOLID, bd=0)
        card.pack(fill=BOTH, expand=True)
        card.config(highlightbackground=COLORS["border"], highlightcolor=COLORS["border"], highlightthickness=1)

        Label(
            card,
            text="✏️  手动选择手牌",
            font=("Microsoft YaHei", 13, "bold"),
            bg=COLORS["card_bg"],
            fg=COLORS["text"]
        ).pack(anchor=W, padx=15, pady=(15, 10))

        Label(
            card,
            text="输入手牌（中文，空格分隔）:",
            font=("Microsoft YaHei", 10),
            bg=COLORS["card_bg"],
            fg=COLORS["text_secondary"]
        ).pack(anchor=W, padx=15, pady=(5, 5))

        self.manual_entry = Entry(
            card,
            font=("Microsoft YaHei", 14),
            bg=COLORS["bg"],
            fg=COLORS["text"],
            relief=SOLID,
            bd=1,
            insertbackground=COLORS["text"]
        )
        self.manual_entry.pack(fill=X, padx=15, pady=(0, 10))
        self.manual_entry.bind("<Return>", lambda e: self.on_manual_submit())

        btn = ModernButton(
            card,
            text="🎯 算牌",
            bg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"],
            command=self.on_manual_submit
        )
        btn.pack(fill=X, padx=15, pady=(10, 15))

        Label(
            card,
            text="示例：一万二万 三条 红中",
            font=("Microsoft YaHei", 9),
            bg=COLORS["card_bg"],
            fg=COLORS["text_secondary"]
        ).pack(anchor=W, padx=15, pady=(0, 15))

    def stop_current(self):
        self.running = False
        # 取消主线程调度
        if hasattr(self, '_after_id') and self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        if self.video_thread:
            self.video_thread.join(timeout=1)
            self.video_thread = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.mode = "idle"
        self.status_label.config(text="等待开始...", fg=COLORS["text_secondary"])
        self.unbind_mouse()
        # 隐藏确定取消按钮
        self.btn_ok.pack_forget()
        self.btn_cancel.pack_forget()
        # 清空Canvas
        self.video_canvas.delete("all")
        self.video_canvas.create_text(
            400, 300,
            text="点击上方按钮开始",
            fill="#9ca3af",
            font=("Microsoft YaHei", 14)
        )

    def on_ok(self):
        """确定按钮处理"""
        if self.mode == "calibrate":
            if not self.current_box:
                messagebox.showwarning("提示", "请先在视频上用鼠标画框选手牌区域！")
                return
            if hasattr(self, 'last_frame'):
                h, w = self.last_frame.shape[:2]
                x1, y1, x2, y2 = self.current_box
                self.hand_region = {
                    "left": x1 / w,
                    "top": y1 / h,
                    "width": (x2 - x1) / w,
                    "height": (y2 - y1) / h,
                }
                self.save_calibration()
                messagebox.showinfo("成功", "标定区域已保存！")
                self.status_label.config(text="标定完成！", fg=COLORS["success"])
                self.stop_current()
        elif self.mode == "capture":
            # 采集模式下，确定就是保存当前模板
            self.save_current_template()

    def on_cancel(self):
        """取消按钮处理"""
        self.stop_current()

    def setup_video(self):
        # 不调用 stop_current()，调用方已经处理了
        # 关闭已有设备
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.video_thread:
            self.video_thread = None
        # 打开相机
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                messagebox.showerror("错误", "打不开相机！")
                return

        # 设高分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # 用 after 在主线程刷新，避免闪烁
        self.running = True
        self._schedule_video_frame()

    def bind_mouse(self):
        self.video_canvas.bind("<Button-1>", self.on_mouse_down)
        self.video_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.video_canvas.bind("<Double-Button-1>", self.on_double_click)

    def unbind_mouse(self):
        try:
            self.video_canvas.unbind("<Button-1>")
            self.video_canvas.unbind("<B1-Motion>")
            self.video_canvas.unbind("<ButtonRelease-1>")
            self.video_canvas.unbind("<Double-Button-1>")
        except Exception:
            pass

    def canvas_to_img_coords(self, canvas_x, canvas_y):
        """把Canvas上的坐标转换为实际图片上的坐标"""
        img_x = (canvas_x - self.img_x_offset) / self.img_scale if self.img_scale > 0 else 0
        img_y = (canvas_y - self.img_y_offset) / self.img_scale if self.img_scale > 0 else 0
        return img_x, img_y

    def on_mouse_down(self, event):
        self.drawing = True
        # 转换坐标
        img_x, img_y = self.canvas_to_img_coords(event.x, event.y)
        self.start_x, self.start_y = img_x, img_y
        self.current_box = None

    def on_mouse_drag(self, event):
        if self.drawing:
            # 转换坐标
            img_x, img_y = self.canvas_to_img_coords(event.x, event.y)
            self.current_box = (
                min(self.start_x, img_x),
                min(self.start_y, img_y),
                max(self.start_x, img_x),
                max(self.start_y, img_y)
            )

    def on_mouse_up(self, event):
        if self.drawing:
            # 转换坐标
            img_x, img_y = self.canvas_to_img_coords(event.x, event.y)
            self.current_box = (
                min(self.start_x, img_x),
                min(self.start_y, img_y),
                max(self.start_x, img_x),
                max(self.start_y, img_y)
            )
        self.drawing = False

    def on_double_click(self, event):
        # 不再使用双击确认，用确定按钮
        pass

    def save_current_template(self):
        if self.mode != "capture":
            return
        if not self.current_box:
            messagebox.showwarning("提示", "请先在视频上用鼠标画框选中要采集的牌！")
            return
        sel = self.capture_listbox.curselection()
        if not sel:
            messagebox.showwarning("提示", "请先在列表中选择要采集的牌！")
            return
        
        sel_text = self.capture_listbox.get(sel[0])
        if "---" in sel_text:
            messagebox.showwarning("提示", "请选择具体的牌，不要选分组标题！")
            return
        
        # 提取编码
        if " - " not in sel_text:
            return
        code = sel_text.split(" - ")[0]
        
        # 检查是否是有效编码
        valid_code = re.match(r'^[1-9][mpsz]$', code)
        if not valid_code:
            messagebox.showwarning("提示", "请选择有效的牌！")
            return

        if hasattr(self, 'last_frame'):
            h, w = self.last_frame.shape[:2]
            # current_box已经是图片坐标了，不需要再转换
            x1, y1, x2, y2 = self.current_box
            rx1 = int(x1)
            ry1 = int(y1)
            rx2 = int(x2)
            ry2 = int(y2)

            tile_crop = self.last_frame[ry1:ry2, rx1:rx2].copy()
            if tile_crop.size == 0:
                messagebox.showerror("错误", "裁剪区域为空！")
                return

            template_dir = ROOT / "tile_templates"
            template_dir.mkdir(exist_ok=True)
            save_path = template_dir / f"{code}.png"
            cv2.imwrite(str(save_path), tile_crop)
            
            # 刷新
            self.load_templates()
            self.update_template_status()
            
            # 刷新采集列表
            if hasattr(self, 'capture_listbox'):
                self.capture_listbox.delete(0, END)
                categories = [
                    ("万子", "m", 1, 9),
                    ("筒子", "p", 1, 9),
                    ("条子", "s", 1, 9),
                    ("字牌", "z", 1, 7)
                ]
                for cat_name, suit, start, end in categories:
                    self.capture_listbox.insert(END, f"--- {cat_name} ---")
                    for i in range(start, end+1):
                        code_i = f"{i}{suit}"
                        name_i = CODE_TO_NAME.get(code_i, code_i)
                        status_i = "✅" if code_i in self.templates else "❌"
                        self.capture_listbox.insert(END, f"{code_i} - {name_i} {status_i}")
            
            messagebox.showinfo("成功", f"已保存 {CODE_TO_NAME.get(code, code)} ({code}) 模板！")
            self.current_box = None

    def save_pending_capture(self):
        """采集模式下：不知道牌名时，截图保存到待确认文件夹"""
        if self.mode != "capture":
            return
        if not self.current_box:
            messagebox.showwarning("提示", "请先在视频上用鼠标画框选中要截取的牌！")
            return
        if not hasattr(self, 'last_frame'):
            return
        h, w = self.last_frame.shape[:2]
        x1, y1, x2, y2 = self.current_box
        tile_crop = self.last_frame[int(y1):int(y2), int(x1):int(x2)].copy()
        if tile_crop.size == 0:
            messagebox.showerror("错误", "裁剪区域为空！")
            return

        # 存到 tile_templates/pending 目录
        pending_dir = ROOT / "tile_templates" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)

        import time
        fname = f"pending_{int(time.time() * 1000)}.png"
        save_path = pending_dir / fname
        cv2.imwrite(str(save_path), tile_crop)

        self.current_box = None
        messagebox.showinfo("截图成功", f"已保存到 pending 文件夹：\n{fname}\n\n请将其重命名为牌面编码（如 5m.png）后\n移到 tile_templates 目录下。")

    def on_manual_submit(self):
        text = self.manual_entry.get().strip()
        if not text:
            return
        codes = []
        i = 0
        while i < len(text):
            if i + 1 < len(text):
                two = text[i:i+2]
                if two in NAME_TO_CODE:
                    codes.append(NAME_TO_CODE[two])
                    i += 2
                    continue
            one = text[i:i+1]
            if one in NAME_TO_CODE:
                codes.append(NAME_TO_CODE[one])
                i += 1
            else:
                i += 1
        if not codes:
            messagebox.showwarning("提示", "未能识别任何牌！请输入如：一万二万三条")
            return

        groups = {"m": [], "p": [], "s": [], "z": []}
        for code in codes:
            num = code[0]
            suit = code[1]
            groups[suit].append(num)
        compact = ""
        for suit in ["m", "p", "s", "z"]:
            if groups[suit]:
                compact += "".join(sorted(groups[suit])) + suit

        self.run_eval(compact)

    def run_eval(self, compact):
        self.result_text.delete(1.0, END)
        self.best_discard = None
        self.best_discard_name = None
        try:
            result = evaluate_hand(compact)
            self.result_text.insert(END, f"手牌: {compact}\n\n")
            if result.options:
                best = result.options[0]
                self.best_discard = best.discard
                self.best_discard_name = CODE_TO_NAME.get(best.discard, best.discard)
                self.result_text.insert(END, "="*40 + "\n")
                self.result_text.insert(END, f"🏆 最佳切牌: {best.discard} ({self.best_discard_name})\n", "best")
                self.result_text.insert(END, f"   分数: {best.total_score:.2f}\n\n")

                self.result_text.insert(END, "所有选项:\n")
                for i, opt in enumerate(result.options[:5]):
                    prefix = "✨ " if i == 0 else f"{i+1}. "
                    self.result_text.insert(END, f"{prefix}{opt.discard} ({CODE_TO_NAME.get(opt.discard, opt.discard)}) - {opt.total_score:.2f}\n")
            else:
                self.result_text.insert(END, "没有找到切牌建议\n")
        except Exception as e:
            self.result_text.insert(END, f"分析出错: {e}\n")

        self.result_text.tag_config("best", foreground=COLORS["success"], font=("Microsoft YaHei", 12, "bold"))

    def _schedule_video_frame(self):
        """在主线程调度下一帧，避免闪烁"""
        if not self.running or not self.cap:
            return
        self._video_tick()
        if self.running:
            self._after_id = self.root.after(30, self._schedule_video_frame)

    def _video_tick(self):
        """处理一帧画面（运行在主线程）"""
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        self.last_frame = frame.copy()
        display = frame.copy()
        h, w = display.shape[:2]

        # 画标定区域
        if self.hand_region:
            x1 = int(self.hand_region["left"] * w)
            y1 = int(self.hand_region["top"] * h)
            x2 = int((self.hand_region["left"] + self.hand_region["width"]) * w)
            y2 = int((self.hand_region["top"] + self.hand_region["height"]) * h)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 3)

        # 画当前框选 - 绿色表示正在画
        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

        # 自动识别模式
        if self.mode == "auto" and self.hand_region and self.engine.templates:
            self._process_auto_recognition(display, h, w)

        # 自动模式下显示最佳切牌建议
        if self.mode == "auto" and getattr(self, 'best_discard_name', None):
            msg = f">> 切 {self.best_discard_name} <<"
            cv2.putText(display, msg, (w // 2 - 120, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # 转换显示 - Canvas 保持比例
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_rgb)
        cw = self.video_canvas.winfo_width()
        ch = self.video_canvas.winfo_height()

        if cw > 20 and ch > 20:
            img_w, img_h = img.size
            scale = min(cw / img_w, ch / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)

            try:
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            except AttributeError:
                img = img.resize((new_w, new_h), Image.LANCZOS)

            x_offset = (cw - new_w) // 2
            y_offset = (ch - new_h) // 2

            self.img_x_offset = x_offset
            self.img_y_offset = y_offset
            self.img_scale = scale
        else:
            x_offset = 0
            y_offset = 0

        imgtk = ImageTk.PhotoImage(image=img)
        self.video_canvas.delete("all")
        self.video_canvas.create_rectangle(0, 0, cw, ch, fill="#1f2937", outline="")
        self.video_canvas.create_image(x_offset, y_offset, anchor="nw", image=imgtk)
        self.video_canvas.imgtk = imgtk

    def _process_auto_recognition(self, display, h, w):
        """自动识别手牌 — 使用 MatchEngine 多策略融合"""
        self._recognize_frame_count = getattr(self, '_recognize_frame_count', 0) + 1
        if self._recognize_frame_count % 10 != 0:
            return

        x1 = int(self.hand_region["left"] * w)
        y1 = int(self.hand_region["top"] * h)
        x2 = int((self.hand_region["left"] + self.hand_region["width"]) * w)
        y2 = int((self.hand_region["top"] + self.hand_region["height"]) * h)
        hand_crop = display[y1:y2, x1:x2].copy()

        if hand_crop.size == 0:
            return
        hh, ww = hand_crop.shape[:2]

        # 使用引擎匹配整手
        recognized_list, compact = self.engine.match_hand(hand_crop)

        # 在画面上标注
        tile_w = ww // 14 if ww >= 14 else 1
        for i, code in enumerate(recognized_list):
            if code == "?":
                continue
            cx = x1 + i * tile_w + tile_w // 2
            cy = y1 + hh // 2
            name = CODE_TO_NAME.get(code, code)
            cv2.putText(display, name, (cx - 25, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 更新右侧面板
        display_str = "识别: " + " ".join(CODE_TO_NAME.get(c, c) for c in recognized_list if c != "?")
        if hasattr(self, 'auto_recognized_label'):
            valid = [c for c in recognized_list if c != "?"]
            self.auto_recognized_label.config(
                text=f"{display_str}\n编码: {compact}\n(位置: {' '.join(valid)})"
            )

        # 算牌
        if compact and compact != getattr(self, 'last_recognized', ''):
            self.last_recognized = compact
            self.run_eval(compact)

    def match_single(self, tile_img):
        """委托给 MatchEngine"""
        code, score = self.engine.match(tile_img)
        return code, score


def main():
    root = Tk()
    app = MahjongApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
