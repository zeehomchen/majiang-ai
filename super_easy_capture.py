"""
超简单版单张采集！
步骤：
1. 程序启动后显示画面
2. 先看终端，输入你要采集的牌面编码
3. 然后在画面上画框
4. 自动保存！
"""

from __future__ import annotations

import sys
import json
import os
from pathlib import Path

# 添加项目路径
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2


def load_calibration():
    try:
        with open("calibration_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg["hand"]
    except Exception:
        return None


def main():
    print("=" * 60)
    print("超简单单张牌采集！")
    print("=" * 60)
    print("\n操作说明：")
    print("  1. 先在终端输入你要采集的牌面编码")
    print("  2. 然后在画面上用鼠标画框选中该牌")
    print("  3. 松开鼠标自动保存！")
    print("  4. 继续下一张或输入 q 退出\n")
    print("编码说明：")
    print("  1m=一万, 2m=二万 ... 9m=九万")
    print("  1p=一筒, 2p=二筒 ... 9p=九筒")
    print("  1s=一条, 2s=二条 ... 9s=九条")
    print("  1z=东, 2z=南, 3z=西, 4z=北, 5z=白板, 6z=发财, 7z=红中\n")

    # 加载标定
    hand_region = load_calibration()

    # 打开相机
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 打不开相机！")
            return

    output_dir = ROOT / "tile_templates"
    output_dir.mkdir(exist_ok=True)

    # 状态
    selecting = False
    start_x, start_y = -1, -1
    end_x, end_y = -1, -1
    waiting_code = True
    current_code = None
    frame = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, start_x, start_y, end_x, end_y
        if not waiting_code and current_code:
            if event == cv2.EVENT_LBUTTONDOWN:
                selecting = True
                start_x, start_y = x, y
            elif event == cv2.EVENT_MOUSEMOVE and selecting:
                end_x, end_y = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                selecting = False
                end_x, end_y = x, y

    cv2.namedWindow("Super Simple Capture")
    cv2.setMouseCallback("Super Simple Capture", mouse_callback)

    try:
        while True:
            # 读帧
            ret, read_frame = cap.read()
            if ret:
                frame = read_frame
            if frame is None:
                cv2.waitKey(30)
                continue

            display = frame.copy()

            # 画标定区域
            if hand_region:
                h, w = display.shape[:2]
                x1 = int(hand_region["left"] * w)
                y1 = int(hand_region["top"] * h)
                x2 = int((hand_region["left"] + hand_region["width"]) * w)
                y2 = int((hand_region["top"] + hand_region["height"]) * h)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 1)

            # 画正在选的框
            if not waiting_code and selecting and start_x != -1 and end_x != -1:
                x1, x2 = sorted([start_x, end_x])
                y1, y2 = sorted([start_y, end_y])
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # 提示文字
            if waiting_code:
                cv2.putText(
                    display,
                    "Input code in terminal first!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
            else:
                cv2.putText(
                    display,
                    f"Draw a box around {current_code}, release to save!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Super Simple Capture", display)
            key = cv2.waitKey(1) & 0xFF

            # 处理输入
            if waiting_code:
                # 非阻塞方式检查输入
                import sys, select
                # Windows下简单处理
                if key == ord('q'):
                    print("\n👋 退出")
                    break

                # 这里在一个单独线程处理输入会更好，我们简化：
                # 按 'i' 进入输入
                if key == ord('i') or True:
                    # 由于OpenCV窗口和终端输入不好配合，我们用一个简单方案
                    print(f"\n请输入牌面编码（q退出）: ", end="", flush=True)
                    # 这里我们用一个简单的方式，让用户在终端输入
                    # 为了不卡住，我们用一个线程来处理
                    # 但为了简单，我们用cv2.waitKey，但最好的方案是：
                    import threading
                    def input_thread():
                        nonlocal current_code, waiting_code
                        try:
                            while True:
                                code = input().strip().lower()
                                if code == 'q':
                                    os._exit(0)
                                # 验证
                                import re
                                valid = re.compile(r'^[1-9][mpsz]$')
                                if not valid.match(code):
                                    print("❌ 无效编码！请用如 1m, 2p, 7z 的格式")
                                    print(f"请输入牌面编码（q退出）: ", end="", flush=True)
                                    continue
                                current_code = code
                                waiting_code = False
                                print(f"✅ 好的！现在在画面上画框选中 {code}！")
                                break
                        except EOFError:
                            pass

                    t = threading.Thread(target=input_thread, daemon=True)
                    t.start()
                    t.join()
                    waiting_code = False

            # 如果画完了框（松开鼠标）
            if not waiting_code and not selecting and start_x != -1 and end_x != -1:
                # 保存
                x1, x2 = sorted([start_x, end_x])
                y1, y2 = sorted([start_y, end_y])
                if (x2 - x1) > 10 and (y2 - y1) > 10 and frame is not None:
                    tile_crop = frame[y1:y2, x1:x2].copy()
                    save_path = output_dir / f"{current_code}.png"
                    cv2.imwrite(str(save_path), tile_crop)
                    print(f"✅ 已保存: {save_path}")

                    # 显示一下成功
                    success_img = display.copy()
                    cv2.putText(
                        success_img,
                        f"SAVED: {current_code}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        3
                    )
                    cv2.imshow("Super Simple Capture", success_img)
                    cv2.waitKey(500)

                # 重置
                start_x, start_y = -1, -1
                end_x, end_y = -1, -1
                waiting_code = True
                current_code = None

            if key == ord('q'):
                print("\n👋 退出")
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ 已停止")


if __name__ == "__main__":
    main()
