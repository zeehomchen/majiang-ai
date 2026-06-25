"""
手动切牌程序！
你在图上用鼠标点一下每张牌的位置！
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# 添加项目路径
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2


# 全局变量
click_points = []
current_image = None
clone = None


def mouse_click(event, x, y, flags, param):
    global click_points, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        # 画点
        cv2.circle(clone, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(
            clone,
            str(len(click_points)),
            (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        print(f"点 {len(click_points)}: ({x}, {y})")


def main():
    global click_points, current_image, clone

    print("=" * 60)
    print("手动切牌程序")
    print("=" * 60)

    # 检查文件
    output_dir = ROOT / "tile_templates"
    hand_path = output_dir / "hand_region.png"

    if not hand_path.exists():
        print(f"❌ 找不到 {hand_path}")
        print("请先运行 simple_capture.py 按 s 保存截图！")
        return

    print(f"✅ 加载手牌图片: {hand_path}")

    # 读取图片
    img = cv2.imread(str(hand_path))
    if img is None:
        print("❌ 打不开图片！")
        return

    current_image = img
    clone = img.copy()

    print("\n操作说明：")
    print("  1. 在每张牌的位置点击鼠标左键（按顺序从左到右）")
    print("  2. 按 'c' 确认并切牌")
    print("  3. 按 'r' 重置点击")
    print("  4. 按 'q' 退出\n")

    # 设置窗口和回调
    cv2.namedWindow("Click on each tile (left to right)")
    cv2.setMouseCallback("Click on each tile (left to right)", mouse_click)

    while True:
        cv2.imshow("Click on each tile (left to right)", clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n👋 退出")
            break
        elif key == ord('r'):
            print("\n🔄 重置点击")
            click_points = []
            clone = current_image.copy()
        elif key == ord('c'):
            if len(click_points) < 5:
                print(f"\n⚠️  只有 {len(click_points)} 个点，太少了！请多点几个！")
                continue

            print(f"\n✅ 切分 {len(click_points)} 张牌...")

            # 按x坐标排序
            click_points.sort(key=lambda p: p[0])

            # 计算牌宽度（取平均间距）
            if len(click_points) >= 2:
                avg_gap = sum(
                    click_points[i+1][0] - click_points[i][0]
                    for i in range(len(click_points)-1)
                ) / (len(click_points)-1)
            else:
                avg_gap = 50

            half_w = int(avg_gap * 0.5)

            # 切牌
            h, w = current_image.shape[:2]
            for i, (x, y) in enumerate(click_points):
                x1 = max(0, x - half_w)
                x2 = min(w, x + half_w)
                crop = current_image[:, x1:x2]

                if crop.size > 0:
                    save_path = output_dir / f"tile_{i+1:02d}.png"
                    cv2.imwrite(str(save_path), crop)
                    print(f"  保存: {save_path}")

            print("\n✅ 完成！")
            print(f"现在请查看 {output_dir} 目录，把 tile_01.png 等重命名为牌面编码！")
            print("例如: tile_01.png -> 1m.png, tile_02.png -> 2p.png 这样！")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
