"""最终模板库建立 — 用户已精调对齐，切出 13 张模板。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import cv2, os, shutil, json
from majiang_ai.capture import Capturer

os.chdir(str(ROOT))

TILE_DATA = json.loads(r'''{"tiles":[{"left":0,"right":88},{"left":88,"right":177},{"left":177,"right":266},{"left":266,"right":356},{"left":356,"right":447},{"left":447,"right":535},{"left":535,"right":625},{"left":625,"right":713},{"left":713,"right":804},{"left":804,"right":893},{"left":893,"right":982},{"left":982,"right":1073},{"left":1073,"right":1190}],"hand_area":{"left":0.0926,"top":0.8417,"width":0.9018,"height":0.146}}''')
LABELS = ["3m","4m","6m","8m","1s","2s","6s","9s","2p","4p","5z","6z","7z"]

# 截图
cap = Capturer(); cap.start()
full = cap._grab_printwindow()
if full is None: full = cap.capture_region(0, 0, 1, 1)
cap.stop()
fh, fw = full.shape[:2]

ha = TILE_DATA["hand_area"]
x1 = int(ha["left"] * fw); y1 = int(ha["top"] * fh)
y2 = int((ha["top"] + ha["height"]) * fh)
print(f"窗口:{fw}x{fh} 手牌起点:({x1},{y1})")

# 保存配置
Path("calibration_config.json").write_text(json.dumps({"hand": ha}, indent=2))

# 切图
out = "tile_templates"
shutil.rmtree(out, ignore_errors=True)
os.makedirs(out)

for i, t in enumerate(TILE_DATA["tiles"]):
    lx = x1 + t["left"]
    rx = x1 + t["right"]
    py1 = max(0, y1 - 6)
    py2 = min(fh, y2 + 6)
    crop = full[py1:py2, lx:rx]
    cv2.imwrite(f"{out}/slot_{i+1:02d}.png", crop)
    print(f"  slot_{i+1:02d}: ({lx},{py1})-({rx},{py2}) {crop.shape[1]}x{crop.shape[0]} -> {LABELS[i]}")

# 写标签
lines = []
for i, lbl in enumerate(LABELS):
    lines.append(f"## 牌#{i+1}: {lbl}")
    lines.append(lbl)
Path("tile_labels.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

# 写 tile_map
tile_map = [f"slot_{i+1:02d}.png={lbl}" for i, lbl in enumerate(LABELS)]
Path("tile_map.txt").write_text("\n".join(tile_map) + "\n", encoding="utf-8")

print(f"\n{len(LABELS)} 张模板 -> {out}/")
print("标签已保存: tile_labels.txt, tile_map.txt")
