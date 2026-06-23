"""生成 13 张牌对齐工具 + 碰牌区检测关闭。"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import cv2, base64
from majiang_ai.capture import Capturer

os = __import__("os"); os.chdir(str(ROOT))

cap = Capturer(); cap.start()
full = cap._grab_printwindow()
if full is None:
    full = cap.capture_region(0, 0, 1, 1)
cap.stop()

fh, fw = full.shape[:2]
print(f"窗口: {fw}x{fh}")

# 用已保存配置
cfg = {"left": 0.0926, "top": 0.8417, "width": 0.9018, "height": 0.146}
try:
    saved = json.loads(Path("calibration_config.json").read_text(encoding="utf-8"))
    cfg.update(saved.get("hand", {}))
except: pass

x1 = int(cfg["left"] * fw); y1 = int(cfg["top"] * fh)
x2 = int((cfg["left"] + cfg["width"]) * fw); y2 = int((cfg["top"] + cfg["height"]) * fh)
hand = full[y1:y2, x1:x2]
cv2.imwrite("hand_current.png", hand)

# 13 张牌，等分起始线
N = 13
tile_w = hand.shape[1] / N
default_lines = [int((i + 1) * tile_w) for i in range(N)]

_, buf = cv2.imencode(".jpg", hand, [cv2.IMWRITE_JPEG_QUALITY, 85])
b64 = base64.b64encode(buf).decode()

hand_area_json = json.dumps({"left": cfg["left"], "top": cfg["top"], "width": cfg["width"], "height": cfg["height"]})

html = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>""" + str(N) + r"""牌对齐</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#eee;font:14px system-ui;display:flex;flex-direction:column;height:100vh}
.toolbar{background:#16213e;padding:8px 12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px}
.toolbar button{border:none;border-radius:3px;padding:5px 14px;cursor:pointer;font-weight:600;font-size:12px}
.toolbar .sep{width:1px;height:20px;background:#333;margin:0 4px}
.btn-done{background:#e94560;color:#fff}
.wrap{flex:1;overflow:auto;display:flex;justify-content:center;align-items:center;background:#0a0a1a;padding:10px}
canvas{cursor:ew-resize}
.status{background:#16213e;padding:6px 12px;font-size:11px;color:#888;text-align:center}
.status kbd{background:#333;padding:1px 5px;border-radius:3px;margin:0 2px}
</style></head><body>
<div class="toolbar">
<button class="btn-done" onclick="copyResult()">复制坐标(调好后点)</button>
<span class="sep"></span>
<span style="color:#5c7" id="info"></span>
</div>
<div class="wrap"><canvas id="cv"></canvas></div>
<div class="status">
拖拽<span style="color:#0f0">绿色</span>/<span style="color:#ff0">黄色</span>竖线对齐牌右边界 |
<kbd>Tab</kbd>切换线 | <kbd>←→</kbd>微调 | 调好后复制坐标发给我
</div>
<script>
var img = null, W = 0, H = 0, S = 1;
var lines = """ + str(default_lines) + """;
var activeLine = -1;
var cv = document.getElementById("cv"), ctx = cv.getContext("2d");

img = new Image();
img.onload = function() {
  W = img.width; H = img.height;
  S = Math.min((window.innerWidth-40)/W, (window.innerHeight-120)/H, 1);
  cv.width = W * S; cv.height = H * S;
  draw();
};
img.src = "data:image/jpeg;base64,""" + b64 + """";

function draw() {
  if (!img) return;
  ctx.drawImage(img, 0, 0, W*S, H*S);
  var prev = 0;
  for (var i = 0; i < lines.length; i++) {
    var rx = lines[i] * S;
    var color = i === activeLine ? "#ff0" : "#0f0";
    ctx.strokeStyle = color; ctx.lineWidth = i === activeLine ? 3 : 2;
    ctx.beginPath(); ctx.moveTo(rx, 0); ctx.lineTo(rx, H*S); ctx.stroke();
    ctx.fillStyle = color; ctx.font = "bold 12px monospace"; ctx.textAlign = "center";
    ctx.fillText(i+1, (prev*S + rx)/2, H*S - 4);
    prev = lines[i];
  }
  var widths = [lines[0]];
  for (var i = 1; i < lines.length; i++) { widths.push(lines[i]-lines[i-1]); }
  document.getElementById("info").textContent = "宽度(px): " + widths.join(" | ");
}

function findLine(ex) {
  var best = -1, bestDist = 999;
  for (var i = 0; i < lines.length; i++) {
    var d = Math.abs(ex - lines[i]*S);
    if (d < 15 && d < bestDist) { best = i; bestDist = d; }
  }
  return best;
}

cv.addEventListener("mousedown", function(e) {
  activeLine = findLine(e.offsetX);
  if (activeLine >= 0) {
    cv.addEventListener("mousemove", onDrag);
    cv.addEventListener("mouseup", onUp);
  }
  draw();
});

function onDrag(e) {
  if (activeLine < 0) return;
  var newX = Math.round(e.offsetX / S);
  var minX = activeLine > 0 ? lines[activeLine-1] + 20 : 5;
  var maxX = activeLine < lines.length-1 ? lines[activeLine+1] - 20 : W - 5;
  lines[activeLine] = Math.max(minX, Math.min(maxX, newX));
  draw();
}

function onUp() {
  cv.removeEventListener("mousemove", onDrag);
  cv.removeEventListener("mouseup", onUp);
}

document.addEventListener("keydown", function(e) {
  if (!img) return;
  if (e.key === "Tab") { e.preventDefault(); activeLine = (activeLine + 1) % lines.length; draw(); return; }
  if (activeLine < 0) return;
  var step = e.shiftKey ? 10 : 1;
  if (e.key === "ArrowLeft") {
    var minX = activeLine > 0 ? lines[activeLine-1] + 20 : 5;
    lines[activeLine] = Math.max(minX, lines[activeLine] - step);
  } else if (e.key === "ArrowRight") {
    var maxX = activeLine < lines.length-1 ? lines[activeLine+1] - 20 : W - 5;
    lines[activeLine] = Math.min(maxX, lines[activeLine] + step);
  } else return;
  e.preventDefault(); draw();
});

function copyResult() {
  var coords = [];
  var prev = 0;
  for (var i=0;i<lines.length;i++) {
    coords.push({left: prev, right: lines[i]});
    prev = lines[i];
  }
  var handArea = """ + hand_area_json + r""";
  navigator.clipboard.writeText(JSON.stringify({tiles: coords, hand_area: handArea}))
    .then(function(){alert("已复制！发给AI助手，并告诉我13张牌分别是什么");});
}
</script></body></html>"""

Path("tuner_" + str(N) + ".html").write_text(html, encoding="utf-8")
print(f"已生成 tuner_{N}.html ({len(html)} bytes)")
print(f"共 {N} 条可拖拽竖线")
