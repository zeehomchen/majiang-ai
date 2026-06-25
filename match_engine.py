"""
牌面匹配引擎 - 多策略融合
边缘 + 模板 + 直方图 + 多尺度 + 空间平滑
"""
from __future__ import annotations

import re
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).parent

CODE_TO_NAME = {
    "1m":"一万","2m":"二万","3m":"三万","4m":"四万","5m":"五万",
    "6m":"六万","7m":"七万","8m":"八万","9m":"九万",
    "1p":"一筒","2p":"二筒","3p":"三筒","4p":"四筒","5p":"五筒",
    "6p":"六筒","7p":"七筒","8p":"八筒","9p":"九筒",
    "1s":"一条","2s":"二条","3s":"三条","4s":"四条","5s":"五条",
    "6s":"六条","7s":"七条","8s":"八条","9s":"九条",
    "1z":"东","2z":"南","3z":"西","4z":"北","5z":"白板","6z":"发财","7z":"红中",
}

ALL_CODES = list(CODE_TO_NAME.keys())


class MatchEngine:
    def __init__(self):
        self.templates = {}       # 原始BGR图
        self._tpl_prep = {}       # 预处理好的各种表示

    def load_all(self):
        self.templates = {}
        self._tpl_prep = {}
        template_dir = ROOT / "tile_templates"
        valid = re.compile(r'^[1-9][mpsz]$')
        if not template_dir.exists():
            return
        for f in template_dir.iterdir():
            if f.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            code = f.stem
            if not valid.match(code):
                continue
            raw = cv2.imread(str(f))
            if raw is None:
                continue
            self.templates[code] = raw
            self._prep_template(code, raw)

    def _prep_template(self, code, raw):
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # 基础：40x60 灰度
        base = cv2.resize(gray, (40, 60))
        base_norm = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)

        # 边缘
        edge_map = cv2.Canny(base_norm, 40, 120)

        # 多尺度： 0.8x, 1.0x, 1.2x
        scales = [0.8, 1.0, 1.2]
        multi = {}
        for s in scales:
            sw = int(40 * s)
            sh = int(60 * s)
            m = cv2.resize(base_norm, (sw, sh))
            multi[s] = m

        # 颜色直方图特征（H通道，区分万/筒/条）
        hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_s = cv2.normalize(hist_s, hist_s).flatten()

        # 纹理特征（LBP 简化版：局部梯度直方图）
        grad_x = cv2.Sobel(base_norm, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(base_norm, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(grad_x**2 + grad_y**2)
        hist_mag = cv2.calcHist([mag.astype(np.float32)], [0], None, [16], [0, 100])
        hist_mag = cv2.normalize(hist_mag, hist_mag).flatten()

        self._tpl_prep[code] = {
            "base_norm": base_norm,
            "edge": edge_map,
            "multi": multi,
            "hist_h": hist_h,
            "hist_s": hist_s,
            "hist_grad": hist_mag,
        }

    def match(self, tile_img):
        """返回 (code, score) 或 (None, 0.0)"""
        if tile_img is None or tile_img.size == 0:
            return None, 0.0

        if not self.templates:
            return None, 0.0

        # 预处理目标
        gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY) if len(tile_img.shape) == 3 else tile_img
        base = cv2.resize(gray, (40, 60))
        base_norm = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
        edge_tile = cv2.Canny(base_norm, 40, 120)

        # 颜色特征
        if len(tile_img.shape) == 3:
            hsv = cv2.cvtColor(tile_img, cv2.COLOR_BGR2HSV)
        else:
            color = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        hist_h_t = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_h_t = cv2.normalize(hist_h_t, hist_h_t).flatten()
        hist_s_t = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_s_t = cv2.normalize(hist_s_t, hist_s_t).flatten()

        # 纹理
        grad_x = cv2.Sobel(base_norm, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(base_norm, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(grad_x**2 + grad_y**2)
        hist_grad_t = cv2.calcHist([mag.astype(np.float32)], [0], None, [16], [0, 100])
        hist_grad_t = cv2.normalize(hist_grad_t, hist_grad_t).flatten()

        # 逐个模板打分
        scores = {}
        for code, prep in self._tpl_prep.items():
            score = self._score_one(
                code, base_norm, edge_tile,
                hist_h_t, hist_s_t, hist_grad_t,
                prep
            )
            scores[code] = score

        # 找最佳
        best_code = max(scores, key=scores.get)
        best_score = scores[best_code]
        if best_score < 0.05:
            return None, 0.0
        return best_code, best_score

    def match_top3(self, tile_img):
        """返回 [(code, score), ...] top3"""
        if tile_img is None or tile_img.size == 0:
            return []
        if not self.templates:
            return []

        gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY) if len(tile_img.shape) == 3 else tile_img
        base = cv2.resize(gray, (40, 60))
        base_norm = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
        edge_tile = cv2.Canny(base_norm, 40, 120)

        if len(tile_img.shape) == 3:
            hsv = cv2.cvtColor(tile_img, cv2.COLOR_BGR2HSV)
        else:
            color = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        hist_h_t = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_h_t = cv2.normalize(hist_h_t, hist_h_t).flatten()
        hist_s_t = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_s_t = cv2.normalize(hist_s_t, hist_s_t).flatten()
        grad_x = cv2.Sobel(base_norm, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(base_norm, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(grad_x**2 + grad_y**2)
        hist_grad_t = cv2.calcHist([mag.astype(np.float32)], [0], None, [16], [0, 100])
        hist_grad_t = cv2.normalize(hist_grad_t, hist_grad_t).flatten()

        s = []
        for code, prep in self._tpl_prep.items():
            score = self._score_one(code, base_norm, edge_tile, hist_h_t, hist_s_t, hist_grad_t, prep)
            s.append((code, score))
        s.sort(key=lambda x: x[1], reverse=True)
        return s[:3]

    def _score_one(self, code, base_norm, edge_tile, hist_h_t, hist_s_t, hist_grad_t, prep):
        suite = code[1]  # m/p/s/z

        # 1) 模板匹配 (多尺度取最大)
        tm_max = -1.0
        for scale, tpl_m in prep["multi"].items():
            try:
                res = cv2.matchTemplate(base_norm, tpl_m, cv2.TM_CCOEFF_NORMED)
                _, m, _, _ = cv2.minMaxLoc(res)
                if m > tm_max:
                    tm_max = m
            except Exception:
                pass

        # 2) 边缘匹配
        try:
            res_e = cv2.matchTemplate(edge_tile, prep["edge"], cv2.TM_CCOEFF_NORMED)
            _, edge_max, _, _ = cv2.minMaxLoc(res_e)
        except Exception:
            edge_max = 0.0

        # 3) 颜色直方图相似度
        hist_h_corr = float(cv2.compareHist(hist_h_t.reshape(-1, 1), prep["hist_h"].reshape(-1, 1), cv2.HISTCMP_CORREL))
        hist_s_corr = float(cv2.compareHist(hist_s_t.reshape(-1, 1), prep["hist_s"].reshape(-1, 1), cv2.HISTCMP_CORREL))
        hist_color = 0.5 * max(hist_h_corr, 0) + 0.5 * max(hist_s_corr, 0)

        # 4) 纹理相似度
        try:
            tex_sim = float(cv2.compareHist(hist_grad_t.reshape(-1, 1), prep["hist_grad"].reshape(-1, 1), cv2.HISTCMP_CORREL))
            tex_sim = max(tex_sim, 0)
        except Exception:
            tex_sim = 0.0

        # 综合得分 — 根据花色自适应权重
        if suite == "z":
            # 字牌：边缘+纹理更重要（字形区分）
            final = (
                0.25 * max(tm_max, 0) +
                0.35 * max(edge_max, 0) +
                0.15 * hist_color +
                0.25 * tex_sim
            )
        else:
            # 数牌：颜色+模板更重要（花色分离）
            final = (
                0.30 * max(tm_max, 0) +
                0.20 * max(edge_max, 0) +
                0.35 * hist_color +
                0.15 * tex_sim
            )
        return final

    def spatial_smooth(self, recognized_codes, tile_img=None):
        """
        空间平滑：利用邻居信息纠正孤立错误
        - 如果某位置花色与前后邻居都不同，且分数低的候选中有匹配邻居花色的，替换
        - 返回修正后的 codes 列表
        """
        if len(recognized_codes) < 3:
            return recognized_codes

        result = list(recognized_codes)
        n = len(result)

        for i in range(n):
            cur = result[i]
            if cur == "?":
                continue

            # 收集邻居花色
            neighbor_suits = set()
            if i > 0 and result[i-1] and result[i-1] != "?":
                neighbor_suits.add(result[i-1][1])
            if i < n-1 and result[i+1] and result[i+1] != "?":
                neighbor_suits.add(result[i+1][1])

            # 如果当前花色与所有邻居都不同，且邻居花色一致 → 可能是错误
            cur_suit = cur[1]
            if len(neighbor_suits) == 1 and cur_suit not in neighbor_suits:
                # 尝试用邻居花色的同数字替换
                target_suit = list(neighbor_suits)[0]
                candidate = cur[0] + target_suit  # 同数字不同花色
                if candidate in self.templates:
                    result[i] = candidate
                    continue

                # 或者标记为不确定
                # 如果 neighbors 是 m, 当前是 p, 可能是 p→m 的错误
                # 简单处理：保留原值但降低算牌时的优先度（这里暂不处理）

        return result

    def match_hand(self, hand_crop, num_tiles=14):
        """
        完整的手牌匹配流程
        返回 (codes_list, compact_str)
        """
        if hand_crop.size == 0:
            return [], ""

        hh, ww = hand_crop.shape[:2]
        tile_w = ww // num_tiles if num_tiles > 0 else 1
        if tile_w < 10:
            return [], ""

        overlap = int(tile_w * 0.25)  # 25% 重叠
        recognized = []

        for i in range(num_tiles):
            tx = max(0, i * tile_w - overlap)
            ex = min(ww, (i + 1) * tile_w + overlap)
            tile_img = hand_crop[:, tx:ex]
            code, score = self.match(tile_img)
            if code and score > 0.06:
                recognized.append(code)
            else:
                recognized.append("?")

        # 空间平滑
        recognized = self.spatial_smooth(recognized)

        # 去除 ? 
        valid = [c for c in recognized if c != "?"]
        if not valid:
            return recognized, ""

        groups = {"m": [], "p": [], "s": [], "z": []}
        for code in valid:
            groups[code[1]].append(code[0])

        compact = ""
        for suit in ["m", "p", "s", "z"]:
            if groups[suit]:
                compact += "".join(sorted(groups[suit])) + suit

        return recognized, compact
