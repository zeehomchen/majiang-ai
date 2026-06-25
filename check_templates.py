"""
检查模板状态程序！
列出所有需要的牌，哪些有了，哪些还缺，中文显示！
"""

from __future__ import annotations

import re
from pathlib import Path


# 完整的牌表
ALL_TILES = [
    # 万子
    ("1m", "一万"), ("2m", "二万"), ("3m", "三万"),
    ("4m", "四万"), ("5m", "五万"), ("6m", "六万"),
    ("7m", "七万"), ("8m", "八万"), ("9m", "九万"),
    # 筒子
    ("1p", "一筒"), ("2p", "二筒"), ("3p", "三筒"),
    ("4p", "四筒"), ("5p", "五筒"), ("6p", "六筒"),
    ("7p", "七筒"), ("8p", "八筒"), ("9p", "九筒"),
    # 条子
    ("1s", "一条"), ("2s", "二条"), ("3s", "三条"),
    ("4s", "四条"), ("5s", "五条"), ("6s", "六条"),
    ("7s", "七条"), ("8s", "八条"), ("9s", "九条"),
    # 字牌
    ("1z", "东"), ("2z", "南"), ("3z", "西"), ("4z", "北"),
    ("5z", "白板"), ("6z", "发财"), ("7z", "红中"),
]


def check_templates():
    ROOT = Path(__file__).parent
    template_dir = ROOT / "tile_templates"

    if not template_dir.exists():
        print("❌ tile_templates 文件夹不存在！")
        return

    # 列出所有模板
    valid_code = re.compile(r'^[1-9][mpsz]\.(png|jpg|jpeg)$', re.IGNORECASE)
    existing = set()

    for f in template_dir.iterdir():
        if f.is_file() and valid_code.match(f.name):
            code = f.stem
            existing.add(code)

    # 分类统计
    categories = {
        "万子": [t for t in ALL_TILES if t[0].endswith("m")],
        "筒子": [t for t in ALL_TILES if t[0].endswith("p")],
        "条子": [t for t in ALL_TILES if t[0].endswith("s")],
        "字牌": [t for t in ALL_TILES if t[0].endswith("z")],
    }

    print("=" * 60)
    print("牌面模板状态检查")
    print("=" * 60)

    for cat_name, tiles in categories.items():
        print(f"\n【{cat_name}】")
        have = []
        missing = []
        for code, name in tiles:
            if code in existing:
                have.append(f"✅ {name} ({code})")
            else:
                missing.append(f"❌ {name} ({code})")

        if have:
            print("  已有:")
            for line in have:
                print(f"    {line}")
        if missing:
            print("  缺失:")
            for line in missing:
                print(f"    {line}")

    # 总计
    total = len(ALL_TILES)
    have_count = len(existing)
    print("\n" + "=" * 60)
    print(f"总计: {have_count}/{total} 张牌面已采集")
    print(f"还缺: {total-have_count} 张")
    print("=" * 60)

    if missing:
        print("\n使用 capture_single.py 来采集缺失的牌！")


if __name__ == "__main__":
    check_templates()
