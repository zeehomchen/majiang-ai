"""命令行入口 -- 支持手动编码输入和 DirectX 实时捕获两种模式。"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .evaluator import evaluate_hand, result_to_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="广东推倒胡智能算牌器决策引擎。")

    # ---- 输入模式 ----
    parser.add_argument(
        "--hand",
        default="",
        help="(手动模式) 紧凑手牌编码，如 123m555678p1122s",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="(实时模式) 通过 DirectX 截图自动捕获手牌和牌河",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="运行区域校准工具，生成校准叠加图",
    )
    parser.add_argument(
        "--capture-templates",
        action="store_true",
        help="采集手牌模板图片用于训练识别器",
    )
    parser.add_argument(
        "--template-dir",
        default="tile_templates",
        help="模板图片目录 (默认: tile_templates)",
    )

    # ---- 公共参数 ----
    parser.add_argument(
        "--visible",
        default="",
        help="(手动模式) 可见牌列表，如 3m,3m,7z",
    )
    parser.add_argument(
        "--melds",
        default="",
        help="他家已碰/杠的牌，如 555p 777z",
    )
    parser.add_argument(
        "--phase",
        default="early",
        choices=("early", "mid", "late"),
        help="牌局阶段 (early/mid/late)",
    )
    parser.add_argument(
        "--open-hand",
        action="store_true",
        help="标记手牌已副露 (限制门清路线)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="显示前 N 个切牌选项",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON 格式",
    )
    parser.add_argument(
        "--window",
        default="",
        help="(实时模式) 窗口标题关键词，如 '腾讯麻将'",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="(实时模式) 轮询间隔 (秒)，默认 3.0",
    )

    return parser


def _save_debug_images(hand_frame, river_frame) -> None:
    """保存调试截图，用于校准区域坐标。"""
    try:
        import cv2

        if hand_frame is not None and hand_frame.size > 0:
            h, w = hand_frame.shape[:2]
            cv2.imwrite("debug_hand.png", hand_frame)
            dark_pct = (hand_frame.max(axis=2) < 80).mean() * 100
            print(f"  调试: 手牌区域 {w}x{h}, 暗像素 {dark_pct:.0f}% -> debug_hand.png")
        if river_frame is not None and river_frame.size > 0:
            h, w = river_frame.shape[:2]
            cv2.imwrite("debug_river.png", river_frame)
            dark_pct = (river_frame.max(axis=2) < 80).mean() * 100
            print(f"  调试: 牌河区域 {w}x{h}, 暗像素 {dark_pct:.0f}% -> debug_river.png")
    except Exception:
        pass


def _print_text_result(result_dict: dict, top: int) -> None:
    options = result_dict["options"][:top]
    best = options[0]
    print(f"当前手牌: {result_dict['hand']}")
    print(f"建议切牌: {best['discard']}")
    print(f"综合评分: {best['total_score']}")
    print()
    print("Top 选项:")
    for index, option in enumerate(options, start=1):
        print(
            f"{index}. 打 {option['discard']} | 总分 {option['total_score']} | "
            f"基础分 {option['base_score']} | 有效进张 {option['effective_draws']}"
        )
        print("   改良牌: " + (", ".join(option["improving_tiles"]) or "-"))
        for line in option["summary"]:
            print("   - " + line)
        if option["route_scores"]:
            route_text = ", ".join(
                f"{route}={score}" for route, score in option["route_scores"].items()
            )
            print("   路线评分: " + route_text)
        print()


def _run_live(args: argparse.Namespace) -> int:
    """DirectX 实时捕获 + 识别 + 决策循环。"""
    from .capture import Capturer, CaptureError, calibrate as do_calibrate
    from .vision import RecognitionResult, recognize_tiles, capture_templates, load_templates, match_with_templates

    # ---- 校准模式 ----
    if args.calibrate:
        print("=== 区域校准模式 ===")
        do_calibrate()
        return 0

    # ---- 模板采集模式 ----
    if args.capture_templates:
        capturer = Capturer(title_keywords=[args.window] if args.window else None)
        capturer.start()
        templates = capture_templates(capturer, args.template_dir)
        capturer.stop()
        print(f"模板采集完成: {len(templates)} 张")
        return 0

    # ---- 主循环 ----
    keywords = [args.window] if args.window else None
    capturer = Capturer(title_keywords=keywords)

    try:
        capturer.start()
        print("DirectX 捕获已启动，按 Ctrl+C 停止...")
        print(f"轮询间隔: {args.interval}s")
        print("提示: 首次使用建议先运行 --calibrate 校准区域坐标")
        print("-" * 50)
    except CaptureError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    templates = load_templates(args.template_dir)
    if templates:
        print(f"已加载 {len(templates)} 张牌面模板")

    try:
        while True:
            # 抓取手牌
            try:
                hand_frame = capturer.capture_hand()
                river_frame = capturer.capture_river()
            except CaptureError as exc:
                print(f"[{time.strftime('%H:%M:%S')}] 捕获失败: {exc}")
                time.sleep(args.interval)
                continue

            # 识别
            result: RecognitionResult = recognize_tiles(hand_frame, river_frame)

            if result.errors:
                print(f"[{time.strftime('%H:%M:%S')}] 识别错误: {'; '.join(result.errors)}")
                _save_debug_images(hand_frame, river_frame)
                time.sleep(args.interval)
                continue

            hand = result.hand_compact
            if not hand or len(hand.strip()) < 4:
                print(f"[{time.strftime('%H:%M:%S')}] 手牌识别不足 (已识别: '{hand}')，跳过本轮")
                _save_debug_images(hand_frame, river_frame)
                time.sleep(args.interval)
                continue

            visible = result.visible_compact or ""

            print(f"[{time.strftime('%H:%M:%S')}] 识别手牌: {hand}")
            if visible:
                print(f"  可见牌: {visible}")

            # 决策
            try:
                eval_result = evaluate_hand(
                    hand=hand,
                    visible_tiles=visible,
                    exposed_triplets=args.melds or "",
                    round_phase=args.phase,
                    closed_hand=not args.open_hand,
                )
                result_dict = result_to_dict(eval_result)

                if args.json:
                    print(json.dumps(result_dict, ensure_ascii=False, indent=2))
                else:
                    _print_text_result(result_dict, top=args.top)
            except ValueError as exc:
                print(f"[{time.strftime('%H:%M:%S')}] 决策失败: {exc}")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        capturer.stop()

    return 0


def _run_manual(args: argparse.Namespace) -> int:
    """手动编码输入模式 (原有逻辑)。"""
    result = evaluate_hand(
        hand=args.hand,
        visible_tiles=args.visible,
        exposed_triplets=args.melds,
        round_phase=args.phase,
        closed_hand=not args.open_hand,
    )
    result_dict = result_to_dict(result)
    if args.json:
        print(json.dumps(result_dict, ensure_ascii=False, indent=2))
    else:
        _print_text_result(result_dict, top=args.top)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # 实时模式
    if args.live or args.calibrate or args.capture_templates:
        return _run_live(args)

    # 手动模式
    if not args.hand:
        parser.error("请指定 --hand 编码或使用 --live 实时模式")
    return _run_manual(args)
