#!/usr/bin/env python3
"""转身标定复核 + 转身序列胶片图。

关键帧、周期、累计路径由 `tools/build_assets.py:turn_geometry()` 在入库时算好写进
`assets/manifest.json`；本脚本**不重算第二套**，只做三件事：
  1. 用路径比例法（累计视觉进度 1/4 / 1/2 / 3/4 圈）交叉验证管线给的 keys
  2. 报告周期检测是否把视频尾部的静止帧排除了
  3. 出转身胶片图，肉眼确认没有多余帧 / 没有起手一顿
    py -3 tools/calibrate_turn.py
    py -3 tools/calibrate_turn.py --strip
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mint_pet import config                      # noqa: E402
from mint_pet.facing import Facing, VideoTurn    # noqa: E402

_spec = importlib.util.spec_from_file_location("ba", ROOT / "tools" / "build_assets.py")
ba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ba)

FACES = ["front", "side_left", "back", "side_right"]


def load():
    m = json.loads((ROOT / "assets" / "manifest.json").read_text(encoding="utf-8"))
    sk = m["skins"]["default"]
    anim = sk["anims"][config.SPIN_ANIM]
    frames = [ROOT / "assets" / anim["dir"] / f"frame_{i:03d}.png" for i in range(anim["frames"])]
    poses = {k: ROOT / "assets" / v["file"] for k, v in sk["poses"].items()}
    return anim, frames, poses


def check(anim, frames, poses) -> int:
    cycle, keys, path = anim["cycle"], anim["keys"], anim["path"]
    th = np.array([ba.thumb(f) for f in frames])
    adj = np.abs(np.diff(th, axis=0, append=th[:1])).mean(1)
    total = float(adj[:cycle].sum())
    print(f"帧数={anim['frames']}  周期={cycle}（尾部丢弃 {anim['frames'] - cycle} 帧静止重复）")
    print(f"一圈视觉路径={total:.1f}  相邻帧差 min/med/max "
          f"{adj[:cycle].min():.2f}/{np.median(adj[:cycle]):.2f}/{adj[:cycle].max():.2f}")
    print(f"keys={keys}  顺序={FACES}")

    cum = np.concatenate([[0.0], np.cumsum(adj[:cycle])]) / total
    print("\n[交叉验证] 路径比例法（累计视觉进度落在 1/4 / 1/2 / 3/4 圈的帧）")
    print("  注意：比例法给的是『角度进度等分点』，参考法给的是『与标准姿势最像的帧』，")
    print("  两者本来就允许有偏差（起步慢、到位有停顿）。这里只用来抓数量级错误。")
    video_order = [0, 3, 2, 1]                    # 视频前进方向：正→右→背→左
    gap = []
    for rank, face in enumerate(video_order):
        if rank == 0:
            continue
        want = rank / 4
        idx = int(np.argmax(cum >= want))
        gap.append(abs(idx - keys[face]))
        print(f"  {FACES[face]:11} keys={keys[face]:3d}  比例法={idx:3d}  差={abs(idx - keys[face]):2d}")
    bad = [g for g in gap if g > cycle // 6]
    print(f"\n结论：{'⚠️ 偏差过大，先查素材再改代码' if bad else '两法在容忍范围内，keys 可用'}")
    return 1 if bad else 0


def strip(anim, frames) -> Path:
    """把每个 90°（外加一个 180°）实际会被播放的帧画出来。"""
    cycle, keys, path = anim["cycle"], anim["keys"], anim["path"]
    seqs = []
    for src, dst in ((0, 3), (3, 2), (2, 1), (1, 0), (0, 2)):
        v = VideoTurn(cycle, keys, path, lambda i: None)
        f = Facing(lambda x: None, count=4, director=v)
        f.value = src
        v.start(f, dst)
        seqs.append((src, dst, v.seq))

    S, HDR = 150, 16
    sheet = Image.new("RGB", (S * (max(len(s) for *_, s in seqs) + 1),
                              (S + HDR + 4) * len(seqs) + 16), (236, 236, 240))
    d = ImageDraw.Draw(sheet)
    for r, (src, dst, seq) in enumerate(seqs):
        y = 8 + r * (S + HDR + 4)
        # 默认位图字体没有中文字形，标签一律用 ASCII
        d.text((6, y), f"{FACES[src]} -> {FACES[dst]}  {seq}", fill=(0, 0, 0))
        for c, idx in enumerate(seq):
            im = Image.open(frames[idx]).convert("RGBA").resize((S, S))
            bg = Image.new("RGBA", (S, S), (255, 0, 255, 255))
            bg.alpha_composite(im)
            sheet.paste(bg.convert("RGB"), (6 + c * S, y + HDR))
    out = ROOT / "docs" / "preview" / "turn_strip.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", action="store_true", help="额外输出转身胶片图")
    args = ap.parse_args()
    anim, frames, poses = load()
    rc = check(anim, frames, poses)
    if args.strip:
        print(f"\n胶片图 → {strip(anim, frames)}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
