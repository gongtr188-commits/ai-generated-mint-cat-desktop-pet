"""逐帧胶片核对：抠图完整性 + 动作连续性（每个动画一张深色底图）。

    py -3 tools/strip_check.py            # 全部动画
    py -3 tools/strip_check.py sitdown    # 只看某一段

为什么需要它：数值指标（面积/包围盒/孤岛/对齐）会全绿而图是坏的——
§11.12 那个光晕缺陷、以及"描边墙"那次把四足走的胸口削掉，都是这张图看出来的。
深色底是故意的：多保留的背景色只有压在深色上才扎眼。

每张图里**红框**标出的是"这一帧的前景面积偏离整段中位超过 12%"的帧，
也就是最可能被抠穿/被吃掉一块的帧，肉眼扫的时候先看它们。
"""
from __future__ import annotations

import sys
import pathlib
import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "preview"
CELL, COLS, MAXCELL = 128, 12, 96
BG = (34, 40, 46)
RED = (255, 70, 70)


def alpha_bbox(a: np.ndarray):
    ys, xs = np.where(a > 8)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def sheet(name: str, anim: dict) -> int:
    d = ROOT / "assets" / anim["dir"]
    n = int(anim["frames"])
    step = max(1, -(-n // MAXCELL))
    idxs = list(range(0, n, step))
    files = [d / f"frame_{i:03d}.png" for i in idxs]
    if not files[0].exists():
        print(f"跳过 {name}：找不到 {files[0]}")
        return 0

    areas, cells, notes = [], [], []
    for i, f in zip(idxs, files):
        arr = np.asarray(Image.open(f).convert("RGBA"))
        a = arr[..., 3]
        areas.append(int((a > 8).sum()))
        bg = Image.new("RGBA", arr.shape[1::-1], BG + (255,))
        bg.alpha_composite(Image.open(f).convert("RGBA"))
        cells.append((i, bg.convert("RGB")))
    med = float(np.median(areas))

    rows = -(-len(cells) // COLS)
    img = Image.new("RGB", (COLS * CELL, rows * (CELL + 14)), (20, 24, 28))
    dr = ImageDraw.Draw(img)
    bad = []
    for k, ((i, c), ar) in enumerate(zip(cells, areas)):
        x, y = (k % COLS) * CELL, (k // COLS) * (CELL + 14)
        img.paste(c.resize((CELL, CELL), Image.LANCZOS), (x, y))
        dev = ar / med - 1
        if abs(dev) > 0.12:
            dr.rectangle([x + 1, y + 1, x + CELL - 2, y + CELL - 2], outline=RED, width=2)
            bad.append((i, round(dev * 100)))
        dr.text((x + 3, y + CELL + 1), f"{i:03d} {dev * 100:+.0f}%", fill=(190, 200, 210))
    out = OUT / f"strip_{name}.png"
    img.save(out)

    # 动作连续性：相邻采样帧的面积跳变与水平中心抖动
    cs, jumps = [], 0
    bbs = []
    for i, f in zip(idxs, files):
        a = np.asarray(Image.open(f).convert("RGBA"))[..., 3]
        bbs.append(alpha_bbox(a))
    for p, q in zip(bbs, bbs[1:]):
        cs.append(abs((p[0] + p[2] - q[0] - q[2]) / 2))
    prev = areas[0]
    for ar in areas[1:]:
        jumps = max(jumps, abs(ar - prev) / max(1, prev) * 100)
        prev = ar
    print(f"{name:<12} {n:>3}帧 步长{step}  面积 {min(areas) / med * 100:.0f}~"
          f"{max(areas) / med * 100:.0f}%（中位 {med:.0f}px）  相邻跳变≤{jumps:.1f}%  "
          f"水平抖动≤{max(cs):.0f}px  红框 {len(bad)} 帧 {bad[:6]}")
    print(f"    → {out.relative_to(ROOT)}")
    return len(bad)


def main():
    import json
    m = json.loads((ROOT / "assets" / "manifest.json").read_text(encoding="utf-8"))
    anims = m["skins"]["default"]["anims"]
    want = sys.argv[1:] or list(anims)
    OUT.mkdir(parents=True, exist_ok=True)
    tot = 0
    for name in want:
        if name not in anims:
            print(f"没有这个动画：{name}（可选：{', '.join(anims)}）")
            continue
        tot += sheet(name, anims[name])
    print(f"\n合计 {tot} 帧被标红。红框不等于有 bug："
          f"走路/爬行本来就有肢体收放造成的面积起伏，要看的是**同一形状反复出现**的那种。")


main()
