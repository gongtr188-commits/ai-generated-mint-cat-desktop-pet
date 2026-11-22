#!/usr/bin/env python3
"""素材量化诊断 + 对比图产出。

用法：
    py -3 tools/inspect_assets.py                # 打印指标并生成 docs/preview/*.png
    py -3 tools/inspect_assets.py --no-images    # 只打印指标
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
PREVIEW = ROOT / "docs" / "preview"
LABELS = {"front": "正面", "side_left": "左面", "side_right": "右面", "back": "背面"}
DISPLAY_H = 300


def load() -> dict[str, Image.Image]:
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    skin = manifest["skins"]["default"]
    return {k: Image.open(ASSETS / v["file"]).convert("RGBA") for k, v in skin["poses"].items()}


def edge_band(img: Image.Image) -> dict:
    """半透明环带的亮度分布 = 白边体检。

    源图边缘是"深色描边 + 白底"的混合色，回填前景色之前环带会明显偏亮。
    实测未处理时视频帧有 15~21% 的环带像素亮度 >200，肉眼就是转身上的一圈白毛边。
    """
    a = np.asarray(img.convert("RGBA")).astype(int)
    A = a[..., 3]
    band = (A > 10) & (A < 245)
    if band.sum() == 0:
        return dict(n=0, mean=0.0, p90=0.0, hot=0.0)
    lum = a[..., :3].mean(2)[band]
    return dict(n=int(band.sum()), mean=round(float(lum.mean()), 1),
                p90=round(float(np.percentile(lum, 90)), 1),
                hot=round(float(100 * (lum > 200).mean()), 2))


def anim_stats(paths: list) -> dict:
    """逐帧内容高 / 底边 / 水平中心。转身"颠簸"的量化口径就在这里。"""
    hs, bs, cs = [], [], []
    for p in paths:
        a = np.asarray(Image.open(p).convert("RGBA"))[..., 3]
        ys, xs = np.where(a > 128)
        hs.append(ys.max() - ys.min() + 1)
        bs.append(ys.max())
        cs.append((xs.min() + xs.max()) / 2)
    hs, bs, cs = np.array(hs), np.array(bs), np.array(cs)
    return dict(n=len(hs), h_min=int(hs.min()), h_max=int(hs.max()),
                h_swing=round(float(100 * (hs.max() - hs.min()) / hs.mean()), 1),
                base_std=round(float(bs.std()), 2), cx_std=round(float(cs.std()), 2))


def metrics(img: Image.Image) -> dict:
    a = np.array(img)[..., 3]
    ys, xs = np.where(a > 8)
    w, h = img.size
    cw, ch = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    _, n = ndimage.label(a > 8)
    return dict(bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                content=(cw, ch), ratio=round(cw * ch / (w * h), 3),
                bottom_gap=h - int(ys.max()) - 1, blobs=n)


def row_sheet(imgs: dict) -> Image.Image:
    cells = []
    for key in LABELS:
        im = imgs[key]
        s = DISPLAY_H / im.height
        cells.append((key, im.resize((round(im.width * s), DISPLAY_H), Image.LANCZOS)))
    pad, top = 16, 34
    total_w = sum(c.width for _, c in cells) + pad * (len(cells) + 1)
    sheet = Image.new("RGB", (total_w, top + DISPLAY_H + 26), (236, 236, 240))
    d = ImageDraw.Draw(sheet)
    x = pad
    base_y = top + DISPLAY_H
    for key, c in cells:
        sheet.paste(c, (x, top), c)
        d.text((x, 12), f"{LABELS[key]} {key}  {c.width}x{c.height}", fill=(30, 30, 30))
        d.line([(x, base_y), (x + c.width, base_y)], fill=(200, 60, 60), width=2)
        x += c.width + pad
    return sheet


def overlay_sheet(imgs: dict) -> Image.Image:
    """四张轮廓叠同一基线，直接看大小/位置是否协调。"""
    tints = {"front": (255, 60, 60), "side_left": (60, 160, 255),
             "side_right": (60, 220, 120), "back": (255, 200, 40)}
    sheet = Image.new("RGBA", imgs["front"].size, (250, 250, 250, 255))
    for key, tint in tints.items():
        a = np.array(imgs[key])[..., 3]
        layer = np.zeros((*a.shape, 4), np.uint8)
        layer[..., :3] = tint
        layer[..., 3] = (a > 8) * 70
        sheet = Image.alpha_composite(sheet, Image.fromarray(layer))
    d = ImageDraw.Draw(sheet)
    d.line([(0, sheet.height - 24), (sheet.width, sheet.height - 24)], fill=(40, 40, 40), width=2)
    return sheet.convert("RGB")


def halo_sheet(imgs: dict) -> Image.Image:
    """铺纯品红底，暴露抠图残留白边/白点。"""
    cells = []
    for key in LABELS:
        im = imgs[key].copy()
        s = (DISPLAY_H * 0.8) / im.height
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        bg = Image.new("RGBA", im.size, (255, 0, 255, 255))
        bg.alpha_composite(im)
        cells.append((key, bg.convert("RGB")))
    pad = 16
    sheet = Image.new("RGB", (sum(c.width for _, c in cells) + pad * 5, DISPLAY_H + 30),
                      (255, 0, 255))
    d = ImageDraw.Draw(sheet)
    x = pad
    for key, c in cells:
        sheet.paste(c, (x, 24))
        d.text((x, 6), key, fill=(255, 255, 255))
        x += c.width + pad
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    imgs = load()

    print(f"{'pose':12} {'content':12} {'占画布比':8} {'底边留白':8} {'连通块':6} "
          f"{'环带亮度':16} {'>200占比':8} bbox")
    for key, img in imgs.items():
        m = metrics(img)
        e = edge_band(img)
        band = "%s/%s" % (e["mean"], e["p90"])
        hot = "%s%%" % e["hot"]
        print(f"{key:12} {str(m['content']):12} {m['ratio']:<8} {m['bottom_gap']:<8} "
              f"{m['blobs']:<6} {band:16} {hot:<8} {m['bbox']}")

    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    anims = manifest["skins"]["default"].get("anims", {})
    for name, a in anims.items():
        d = ASSETS / a["dir"]
        paths = [d / f"frame_{i:03d}.png" for i in range(a["frames"])]
        st = anim_stats(paths)
        cyc = a.get("cycle", st["n"])
        swing = anim_stats(paths[:cyc])
        print()
        print(f"anim {name}: 全 {st['n']} 帧 / 有效周期 {cyc} 帧")
        print(f"  周期内内容高 {swing['h_min']}~{swing['h_max']}  起伏 {swing['h_swing']}%"
              f"   底边 std {swing['base_std']}   中心x std {swing['cx_std']}")
        e = edge_band(Image.open(paths[0]))
        print(f"  环带亮度 {e['mean']}/{e['p90']}   >200 占比 {e['hot']}%")
        if swing["h_swing"] > 1.5:
            print(f"  [警告] 内容高起伏 {swing['h_swing']}% >1.5%，转身会看起来颠簸")
        if e["hot"] > 2.0:
            print(f"  [警告] 环带 >200 占比 {e['hot']}% >2%，会有可见白边")

    if args.no_images:
        return
    PREVIEW.mkdir(parents=True, exist_ok=True)
    row_sheet(imgs).save(PREVIEW / "poses_row.png")
    overlay_sheet(imgs).save(PREVIEW / "poses_overlay.png")
    halo_sheet(imgs).save(PREVIEW / "poses_halo.png")
    print(f"\n对比图已输出到 {PREVIEW}")


if __name__ == "__main__":
    main()
