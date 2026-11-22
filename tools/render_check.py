"""离屏渲染 PetWindow，验证真实绘制路径（AssetStore→QPixmap→QLabel→透明窗）。

    py -3 tools/render_check.py

出图在 docs/preview/：
  render_check.png      所有姿势 + 关键动画帧的拼图，用肉眼看有没有裁切/漂移/毛边
  数字那部分是逐帧包围盒，肉眼看不出来的几像素漂移靠它兜底。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                    # noqa: E402
from PIL import Image                                 # noqa: E402
from PyQt5.QtGui import QColor, QImage, QPainter      # noqa: E402
from PyQt5.QtWidgets import QApplication             # noqa: E402

from mint_pet.app import ensure_qt_plugins            # noqa: E402
from mint_pet.assets import AssetStore                # noqa: E402
from mint_pet.manifest import Manifest                # noqa: E402
from mint_pet.stage import PetWindow                  # noqa: E402

ensure_qt_plugins()
OUT = ROOT / "docs" / "preview"
H, PAD = 200, 14
MAGENTA = QColor(255, 0, 255)

# 文案一律跟素材自己的命名走（左面/右面），不自造"左侧身"这种词
LABELS = {"front": "正面", "side_left": "左面", "back": "背面", "side_right": "右面",
          "sit_front": "坐下·正面", "prone_front": "正面趴下",
          "prone_left": "向左趴着", "prone_right": "向右趴着",
          "doze_front": "打瞌睡·正面", "sleep_left": "向左趴着睡",
          "sleep_right": "向右趴着睡"}

# 动画抽样：转身看标定 key，四足走看 recenter 会不会让脚底漂，坐下看过渡/驻留分界，
# 跑步把整圈相位都铺开（它的素材长度不是整数个周期，接缝最容易出问题）
ANIM_SHOTS = [
    ("walk4_right", [0, 24, 48, 72, 96]),
    ("walk4_left", [0, 48, 96]),
    ("run_right", [0, 13, 26, 39, 52, 65, 78, 91]),
    ("run_left", [0, 18, 36, 54, 72, 90]),
    ("sitdown", [0, 14, 27, 28, 90, 144]),
    ("prone_turn", [8, 40, 94]),
    ("walk_right", [0, 60, 120]),
    # 驻留微动是**常驻**画面（一待就是十几到三十秒），光晕/毛边会被一直看着，必须上深色底复检
    ("prone_idle", [0, 22, 43]),
    ("doze", [0, 21, 42]),
    ("sleep_left", [0, 48, 96]),
    ("sleep_right", [0, 20, 40]),
    ("hug", [0, 8, 20, 40, 54, 75]),
]


def bbox_of(path) -> tuple:
    """素材里真正有像素的那一块。alpha>8 才算，半透明裙边不计入。"""
    p = Path(path)
    if not p.exists():
        p = MANIFEST.root / p
    a = np.asarray(Image.open(p).convert("RGBA"))[..., 3]
    ys, xs = np.where(a > 8)
    if not len(xs):
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def drift_report(name: str, frames: list):
    anim = MANIFEST.anim(name)
    d = MANIFEST.root / anim["dir"]
    files = [d / f"frame_{i:03d}.png" for i in frames]
    boxes = [bbox_of(f) for f in files]
    w = [b[2] - b[0] + 1 for b in boxes]
    h = [b[3] - b[1] + 1 for b in boxes]
    cx = [(b[0] + b[2]) / 2 for b in boxes]
    bottom = [b[3] for b in boxes]
    print(f"  {name:<12} 帧{frames}")
    print(f"    宽 {min(w)}~{max(w)}  高 {min(h)}~{max(h)}"
          f"  水平中心摆幅 ±{(max(cx) - min(cx)) / 2:.1f}px"
          f"  底边 {min(bottom)}~{max(bottom)}")


def main():
    global MANIFEST
    app = QApplication(sys.argv)
    m = MANIFEST = Manifest()
    store = AssetStore()
    store.preload([p.path for p in m.skin.poses.values()])
    pet = PetWindow(m, store)
    pet.set_size(H)
    pet.show()

    cells = []
    for key, pose in m.skin.poses.items():
        pet.label.setPixmap(store.pix(pose.path, pet.pet_h, 1.0))
        app.processEvents()
        cells.append((LABELS.get(key, key), key, grab(pet)))

    for name, frames in ANIM_SHOTS:
        for i in frames:
            pet.show_anim_frame(name, i)
            app.processEvents()
            cells.append((f"{name} #{i}", "", grab(pet)))

    cells = [c for c in cells if c[2] is not None]
    per_row = 5
    rows = [cells[i:i + per_row] for i in range(0, len(cells), per_row)]
    cw = max(img.width() for _, _, img in cells)
    ch = max(img.height() for _, _, img in cells) + 30
    OUT.mkdir(parents=True, exist_ok=True)

    def compose(fname, bg, with_labels):
        sheet = QImage(PAD + (cw + PAD) * per_row, PAD + int(ch) * len(rows), QImage.Format_RGB32)
        sheet.fill(bg)
        p = QPainter(sheet)
        for r, row in enumerate(rows):
            for j, (label, key, img) in enumerate(row):
                x, y = PAD + j * (cw + PAD), PAD + int(ch) * r
                if with_labels:
                    p.setPen(QColor(0x22, 0x22, 0x22) if bg.red() > 128 else QColor(0xDD, 0xE4, 0xEA))
                    p.drawText(x, y + 16, f"{label}" + (f" {key}" if key else ""))
                p.drawImage(x, y + 22, img)
        p.end()
        sheet.save(str(OUT / fname))
        print(f"写出 {OUT / fname}  {sheet.width()}x{sheet.height()}")

    compose("render_check.png", QColor(0xF2, 0xF4, 0xF5), True)
    # 品红底专抓"漏抠"：任何本该透明却留下来的地方，在白底上根本看不出来。
    compose("render_check_magenta.png", MAGENTA, False)
    # 深色桌面那一张才是"用户实际看到什么"。§11.12 那个光晕缺陷就是这么发现的，
    # 当时所有数值指标全绿——多保留下来的背景色只有压在深色上才扎眼。
    compose("render_check_dark.png", QColor(0x22, 0x28, 0x2E), False)

    print("\n逐帧包围盒（画布 896x768 上的原始像素，用来发现肉眼看不出的漂移）：")
    for name, frames in ANIM_SHOTS:
        drift_report(name, frames)

    print("\n常驻姿势：")
    for key, pose in m.skin.poses.items():
        b = bbox_of(pose.path)
        print(f"  {key:<12} x{b[0]}~{b[2]} y{b[1]}~{b[3]}  "
              f"{'⚠ 触边' if b[0] <= 1 or b[2] >= m.canvas.w - 2 or b[3] >= m.canvas.h - 2 else 'ok'}")
    pet.close()


def grab(pet):
    """窗口当前的真实绘制结果（带 alpha）。底纹由 compose() 铺，同一张图要压两种底色看。"""
    return pet.grab().toImage().convertToFormat(QImage.Format_ARGB32_Premultiplied)


if __name__ == "__main__":
    main()
