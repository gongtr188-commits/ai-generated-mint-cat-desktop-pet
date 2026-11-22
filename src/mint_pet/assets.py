"""素材加载：原图只读盘一次，按显示高缩放并缓存。

43 MB / 125 张图若全部预缩放到每种尺寸，启动时间和内存都不可接受。
"""
from __future__ import annotations

from pathlib import Path

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class AssetStore:
    def __init__(self):
        self._raw: dict[Path, QImage] = {}
        self._scaled: dict[tuple, QPixmap] = {}

    def raw(self, path: Path) -> QImage:
        img = self._raw.get(path)
        if img is None:
            img = QImage(str(path))
            if img.isNull():
                raise FileNotFoundError(f"素材读取失败：{path}")
            img = img.convertToFormat(QImage.Format_ARGB32_Premultiplied)
            self._raw[path] = img
        return img

    def pix(self, path: Path, h: int, k: float = 1.0) -> QPixmap:
        key = (path, h, round(k, 3))
        pm = self._scaled.get(key)
        if pm is None:
            src = self.raw(path)
            target = max(1, int(h * k))
            scaled = src.scaledToHeight(target, Qt.SmoothTransformation)
            pm = QPixmap.fromImage(scaled)
            self._scaled[key] = pm
        return pm

    def frame_pix(self, anim_dir: Path, index: int, h: int) -> QPixmap:
        """动画帧：只缓存缩放后的图，原图读完即弃。

        121 张 768² 原图常驻要 285 MB，而缩放后整段也就 10 MB 量级；
        静态姿势才值得进 `_raw`（只有 4 张）。
        """
        path = anim_dir / f"frame_{index:03d}.png"
        key = (path, h, 1.0)
        pm = self._scaled.get(key)
        if pm is None:
            img = QImage(str(path))
            if img.isNull():
                raise FileNotFoundError(f"动画帧读取失败：{path}")
            # 必须先转预乘再缩放：非预乘下 SmoothTransformation 会独立插值 alpha 与 RGB，
            # 半透明边缘会被拉出一圈亮边，正是"转动时白边没扣干净"的放大器。
            img = img.convertToFormat(QImage.Format_ARGB32_Premultiplied)
            pm = QPixmap.fromImage(img.scaledToHeight(max(1, h), Qt.SmoothTransformation))
            self._scaled[key] = pm
        return pm

    def drop_scale_cache(self):
        """切尺寸时只清缩放缓存，原图常驻，所以缩放是瞬时的。"""
        self._scaled.clear()

    def preload(self, paths) -> int:
        n = 0
        for p in paths:
            self.raw(Path(p))
            n += 1
        return n
