"""应用引导。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .assets import AssetStore
from .manifest import Manifest


def ensure_qt_plugins():
    """显式把 PyQt5 自带 plugins 目录塞进环境变量。

    本机 Python 装在中文路径下，QLibraryInfo 解析出来的插件路径中文会变乱码
    （`D:/????/python/...`），QApplication 直接起不来。必须在创建 QApplication 之前调用。
    菲比项目打包时踩的是同一个坑（PyInstaller 的 Qt hook）。
    """
    try:
        import PyQt5
    except ImportError:
        return
    base = Path(PyQt5.__file__).resolve().parent
    for name in ("Qt5", "Qt"):
        plugins = base / name / "plugins"
        if plugins.is_dir():
            os.environ.setdefault("QT_PLUGIN_PATH", str(plugins))
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plugins / "platforms"))
            return


def main(argv=None) -> int:
    ensure_qt_plugins()
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("薄荷（猫猫）")
    app.setQuitOnLastWindowClosed(False)

    manifest = Manifest()
    store = AssetStore()
    store.preload([p.path for p in manifest.skin.poses.values()])

    from .stage import PetWindow
    pet = PetWindow(manifest, store)
    pet.place_default()
    pet.show()
    return app.exec_()
