"""鼠标交互（拖拽 / 单击换朝向）与右键菜单。

以 mixin 形式混入 PetWindow，让窗口本体只管绘制与尺寸。
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMenu, QSlider,
                             QWidget, QWidgetAction)

from . import config

FACE_LABELS = {"front": "正面", "side_left": "左面", "back": "背面", "side_right": "右面"}

MENU_QSS_TMPL = """
QMenu {{
    background: #ffffff; border: 1px solid #d8e6e4; border-radius: 10px;
    padding: 6px 4px; font: 13px "Microsoft YaHei UI";
}}
QMenu::item {{ padding: 5px 26px 5px 18px; border-radius: 6px; color: #2c3d3b; }}
QMenu::item:selected {{ background: {accent}; color: #ffffff; }}
QMenu::item:checked {{ color: {accent}; font-weight: 600; }}
QMenu::separator {{ height: 1px; background: #e6efee; margin: 5px 10px; }}
QMenu::right-arrow {{ margin-right: 8px; }}
QLabel {{ background: transparent; }}
QLabel#sizeName {{ color: #2c3d3b; }}
QLabel#sizeValue {{ color: {accent}; font-weight: 600; min-width: 52px; }}
QSlider {{ background: transparent; min-height: 20px; }}
QSlider::groove:horizontal {{ height: 4px; background: #d8e6e4; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; margin: -6px 0; border-radius: 8px;
    background: #ffffff; border: 2px solid {accent};
}}
QSlider::handle:horizontal:hover {{ background: {accent}; }}
"""


class InteractionMixin:
    def init_interaction(self):
        self._press_pos = None
        self._drag_offset = None
        self._dragging = False
        self._pending_size = None
        self._menu_open = False     # 菜单开着的时候她不许跟随（见 Follower._busy）
        # 一次性定时器必须挂父对象，裸 QTimer() 在 widget 销毁后回调会悬空
        self._size_timer = QTimer(self, interval=config.SIZE_DEBOUNCE_MS,
                                  singleShot=True, timeout=self._apply_size)

    # ── 鼠标 ────────────────────────────────────────────────────────────
    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        self._press_pos = e.globalPos()
        self._drag_offset = e.globalPos() - self.frameGeometry().topLeft()
        self._dragging = False
        self.behaviors.note_user()      # 用户主动出手，接下来一会儿别自己乱动
        e.accept()

    def mouseMoveEvent(self, e):
        if self._press_pos is None:
            return
        # 阈值 4px：连击时手抖 1~2px 就进拖拽态，会把后续点击语义全带偏
        if not self._dragging and \
                (e.globalPos() - self._press_pos).manhattanLength() < config.DRAG_THRESHOLD:
            return
        if not self._dragging:
            self.walker.stop()          # 用户上手了就走不下去了
            for h in self.holders.values():
                h.stop()                # 坐着/趴着被拖起来时立刻起身，别手底下还在演坐下
            self.crawler.stop()         # 四足爬走同理
            # 排队里的姿态意图同理：手已经把她拿起来了，
            # 松手之后她不该突然自己坐下去 —— 那是惊吓，不是响应刚才那个点击。
            self.clear_posture_want()
            self.oneshot.stop()        # 一次性动作同理，不演完
        self._dragging = True
        self.move(e.globalPos() - self._drag_offset)
        self._clamp_to_screen()
        e.accept()

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        dragged, self._press_pos = self._dragging, None
        self._dragging = False
        self._drag_offset = None
        if not dragged:
            self.facing.cycle()
        e.accept()

    def contextMenuEvent(self, e):
        self.show_menu(e.globalPos())

    def _clamp_to_screen(self):
        g = QApplication.desktop().availableGeometry(self)
        self.move(max(g.left(), min(self.x(), g.right() - self.width() + 1)),
                  max(g.top(), min(self.y(), g.bottom() - self.height() + 1)))

    # ── 菜单 ────────────────────────────────────────────────────────────
    def show_menu(self, global_pos):
        menu = self.build_menu()
        self._style_menu(menu)
        # 菜单是阻塞事件循环，开着的那几秒她还在 tick。两头各按一次安静期，
        # 免得用户正在挑选项，桌宠在底下自己走开。
        self.behaviors.note_user()
        self._menu_open = True      # 跟随也要停：否则她会跟着光标一路走进菜单里
        try:
            menu.exec_(global_pos)
        finally:
            self._menu_open = False
            self.behaviors.note_user()

    def _style_menu(self, menu):
        # QMenu 默认是不透明方形，不加这两行圆角外会露出一圈方角黑底
        menu.setWindowFlags(menu.windowFlags() | Qt.FramelessWindowHint
                            | Qt.NoDropShadowWindowHint)
        menu.setAttribute(Qt.WA_TranslucentBackground)
        menu.setStyleSheet(MENU_QSS_TMPL.format(accent=config.MENU_ACCENT))

    def _later(self, fn):
        """菜单动作必须等 exec_ 退出后再执行，否则和菜单自己的事件循环打架。

        传进来的 fn **只准捕获 int / 布尔 / self**，绝不能捕获菜单拥有的 QAction 或 QWidget：
        `build_menu()` 建的 QMenu 没有父对象，`show_menu()` 一返回它就被 C++ 析构，
        那时回调再碰那些对象是解引用已销毁内存，PyQt 直接 abort 整个进程且不抛 Python 异常。
        朝向项就因为捕获了 `a` 而必崩（点"左侧身/右侧身"闪退的根因）。
        """
        QTimer.singleShot(0, fn)

    def _turn_later(self, action):
        """朝向项：趁菜单还活着把 data() 取成普通 int，再交给延迟回调。"""
        q = action.data()
        self._later(lambda: self.facing.turn_to(q))

    def _size_action(self, menu) -> QWidgetAction:
        """菜单内嵌尺寸滑条：左边名字、右边实时像素值。

        不用 `_later()`：滑条要的是**拖到哪就立刻变成多大**，
        而 `_later` 会把动作推到菜单关闭之后，那就变成"调完看不到效果"。
        """
        act = QWidgetAction(menu)
        row = QWidget(menu)
        lay = QHBoxLayout(row)
        lay.setContentsMargins(16, 6, 18, 6)
        lay.setSpacing(10)

        name = QLabel("尺寸", row)
        name.setObjectName("sizeName")
        value = QLabel(f"{self.pet_h} px", row)
        value.setObjectName("sizeValue")
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        slider = QSlider(Qt.Horizontal, row)
        slider.setRange(config.SIZE_MIN, config.SIZE_MAX)
        slider.setSingleStep(10)
        slider.setPageStep(40)
        slider.setValue(self.pet_h)
        slider.valueChanged.connect(lambda v: self._preview_size(v, value))

        lay.addWidget(name)
        lay.addWidget(slider, 1)
        lay.addWidget(value)
        act.setDefaultWidget(row)
        return act

    def _preview_size(self, px: int, label: QLabel):
        label.setText(f"{px} px")
        self._pending_size = px
        if not self._size_timer.isActive():
            self._size_timer.start()      # 30ms 防抖：一次拖动会刷出上百个信号

    def _apply_size(self):
        if self._pending_size is not None:
            self.set_size(self._pending_size)

    def build_menu(self):
        menu = QMenu()

        # 朝向表只对照站姿给：坐着只有一张图、趴着是另一套三张，
        # 把 正面/左面/背面/右面 摆过去等于摆一排点了会指到不存在姿势的项。
        if self.posture == config.BASE_POSTURE and len(self.manifest.faces) > 1:
            faces = menu.addMenu("朝向")
            for i, key in enumerate(self.manifest.faces):
                act = faces.addAction(FACE_LABELS.get(key, key))
                act.setCheckable(True)
                act.setChecked(i == self.facing.value)
                act.setData(i)
            faces.triggered.connect(self._turn_later)

        move = menu.addMenu("移动")
        # 分组按"目的地谁说了算"，不按步态：随机在前（最常点），明确方向在后。
        # 以前这个子菜单叫"走路"，里面却装着"跑向远处" —— 标题和内容不符，
        # 用户说的"菜单混乱"最实在的一处就是这个。
        move.addAction("随便走走").triggered.connect(
            lambda: self._later(lambda: self.wants.request("roam_walk", manual=True)))
        # 跑步没有单独的"向左跑/向右跑"：它本来就是"这趟太远，改成跑"，
        # 方向该由去哪决定而不是由按钮决定。自发漫游走远了会自动用它。
        move.addAction("跑向远处").triggered.connect(
            lambda: self._later(lambda: self.wants.request("roam_run", manual=True)))
        move.addSeparator()
        # 用户点出来的每一项都必须和文案对得上：写"走走"就只是走。
        # 自动升级成跑步只发生在**自发**漫游上（那时没有文案可违约）。
        move.addAction("向左走").triggered.connect(
            lambda: self._later(lambda: self.wants.request("walk_left", manual=True)))
        move.addAction("向右走").triggered.connect(
            lambda: self._later(lambda: self.wants.request("walk_right", manual=True)))

        pose = menu.addMenu("姿态")
        pose.addAction("坐下").triggered.connect(
            lambda: self._later(lambda: self.request_posture("sitting")))
        # 打瞌睡是坐着打盹（素材里闭眼点头），不受"左右趴着"开关管：
        # 它用的就是正面那一套身体。
        pose.addAction("打瞌睡").triggered.connect(
            lambda: self._later(lambda: self.request_posture("dozing")))
        # 正面趴下是常驻动作，不受"左右趴着"开关管 —— 那个开关只管侧过去和四足爬走。
        pose.addAction("趴下").triggered.connect(
            lambda: self._later(lambda: self.request_posture("prone")))
        # 趴睡是**另一种身体**（侧躺、闭眼），素材只有左躺和右躺两套，
        # 所以它跟着"左右趴着"开关出现 —— 开关关掉时这个姿态根本不该存在。
        if self.side_prone:
            pose.addAction("趴着睡").triggered.connect(
                lambda: self._later(lambda: self.request_posture("sleeping")))
        # 一次性动作既进随机池也摆在这里：播一遍就结束，但做出来的形状就是一个姿势，
        # 归在"姿态"里比单挂一级菜单更贴合用户找它的直觉。
        pose.addAction("抱抱").triggered.connect(
            lambda: self._later(lambda: self.wants.request("hug", manual=True)))

        # 三段分隔各有理由：动作是"做一次"，开关是"能不能做"，窗口项是"怎么摆窗口"。
        # 「左右趴着」紧贴着"姿态"子菜单放，因为它就是趴着那一项能不能侧过去的前提。
        menu.addSeparator()
        prone = menu.addAction("左右趴着")
        prone.setCheckable(True)
        prone.setChecked(self.side_prone)
        prone.toggled.connect(lambda v: self._later(lambda: self.set_side_prone(v)))

        auto = menu.addAction("随机动作")
        auto.setCheckable(True)
        auto.setChecked(self.behaviors.enabled)
        auto.toggled.connect(lambda v: self._later(lambda: self.behaviors.set_enabled(v)))

        follow = menu.addAction("跟随鼠标")
        follow.setCheckable(True)
        follow.setChecked(self.follower.enabled)
        follow.toggled.connect(lambda v: self._later(lambda: self.set_follow(v)))

        menu.addSeparator()
        menu.addAction(self._size_action(menu))

        top = menu.addAction("总在最前")
        top.setCheckable(True)
        top.setChecked(self.always_on_top)
        top.toggled.connect(lambda v: self._later(lambda: self.set_always_on_top(v)))

        menu.addSeparator()
        menu.addAction("回到右下角", lambda: self._later(self.place_default))
        menu.addAction("退出", QApplication.quit)
        return menu
