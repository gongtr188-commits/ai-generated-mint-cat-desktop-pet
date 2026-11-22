"""桌宠窗口：透明置顶无边框 + 唯一 80ms 帧循环 + 素材绘制。

窗口尺寸直接等于画布尺寸——所有素材共用统一画布（tools/build_assets.py 的
CANVAS_W/H），所以这里不需要按每张图单独算宽度和偏移，素材天然对齐。
"""
from __future__ import annotations

from PyQt5.QtCore import QRectF, Qt, QTimer
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication, QLabel, QWidget

from . import config
from .actions import OneShot
from .behavior import BehaviorPool, Crawler, Follower, Walker
from .facing import Facing, InstantTurn, LinearTurn, VideoTurn
from .idle import IdlePlayer
from .posture import PostureHold, PostureWants, settle_side_prone
from .interaction import InteractionMixin


class PetWindow(InteractionMixin, QWidget):
    def __init__(self, manifest, store):
        super().__init__()
        self.manifest = manifest
        self.store = store
        self.variant = None
        self.pet_h = config.DEFAULT_H
        self.always_on_top = True
        self.init_interaction()

        self.setWindowFlags(self._window_flags())
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 桌宠经常 show/hide（探头、取快递）。少了这行，每次出现都会抢走
        # 用户正在输入的那个窗口的焦点。
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setWindowTitle("薄荷（猫猫）")

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignBottom | Qt.AlignHCenter)

        self.facing = Facing(self._on_facing_change, director=self._build_director())
        self.idle = self._build_idle()
        self.walker = Walker(self)
        self.crawler = Crawler(self)
        self.follower = Follower(self)
        self.oneshot = OneShot(self, config.HUG_ANIM)
        self.posture = config.BASE_POSTURE
        self.side_prone = config.SIDE_PRONE     # 开关：趴着时能不能侧过去（左右趴着 + 四足爬走）
        # 用哪个素材由清单说，不在这里再列一遍：坐下用 坐下.mp4 的过渡段和待机段，
        # 趴着没有过渡素材但有驻留微动。素材换了名字只改管线，不改这里。
        # `posture_idle` 可能是**按朝向对齐的列表**（趴着 / 趴睡），解析在 PostureHold。
        self.holders = {
            key: PostureHold(self, key, hold,
                             enter_anim=manifest.posture_enter(key),
                             loop_anim=manifest.posture_idle(key)
                                       or manifest.posture_enter(key))
            for key, hold in config.POSTURE_HOLD_S.items()
        }
        self.wants = PostureWants(self)         # 姿态意图队列，一次只存一个
        self.behaviors = BehaviorPool(self)
        self.resize_to_pet()
        self._on_facing_change(self.facing.value)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(manifest.fps_ms)

    # ── 窗口 ────────────────────────────────────────────────────────────
    def _window_flags(self):
        flags = Qt.FramelessWindowHint | Qt.SubWindow | Qt.WindowDoesNotAcceptFocus
        if self.always_on_top:
            flags |= Qt.WindowStaysOnTopHint
        return flags

    def resize_to_pet(self):
        # 画布是横向的（坐姿/趴姿比站姿宽得多），所以窗口不再是正方形：
        # 滑条调的是**高度**，宽度按画布比例派生。
        self.setFixedSize(self.pet_w, self.pet_h)
        self.label.setGeometry(0, 0, self.pet_w, self.pet_h)

    @property
    def pet_w(self) -> int:
        return int(round(self.pet_h * self.manifest.canvas.w / self.manifest.canvas.h))

    def content_rect(self) -> QRectF:
        """素材内容在窗口内的矩形（浮点，别取整：坐窗跟踪每轮都要重算，误差会累积）。

        窗口比素材宽（画布留白），所以将来"坐窗沿 / 贴屏幕边"必须按这个矩形
        算留白，直接拿窗口边缘对齐会看起来"离边一截"。
        """
        pose = self.current_pose()
        if not pose.content:
            return QRectF(self.rect())
        s = self.pet_h / self.manifest.canvas.h
        cw, ch = pose.content
        bottom = self.pet_h - self.manifest.canvas.baseline * s
        return QRectF((self.pet_w - cw * s) / 2, bottom - ch * s, cw * s, ch * s)

    def place_default(self):
        # 先作废行程再挪窗口：她的目标点是出发时算的绝对坐标，
        # 不取消的话下一 tick 又朝旧目标走回去，"回到右下角"当场就错位。
        self.walker.stop()
        # 跟随一并关掉：把她摆回右下角之后，开着的跟随会立刻把她拽回光标那边，
        # 这条菜单项就等于没生效。要跟随时用户再勾一次，语义清楚。
        self.set_follow(False)
        g = QApplication.desktop().availableGeometry(self)
        self.move(g.right() - self.width() - 80, g.bottom() - self.height() - 80)

    # ── 帧循环 ──────────────────────────────────────────────────────────
    def tick(self):
        self.facing.tick()
        self.follower.tick()
        self.walker.tick()
        for h in self.holders.values():
            h.tick()
        self.oneshot.tick()
        # 姿态意图要在 holder 之后、行为池之前：holder 刚收尾的这一 tick 就是它该兑现的
        # 那一 tick；而它必须赶在行为池之前，否则同一 tick 里池子会抢在她"想起要坐下"之前
        # 挑一个随机动作把位置又占掉。
        self._drain_posture()
        self.idle.tick()
        self.behaviors.tick()
    # ── 姿态 ────────────────────────────────────────────────────────────
    def active_posture_busy(self) -> bool:
        """任一姿态驻留中。走路和别的姿态都要让路。"""
        return any(h.busy for h in self.holders.values())

    def show_posture(self, posture_key: str, facing: int | None = None):
        """画某个姿态的静帧（姿态名直接就是姿势 key 的前缀）。

        默认画**当前朝向**那一张，而且要按开关裁过的朝向表来取：
        写死 0 的话，侧趴着驻留时每一 tick 都被重画成正趴 —— 看着就是她趴着趴着
        自己翻回正面；绕过 `posture_faces()` 则会取到被开关禁掉的姿势。
        """
        faces = self.posture_faces(posture_key)
        q = self.facing.value if facing is None else facing
        self.show_pose(self.manifest.pose(faces[q % len(faces)], self.variant))

    def posture_faces(self, posture: str | None = None) -> list:
        """该姿态下可用的朝向。开关关掉时趴着只剩正面一个 —— 这张表就是
        "左右趴着"这个开关的**唯一**落点：朝向索引到不了被禁姿势，
        转身调度器也换不到那份按三个姿势标定的 keys 表（见 _build_director）。

        不传参时**必须**按当前姿态裁。`current_pose()` 就是不带参数调它的：
        那一处漏掉裁减不会报错，只会让"关掉左右趴着之后"仍能索引到 prone_left。
        """
        posture = posture or self.posture
        faces = self.manifest.posture_faces(posture)
        if posture == "prone" and not self.side_prone:
            return faces[:1]
        return faces

    def _build_director(self, posture: str | None = None):
        """按姿态取转身调度器；标定过的帧在就用视频，否则退回静态硬切。

        站立用 转身360度（环形），趴着用 趴着转180度（线性弧）——
        两者连"从 A 到 B 要走多远"的定义都不一样，必须分开建模。
        素材没到位时程序照样能跑，素材一入库转身自动就活了。

        只有一个可用朝向时**必须**用 InstantTurn：视频调度器手里的 keys 表是按
        完整姿势集标定的（趴着 3 个），而 `start()` 是拿 `to % len(keys)` 取帧的 ——
        坐姿只有 1 张图，给它视频调度器就会拿索引去查不存在的姿势。
        """
        posture = posture or config.BASE_POSTURE
        if len(self.posture_faces(posture)) < 2:
            return InstantTurn()
        name = self.manifest.posture_turn(posture)
        anim = self.manifest.anim(name) if name else None
        if anim and anim.get("keys") and anim.get("path"):
            cls = LinearTurn if anim.get("linear") else VideoTurn
            return cls(anim["cycle"], anim["keys"], anim["path"],
                       lambda i, n=anim["name"]: self.show_turn_frame(n, i))
        return InstantTurn()

    def enter_posture(self, posture: str) -> None:
        """切姿态：换朝向集合 + 换转身调度器。"""
        self.posture = posture
        self.facing.set_posture(self.posture_faces(posture), self._build_director(posture))

    def request_posture(self, key: str, manual: bool = True) -> bool:
        """表达"我要换姿态"：能立刻做就立刻做，被占着就排队。逻辑在 `PostureWants`。

        默认 `manual=True`：这个入口就是菜单点的，用户点了一下就必须有反应。
        """
        return self.wants.request(key, manual)

    def clear_posture_want(self) -> None:
        self.wants.clear()

    def _drain_posture(self) -> None:
        self.wants.drain()

    def set_side_prone(self, on: bool) -> None:
        """开关：趴着时能不能侧过去。管的是 向左/向右趴着 两个朝向、趴着转180度、
        要用这两套朝向加四足步态的四足爬走，以及**左右趴着睡**（那本来就是侧躺的身体）。
        **正面趴下不归它管**，所以关掉时
        既不用把队列里的"趴下"作废，也不用把正趴着的她拽起来 —— 只退回到正面趴。

        爬走不需要在这里额外收尾：它不走意图队列，池子每 tick 用
        `available()` 重新问一次开关，关掉之后它自己就再也不会被挑中。
        """
        on = bool(on)
        if on == self.side_prone:
            return
        self.side_prone = on
        if not on:
            if self.wants.key == "sleeping":
                self.wants.clear()   # 排了半小时的队之后突然自己躺下睡，那是惊吓不是响应
            settle_side_prone(self)

    def set_follow(self, on: bool) -> None:
        """开关：跟随鼠标。打开时先把她从手里那件事里放出来，否则要等姿态驻留结束
        才看得见效果（趴睡一驻留就是 30 秒，用户会以为开关坏了）。

        关闭时不掐断正在走的那一小步：让她把脚下这一步走完再停，比原地钉住自然。
        """
        on = bool(on)
        if on == self.follower.enabled:
            return
        self.follower.set_enabled(on)
        if on:
            self.clear_posture_want()
            self.walker.stop()
            self.crawler.stop()
            self.oneshot.stop()
            for h in self.holders.values():
                h.stop()

    # ── 绘制 ────────────────────────────────────────────────────────────
    def show_anim_frame(self, anim_name: str, index: int):
        anim = self.manifest.anim(anim_name)
        if not anim:
            return
        # 帧号越界会直接 FileNotFoundError 把桌宠弄死。状态机自己已经保证在范围内，
        # 这里再夹一道：常驻进程里一个越界索引不该是致命错误。
        n = int(anim["frames"])
        index = max(0, min(int(index), n - 1))
        self.label.setPixmap(self.store.frame_pix(
            self.manifest.root / anim["dir"], index, self.pet_h))

    def show_turn_frame(self, anim_name: str, index: int):
        self.show_anim_frame(anim_name, index)

    def show_idle_frame(self, index: int):
        self.show_anim_frame(config.IDLE_ANIM, index)

    def _build_idle(self) -> IdlePlayer:
        anim = self.manifest.anim(config.IDLE_ANIM)
        if not anim:
            return IdlePlayer(0, 1, lambda i: None, lambda: False)
        return IdlePlayer(anim["frames"], anim.get("stride", config.IDLE_STRIDE),
                          self.show_idle_frame, self._idle_allowed)

    def _idle_allowed(self) -> bool:
        """待机微动的准入条件。任何一条不满足就完全停手，不许和别的渲染源抢。"""
        return (self.posture == config.BASE_POSTURE
                and self.facing.value == config.IDLE_FACING
                and not self.facing.turning
                and not self.walker.busy
                and not self.active_posture_busy()
                and not self.oneshot.busy
                and not self._dragging)

    def screen_left(self) -> int:
        return self._work_area().left()

    def cursor_pos(self) -> tuple:
        """鼠标当前全局坐标。**跟随层只认这个签名**，所以 `Follower` 不碰 Qt、能不开 Qt 就测。"""
        p = QCursor.pos()
        return (int(p.x()), int(p.y()))

    def screen_right(self) -> int:
        return self._work_area().right() + 1

    def screen_top(self) -> int:
        return self._work_area().top()

    def screen_bottom(self) -> int:
        return self._work_area().bottom() + 1

    def _work_area(self):
        """她可以走到的范围 = 屏幕可用区（已扣掉任务栏）。"""
        return QApplication.desktop().availableGeometry(self)

    def current_pose(self):
        """当前该画的静帧。朝向索引只在**当前姿态内**有意义，不能直接拿去查站立表。"""
        return self.manifest.pose(self.posture_faces()[self.facing.value], self.variant)

    def _on_facing_change(self, value: int):
        self.show_pose(self.current_pose())

    def closeEvent(self, e):
        """析构前把在飞的回调全部作废。

        姿态/行程状态机把完成契约挂在转身调度器上，而那些闭包引用的是本 widget。
        窗口先被 C++ 侧删掉、回调随后触发，就是解引用已销毁对象 —— PyQt 直接
        abort 整个进程且不抛 Python 异常（和点菜单必崩那次同一类，见开发文档 §10.1）。
        """
        self.timer.stop()
        for h in self.holders.values():
            h.done = None
        self.walker.done = None
        self.crawler.done = None
        self.oneshot.done = None
        self.facing.abort()
        super().closeEvent(e)

    def rerender(self):
        """改尺寸 / 换皮肤之后重画当前画面。

        转身中途必须重绘**当前那一帧视频**，直接 show_pose 会把她从半路拽回标准姿势。
        """
        director = self.facing.director
        holder = next((h for h in self.holders.values() if h.busy), None)
        if self.facing.turning and getattr(director, "shown", None) is not None:
            self.show_turn_frame(self.manifest.posture_turn(self.posture), director.shown)
        elif holder:
            holder.redraw()
        elif self.oneshot.phase == "playing":
            self.oneshot.redraw()
        elif self.idle.running:
            self.show_idle_frame(self.idle.shown)
        else:
            self.show_pose(self.current_pose())

    def show_pose(self, pose):
        self.label.setPixmap(self.store.pix(pose.path, self.pet_h, pose.k))

    def set_variant(self, variant: str | None):
        self.variant = variant
        self.idle.reset()
        self.rerender()

    # ── 尺寸 / 置顶 ─────────────────────────────────────────────────────
    def set_size(self, h: int):
        h = max(config.SIZE_MIN, min(config.SIZE_MAX, int(h)))
        if h == self.pet_h:
            return
        g = self.geometry()
        cx, bottom = g.center().x(), g.bottom()
        self.pet_h = h
        self.store.drop_scale_cache()
        self.resize_to_pet()
        self.move(cx - self.width() // 2, bottom - self.height())
        # 放大时按"底边不动 + 水平居中"重排，靠边的她会被推出屏幕外，必须夹回来
        self._clamp_to_screen()
        self.rerender()

    def set_always_on_top(self, on: bool):
        if on == self.always_on_top:
            return
        self.always_on_top = on
        # 改 flags 必须重新 show 才生效
        self.setWindowFlags(self._window_flags())
        self.show()
