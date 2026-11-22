"""自发行为编排：走路 / 全屏漫游 / 转圈，以及挑它们的随机调度器。

每个动作都是"phase + tick"的小状态机，**不开线程、不用 QTimer**，
全部由 stage 里那唯一的 80ms tick 驱动 —— 这样任何时刻的状态都能被测试逐 tick 推进。
"""
from __future__ import annotations

import math
import random

from . import config


class Walker:
    """一次行程：转成侧视朝向 → **斜着**走到目标点 → 转回收尾朝向。

    素材只有左右两套侧视步态，没有"朝镜头走 / 背镜头走"。所以纵向不单独走
    （竖着平移要么像贴墙爬要么像飘），而是**横向迈步的同时按比例带一点纵向位移**，
    也就是斜着走过去 —— 侧视小人斜向移动本来就成立。

    斜率必须有上限：太陡就成了竖着爬。纵向够不着时按斜率夹掉，
    宁可到不了那个点，也不把身子拧成一个不像走路的角度。
    """

    def __init__(self, pet, rng: random.Random | None = None):
        self.pet = pet
        self.rng = rng or random.Random()
        self.phase = "off"          # off / turning / walking / returning
        self.tx = 0
        self.ty = 0
        self.dirn = 0               # 横向行进方向：-1 左 / +1 右
        self.gait = "walk"          # 这一程的步态：walk 走 / run 跑（按行程长短自动选）
        self.face = 3               # 这一程要用的侧视朝向（索引随姿态变：站着右=3，趴着右=2）
        self.step_y = 0.0           # 每 tick 纵向位移，出发时按"两轴同时到达"算定
        self.base_y = 0.0           # 纵向逻辑位置（浮点，免得小步长被取整吃掉）
        self._shown_y = 0           # 上一 tick 真正写进窗口的 y，用来发现"被外力挪过"
        self.frame = 0
        self.home = None            # 行程收尾转到哪个朝向；None = 回到出发朝向
        self.keep_facing = False    # 收尾不转身，保持行进方向（跟随的连续小步用）
        self.done = None

    @property
    def busy(self) -> bool:
        return self.phase != "off"

    def _locked(self) -> bool:
        """坐着/趴着期间不许迈步。

        姿态素材只有正面，走路必须先转成侧视 —— 那等于把她从坐姿硬掰成站姿侧身，
        画面会当场跳掉。这条守卫漏掉过一次：实测"坐着期间走不了路"直接失败。
        """
        return (self.busy or self.pet.facing.turning or self.pet.active_posture_busy()
                or self.pet.oneshot.busy)

    def _side(self) -> str:
        return "right" if self.dirn > 0 else "left"

    def _can_walk(self) -> bool:
        """当前姿态有没有对应的侧视步态。趴着时四足走要总开关打开才有。"""
        return self.pet.manifest.has_gait(self.pet.posture, "walk")

    def _step_for(self, gait: str | None) -> int:
        """每 tick 推进多少像素。趴着是**爬**，步子比直立的短，所以单列一档 ——
        这条不是拍脑袋：爬走一个步态周期 22 帧、走路 23~25 帧，节拍几乎同频，
        可画面是压低的四足趴伏、前脚伸出很短，同节拍再同速度就是"爬着走出散步的步子"。"""
        return self._px(self._speed_for(gait))

    def _speed_for(self, gait: str | None) -> float:
        if self.pet.posture != config.BASE_POSTURE:
            return config.CRAWL_SPEED
        return config.RUN_SPEED if gait == "run" else config.WALK_SPEED

    def _speed(self) -> float:
        return self._speed_for(self.gait)

    def _pick_gait(self, dx: int) -> str:
        """这一程用走还是跑。

        判据是"按走路的步子这趟要走几秒"，不是像素距离 —— 像素会随显示尺寸变，
        而"要走 4 秒以上就该跑"才是人对远近的直觉。跑只给了站姿素材，
        所以趴着（四足爬走）永远只是走。
        """
        pet = self.pet
        if pet.posture != config.BASE_POSTURE or not pet.manifest.has_gait(pet.posture, "run"):
            return "walk"
        secs = abs(dx) / self._px(config.WALK_SPEED) * config.FPS_MS / 1000
        return "run" if secs > config.RUN_FROM_S else "walk"

    # ── 发起 ────────────────────────────────────────────────────────────
    def _px(self, per_tick: float) -> int:
        """按显示高等比换算每 tick 位移。菲比 §5.1 的教训：放大后不提速就是小碎步滑行。"""
        return max(1, round(self.pet.pet_h / config.DEFAULT_H * per_tick))

    def _legal(self, x: int, y: int) -> tuple:
        """把窗口坐标夹进屏幕可用区。目标点先过这一关，走的过程中就不可能越界。"""
        pet = self.pet
        return (max(pet.screen_left(), min(x, pet.screen_right() - pet.width())),
                max(pet.screen_top(), min(y, pet.screen_bottom() - pet.height())))

    def _clamp_dy(self, dy: int, dx: int) -> int:
        """纵向位移同时受斜率与绝对上限约束。斜率按 |dx| 算，所以横向没位移就没纵向。"""
        lim = min(abs(dx) * config.WALK_SLOPE, config.ROAM_MAX_DY)
        return int(max(-lim, min(lim, dy)))

    def start(self, dirn: int | None = None, seconds: float | None = None,
              done=None, home=None) -> bool:
        """横着走一段（不改变高度）。dirn / seconds 留空即随机。

        home 留空 = 走完回正面（`WALK_HOME_FACING`）。以前留空是"回落到出发前朝向"，
        于是"勾左面 → 向右走"走完她自己又转回左面，和"随便走走"回正面不一致。
        """
        if self._locked() or not self._can_walk():
            return False
        if dirn is None:
            dirn = self._pick_dir()
        if not dirn:
            return False
        if seconds is None:
            seconds = self.rng.uniform(*config.WALK_S)
        pet = self.pet
        dist = self._step_for("walk") * config.ticks(seconds)
        tx, ty = self._legal(pet.x() + (dist if dirn > 0 else -dist), pet.y())
        # 菜单上写的是"向左走/向右走"，那就一定只是走 —— 让 _begin 自动升级成跑
        # 会让菜单文案和实际动作对不上。跑只由漫游这类"自己决定去哪"的行程触发。
        return self._begin(tx, ty, done,
                           config.WALK_HOME_FACING if home is None else home, gait="walk")

    def start_to(self, x: int, y: int, done=None, home=None, gait: str | None = None,
                 keep_facing: bool = False) -> bool:
        """斜着走到某个点。纵向够不着时按斜率夹掉，落点会比目标更贴近原高度。

        gait 留空 = 按行程长短自动选走/跑；测试和"明确只要走"的调用方可以钉死。
        keep_facing = 到了不转回收尾朝向。跟随鼠标是连续小步，每步都转一次身
        会把她原地拧成陀螺；保持行进方向，下一段反向时再自然转过去。
        """
        if self._locked() or not self._can_walk():
            return False
        pet = self.pet
        tx, ty = self._legal(x, y)
        dx = tx - pet.x()
        if not dx:
            return False                      # 纯纵向去不了：没有上下走的步态
        ty = self._legal(tx, pet.y() + self._clamp_dy(ty - pet.y(), dx))[1]
        return self._begin(tx, ty, done, home, gait, keep_facing)

    def retarget(self, x: int, y: int, gait: str | None = None) -> bool:
        """把**在飞的这一趟**改道到新目标。跟随鼠标要用：目标每 tick 都在动。

        不这么做的话她走的是"出发时算的那个点"，走完再重新看一眼 ——
        观感就是走几步停一下、永远慢半拍。改道之后她是**连续**朝光标走的，
        停不停由 `Follower` 按死区决定，而不是由"到没到那个旧终点"决定。

        方向反了必须重新转身：素材只有左右两套侧视步态，没有"倒着走"。
        """
        if self.phase not in ("turning", "walking"):
            return False
        pet = self.pet
        tx, ty = self._legal(x, y)
        dx = tx - pet.x()
        if not dx:
            self.stop()
            return False
        ty = self._legal(tx, pet.y() + self._clamp_dy(ty - pet.y(), dx))[1]
        self.tx, self.ty = tx, ty
        if gait and gait != self.gait and pet.manifest.has_gait(pet.posture, gait):
            self.gait = gait
        self.step_y = (ty - pet.y()) / max(1, abs(dx) / self._px(self._speed()))
        side = "right" if dx > 0 else "left"
        if side != self._side():
            self.dirn = 1 if dx > 0 else -1
            self.face = pet.manifest.walk_facing(pet.posture, side)
            self.phase = "turning"
            pet.facing.turn_to(self.face, done=self._walk)
        return True

    def _roam_target(self, gait: str | None = None):
        """**构造**一个可行的漫游目标点，没地方可走就返回 None。

        是构造而不是随机取了再筛：斜率下限（纵向至少要走 ROAM_MIN_PX，
        而纵向 = |dx|·斜率）和时长上限合起来把可行域压成 |dx| 的一个窄带，
        全屏幕均匀抽样会连续六次都落带外 —— 实测就是这么偶发发起不了的。

        单独拆出来是给 `Crawler.start()` 用的：**先确认有地方爬，再改姿态**。
        """
        pet = self.pet
        step = self._step_for(gait)
        # 纵向要走得出 ROAM_MIN_PX，斜率就反推出横向至少得走这么远
        dx_min = max(config.ROAM_MIN_DX, config.ROAM_MIN_PX / config.WALK_SLOPE)
        dx_max = config.ROAM_MAX_S * 1000 / config.FPS_MS * step
        room = {-1: pet.x() - pet.screen_left(),
                1: pet.screen_right() - pet.width() - pet.x()}
        ok = [d for d in (-1, 1) if room[d] >= dx_min]
        if not ok or dx_max < dx_min:
            return None
        dirn = self.rng.choice(ok)
        dx = int(self.rng.uniform(dx_min, min(dx_max, room[dirn])))
        lim = min(dx * config.WALK_SLOPE, config.ROAM_MAX_DY)
        dy = int(self.rng.uniform(config.ROAM_MIN_PX, lim) * self.rng.choice((-1, 1)))
        return self._legal(pet.x() + dirn * dx, pet.y() + dy)

    def roam(self, done=None, gait: str | None = None) -> bool:
        """在屏幕上挑一个点斜着走过去。这就是"她自己溜达"。

        gait 留空 = 按行程远近自动选；菜单上"跑向远处"钉成 run。
        """
        target = self._roam_target(gait)
        if target is None:
            return False
        # 到达后转身面向镜头：这是"我回来啦"的收尾，比转回出发朝向讨喜
        return self.start_to(target[0], target[1], done=done, gait=gait)

    def _pick_dir(self) -> int:
        """随机挑一侧，但排除已经没有余量的那一侧。

        不排的话贴着右墙随机到"向右走"，她会原地踩好几秒步态 —— 观感就是卡住了。
        """
        pet = self.pet
        room = {-1: pet.x() - pet.screen_left(),
                1: pet.screen_right() - (pet.x() + pet.width())}
        ok = [d for d in (-1, 1) if room[d] > pet.width() * 0.6]
        return self.rng.choice(ok) if ok else 0

    def _begin(self, tx: int, ty: int, done, home, gait: str | None = None,
               keep_facing: bool = False) -> bool:
        pet = self.pet
        dx = tx - pet.x()
        if not dx:
            return False
        self.tx, self.ty = tx, ty
        self.dirn = 1 if dx > 0 else -1
        self.gait = gait or self._pick_gait(dx)
        self.done = done
        self.keep_facing = keep_facing
        # 收工朝向的唯一默认值放在这里：三条入口（菜单定向走、随便走走、爬走）
        # 以前各有各的兜底，"向左走"回落到出发朝向、"随便走走"回正面，同一个动作两套语义。
        self.home = config.WALK_HOME_FACING if home is None else home
        self.base_y = float(pet.y())
        self._shown_y = pet.y()
        # 纵向每 tick 走多少 = 总纵向 ÷ 横向要走的 tick 数，保证两轴同时到达
        self.step_y = (ty - pet.y()) / max(1, abs(dx) / self._px(self._speed()))
        self.phase = "turning"
        # 朝向索引只在当前姿态内有意义：站着的"朝右"是 3，趴着的"朝右"是 2。
        # 存进 self.face 的必须是同一个东西，否则 _guard_stolen_callback 重新发起这一转时
        # 会拿站立的索引去查趴姿的朝向表。
        self.face = pet.manifest.walk_facing(pet.posture, self._side())
        pet.facing.turn_to(self.face, done=self._walk)
        return True

    def _walk(self):
        if self.phase != "turning":
            return
        self.phase = "walking"
        self.frame = 0

    def _finish_trip(self):
        pet = self.pet
        if self.keep_facing:
            # 收尾不转身 ≠ 什么都不用管：正在飞的那一转还挂着 `_walk`，
            # 放任它跑完就会朝**出发时算的旧目标**迈步 —— 实测"跟随途中回到右下角"
            # 之后她又自己走开 6px。把那一转的 done 换成 `_end`：转完就收工，
            # 朝向仍停在行进方向（这正是 keep_facing 要的）。
            if self.phase == "turning" and pet.facing.turning:
                self.phase = "returning"
                pet.facing.turn_to(self.face, done=self._end)
            else:
                self._end()
            return
        self.phase = "returning"
        home = self.home
        # 必须在**有转身在飞**时也走 turn_to：那一转的 done 挂着 `_walk`，
        # 就地短路的话它照样会触发，把她重新推进 walking 朝旧目标走 ——
        # 实测"向右走途中拖到左上"因此又走出 476px。重新发起一转会把 done 换掉。
        if pet.facing.value == home and not pet.facing.turning:
            self._end()
        else:
            pet.facing.turn_to(home, done=self._end)

    def _end(self):
        self.phase = "off"
        # 步态归位。留着 "run" 不危险（下一次 _begin 会重算），但一个"已经不走路了
        # 却还标着跑步态"的字段会让任何事后读它的代码读到假状态 —— 少一类 bug 就少一处心算。
        self.gait = "walk"
        self.keep_facing = False
        done, self.done = self.done, None
        if done:
            done()

    def stop(self):
        """中止行程：用户上手拖她、或程序把她挪到别处（回到右下角）。

        两条都要，缺一不可：
        1. 不能写成"走完这段再停" —— 那会在用户拖着窗口的时候，她的 tick 还在往
           另一个方向 move 同一个窗口，两边抢手。
        2. 连"还没迈步、正在转身"那一段也要作废。目标点是出发时算的绝对坐标，
           她一旦被挪走，那个点就成了别人的位置，继续走过去就是"位置错位"。
        """
        if self.phase in ("walking", "turning"):
            self._finish_trip()

    # ── 每帧 ────────────────────────────────────────────────────────────
    def _guard_stolen_callback(self) -> None:
        """兜底：转身已经落地了，却没收到回调。

        `Facing.turn_to(q)` 不带 done 时会把**上一个调用者挂着的 done 一起顶掉**
        （用户中途单击换朝向、或从菜单直选朝向都会这样）。那个语义是故意的 ——
        被打断的一转不该再兑现完成契约 —— 但行程状态机不能因此信不过自己：
        只等回调的话她就永远卡在 turning/returning，既不呼吸也不再自发行动。
        实测"向右走→单击循环→菜单转正面→拖到左上"就是这么永久冻住的。
        """
        pet = self.pet
        if pet.facing.turning:
            return
        if self.phase == "turning":
            # 重新发起这一转，而不是硬迈步：朝向没到位就走路等于横着滑
            pet.facing.turn_to(self.face, done=self._walk)
        else:
            self._end()

    def tick(self) -> None:
        if self.phase in ("turning", "returning"):
            self._guard_stolen_callback()
            return
        if self.phase != "walking":
            return
        pet = self.pet
        # 目标点每 tick 重夹一次。两个理由：
        # 1. 途中改尺寸会缩小可放范围，出发时算好的绝对目标可能已落到屏幕外
        #    （矩阵实测纵向越界 193px）；
        # 2. 撞边早停时若不夹，她会停在越过边界 1px 处。
        tx = int(min(max(self.tx, pet.screen_left()), pet.screen_right() - pet.width()))
        ty = int(min(max(self.ty, pet.screen_top()), pet.screen_bottom() - pet.height()))
        nx = pet.x() + self._px(self._speed()) * self.dirn
        arrived_x = nx * self.dirn >= tx * self.dirn
        if arrived_x:
            nx = tx

        if not self.step_y:
            ny, arrived_y = pet.y(), True
        else:
            if pet.y() != self._shown_y:
                # 窗口被外力挪过（改尺寸会按底边重排）。不重设逻辑位置的话，
                # 下一 tick 会把 y 写回旧 base_y —— 她当场跳回原来的高度。
                self.base_y = float(pet.y())
            # 用"朝目标取 min/max"而不是"判是否越过"：目标被重夹到身后时，
            # 穿越判据会永远不再成立，她就一路走出屏幕外并卡死（矩阵实测 y=-141）。
            lo, hi = pet.screen_top(), pet.screen_bottom() - pet.height()
            self.base_y = min(max(self.base_y, lo), hi)
            self.base_y = (min(self.base_y + self.step_y, ty) if self.step_y > 0
                           else max(self.base_y + self.step_y, ty))
            ny = int(round(self.base_y))
            arrived_y = self.base_y == float(ty)
        pet.move(nx, ny)
        self._shown_y = ny

        anim = pet.manifest.gait_anim(pet.posture, self._side(), self.gait)
        if not anim:
            self._finish_trip()
            return
        # 循环长度可能比素材短：管线会把接不上相的步态截到整数个周期（见 manifest 的 gait.loop）。
        # 拿 anim["frames"] 取模的话，每循环一次相位就错开一次，观感是"走两步绊一下"。
        g = anim.get("gait", {})
        lo, hi = g.get("loop", [0, int(anim["frames"])])
        n = max(1, hi - lo)
        # 起步相位由管线标定 = 整段里最像"起步前那个静态朝向"的一帧（gait.phase0）。
        # 素材第 0 帧不保证是标准相位：向右跑第 0 帧离"右面"差 1.192，第 22 帧只差 0.882，
        # 从 0 起步就是迈步那一拍"猛地一歪"。
        k = (g.get("phase0", 0) - lo + self.frame * anim.get("stride", 2)) % n
        pet.show_anim_frame(anim["name"], lo + k)
        self.frame += 1

        # 撞边不再单独判：目标点已经夹进可放范围，走到 tx/ty 就是走到边。
        # 少一条独立的停止条件，就少一处会和"到达"互相抢着结束行程的地方。
        if arrived_x and arrived_y:
            self._finish_trip()


class Follower:
    """跟随鼠标：**连续**朝光标走，光标停了她才停。

    发起一趟之后每 tick 把终点改到光标当前位置（`Walker.retarget`），
    而不是"走完一程再重新看一眼"。后者实测就是用户说的"不实时"：
    一程 3~5 秒，她每程都要重新起步、重新判断方向，观感是慢半拍 + 一顿一顿。

    纵向仍由 `WALK_SLOPE` 夹（她没有朝镜头走的步态），所以正上/正下方够不着，
    这一点靠横向先靠过去再补，不硬飘。
    """

    def __init__(self, pet):
        self.pet = pet
        self.enabled = False
        self.following = False      # 当前这一趟是不是跟随发起的

    def set_enabled(self, on: bool) -> None:
        on = bool(on)
        self.enabled = on
        if not on:
            self._halt()

    def _halt(self) -> None:
        """收掉跟随那一趟：关掉开关 / 用户点了别的动作 / 进了死区，都走这一条。"""
        pet = self.pet
        if pet.walker.busy and self.following:
            pet.walker.stop()
        self.following = False

    def _center(self) -> tuple:
        pet = self.pet
        return pet.x() + pet.width() // 2, pet.y() + pet.height() // 2

    def _goal(self) -> tuple:
        """让窗口中心落在光标上的目标坐标。"""
        pet = self.pet
        mx, my = pet.cursor_pos()
        return mx - pet.width() // 2, my - pet.height() // 2

    def _dist(self) -> float:
        cx, cy = self._center()
        mx, my = self.pet.cursor_pos()
        return math.hypot(mx - cx, my - cy)

    def _blocked(self, following: bool = False) -> bool:
        """不该跟的几种时刻。用户点了姿态（`wants` 里有待兑现的意图）必须让路，
        否则她会一边排队坐下一边继续走，那个意图永远等不到空档。

        `following=True` 时**不把"正在转身"算作阻塞**：转身本来就是跟随起步的
        第一步（先转成侧视再迈步）。把它当"忙"会让她刚起步就把自己叫停、
        隔 3 tick 再起步 —— 实测这就是"跟随一顿一顿"的真正来源。
        """
        pet = self.pet
        if following:
            return (pet._dragging or pet._menu_open or pet.wants.key is not None
                    or pet.posture != config.BASE_POSTURE
                    or pet.active_posture_busy() or pet.crawler.busy or pet.oneshot.busy)
        return (self._blocked(True) or pet.facing.turning or pet.walker.busy)

    def tick(self) -> None:
        if not self.enabled:
            return
        pet = self.pet
        scale = pet.pet_h / config.DEFAULT_H
        if self.following and not pet.walker.busy:
            self.following = False          # 这一趟被别人（拖拽/回位/姿态）收掉了
        if self.following:
            if self._blocked(following=True):
                self._halt()
                return
            if self._dist() <= config.FOLLOW_DEADZONE_PX * scale:
                self._halt()                # 到跟前就地停：保持行进方向，不转身
                return
            if pet.walker.phase in ("turning", "walking"):
                pet.walker.retarget(*self._goal(),
                                    gait="run" if self._dist() > config.FOLLOW_RUN_PX * scale
                                    else "walk")
            return
        if self._blocked():
            return
        if self._dist() <= config.FOLLOW_DEADZONE_PX * scale:
            return
        tx, ty = self._goal()
        gait = "run" if self._dist() > config.FOLLOW_RUN_PX * scale else "walk"
        self.following = bool(pet.walker.start_to(tx, ty, keep_facing=True, gait=gait))


class Crawler:
    """四足爬走：趴下 → 用四足步态斜着走到一个点 → 站起来。

    不复用 `PostureHold`：那个的语义是"驻留期间独占，走路必须让路"，
    而爬走恰恰要**边趴着边走**。趴下又没有过渡素材，所以这里直接切姿态就够了。
    步态本身不用改 —— `Walker` 按当前姿态查步态，趴着时自然拿到 walk4_*。
    """

    def __init__(self, pet):
        self.pet = pet
        self.phase = "off"          # off / going / getting_up

    @property
    def busy(self) -> bool:
        return self.phase != "off"

    def start(self, done=None) -> bool:
        pet = self.pet
        if (self.busy or pet.facing.turning or pet.walker.busy or pet.active_posture_busy()
                or pet.oneshot.busy):
            return False
        if not pet.side_prone or not pet.manifest.has_gait("prone", "walk"):
            return False
        target = pet.walker._roam_target()
        if target is None:
            # **先确认有地方爬，再改姿态。** 旧顺序是"趴下 → roam 失败 → 立刻起身"，
            # 观感就是趴一下又弹起来；而且 `start()` 照样返回 True，
            # 于是行为池把一次什么都没发生的尝试记成了"她爬走了一趟"
            # （实测让 crawl 的挑中占比从权重该有的 27% 虚高到 32%）。
            return False
        self.done = done
        pet.enter_posture("prone")
        self.phase = "going"
        pet.walker.start_to(target[0], target[1], done=self._arrived)
        return True

    def _arrived(self):
        self.phase = "getting_up"
        self.pet.enter_posture(config.BASE_POSTURE)
        self.phase = "off"
        done, self.done = self.done, None
        if done:
            done()

    def stop(self):
        if self.busy:
            self._arrived()


class BehaviorPool:
    """随机挑一个自发动作来做，并负责三层节流。

    三层节流各自治一种"看着烦"：
      hold  —— 动作收尾后再占一会，治"做完立刻再做"的抽搐感；
      cd    —— 每类动作独立冷却，治"整屏就在那走来走去"；
      quiet —— 用户刚操作过就闭嘴，治"我在拖窗口她非要抢着转身"。
    挑动作时会给"上一轮那个"降权（`BEHAVIOR_REPEAT_PENALTY`），避免连续五次都在走路。
    各动作实际占多少时间、多久动一次，用 `tools/behavior_check.py` 量，别拿权重当占比读。
    """

    def __init__(self, pet, rng: random.Random | None = None):
        self.pet = pet
        self.rng = rng or random.Random()
        self.enabled = config.AUTO_BEHAVIOR
        self.hold = 0
        self.quiet = 0
        self.cd = {name: 0 for name in config.BEHAVIOR_WEIGHTS}
        self.next_in = self._gap()
        self.last = None
        self.fired = 0              # 观测点：测试靠它确认"随机"真的发生过
        self._was = False           # 上一 tick 执行者忙不忙，用来抓下降沿

    def _gap(self) -> int:
        return config.ticks(self.rng.uniform(*config.BEHAVIOR_GAP_S))

    def note_user(self) -> None:
        """用户碰过一下：接下来一段时间别自发行动。"""
        self.quiet = config.ticks(config.BEHAVIOR_QUIET_S)

    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)
        if not on:
            self.next_in = self._gap()

    def actors_busy(self) -> bool:
        """执行者本身在动。判定必须和 busy() 分开，见 tick 里那条注释。"""
        pet = self.pet
        return bool(pet.facing.turning or pet.walker.busy or pet.active_posture_busy()
                    or pet.crawler.busy or pet.oneshot.busy)

    def busy(self) -> bool:
        return self.hold > 0 or self.actors_busy()

    def available(self, name: str) -> bool:
        if self.cd[name] or not config.BEHAVIOR_WEIGHTS.get(name):
            return False
        if name in ("crawl", "sleep") and not self.pet.side_prone:
            # 爬走要侧趴朝向 + 四足步态；左右趴着睡本身就是侧躺的身体。
            # 两者都归"左右趴着"开关管，正面趴下（prone）不归它管。
            return False
        if name in ("sit", "doze", "prone", "sleep"):
            # 姿态期间不再挑姿态动作：她正坐着，不该坐着坐着又趴下。
            return not self.pet.active_posture_busy() and not self.pet.crawler.busy
        if name == "crawl":
            return not self.pet.active_posture_busy()
        return True

    def pick(self):
        names = [n for n in self.cd if self.available(n)]
        if not names:
            return None
        # 上一轮那个**降权**，不是踢出候选。
        # 踢出的代价实测很硬：候选只剩 roam/sit 两类时，"不许连着两次同一个"
        # 就等于"必须严格交替"—— 重复率恒为 0，配的 3:2 权重也完全不生效
        # （量出来各 50.0%）。降权既留住"别连着五次都在走路"的原意，
        # 又让分布重新由权重说了算，而且允许偶尔连做两件事（那才像活物）。
        w = [config.BEHAVIOR_WEIGHTS[n] * (config.BEHAVIOR_REPEAT_PENALTY
                                           if n == self.last else 1.0) for n in names]
        return self.rng.choices(names, weights=w)[0]

    def actors(self) -> dict:
        """动作名 → 执行者。这张表就是"池子里有哪些动作"的唯一名单。

        单独成方法而不是埋在 `run()` 里，是为了让冒烟能拿它和
        `config.BEHAVIOR_WEIGHTS` 对账 —— 两边漏一个都是"配了没人做"或"做了没配"。
        """
        pet = self.pet
        return {"roam": pet.walker.roam,
                "sit": lambda: pet.holders["sitting"].start(),
                "doze": lambda: pet.holders["dozing"].start(),
                "prone": lambda: pet.holders["prone"].start(),
                "sleep": lambda: pet.holders["sleeping"].start(),
                "crawl": lambda: pet.crawler.start(),
                "hug": pet.oneshot.start}

    def run(self, name: str) -> bool:
        """入口：菜单手动点和随机调度共用这一条。"""
        actor = self.actors().get(name)
        if not actor or not actor():
            return False
        # 冷却**从收尾开始算**，不是从开始算。
        # 从开始算的话，动作时长会直接吃掉冷却：坐下整段约 21s（过渡 5s + 驻留 16.5s），
        # 45s 冷却只剩 24s 落在她起身之后；走路整段约 7s，30s 冷却剩 23s —— 两者几乎相等，
        # 于是"刚做过的那个一定还在冷却"，实测连续重复率恒为 0，
        # 她就是走走·坐坐·走走·坐坐的钟摆，权重和降权全都白配。
        self.last = name
        self.fired += 1
        return True

    def tick(self) -> None:
        self.hold = max(0, self.hold - 1)
        self.quiet = max(0, self.quiet - 1)
        for k in self.cd:
            self.cd[k] = max(0, self.cd[k] - 1)

        # "刚收尾"必须按**执行者**的下降沿判定，不能按含 hold 的 busy()：
        # 那样 hold 自己会让 busy 为真，下一 tick 又看到"从忙转闲"，再补一个 hold ——
        # 池子被自己设的 hold 永久锁死，一个自发动作都发不出来。
        working = self.actors_busy()
        if self._was and not working:
            self.hold = config.ticks(config.BEHAVIOR_HOLD_S)
            self.next_in = self._gap()       # 做完一件事就重新歇一整段
            if self.last:
                self.cd[self.last] = config.ticks(config.BEHAVIOR_COOLDOWN_S[self.last])
        self._was = working
        if self.busy():
            return
        # 跟随期间池子完全让位：否则她刚跟到光标旁边，又自己溜达走了。
        # 不在这里动 enabled / next_in —— 关掉跟随后要接着原来的节奏，
        # 而不是因为"暂停了几分钟"立刻蹦出一个动作。
        if self.pet.follower.enabled:
            return
        if not self.enabled or self.quiet or self.pet._dragging:
            return
        self.next_in -= 1
        if self.next_in > 0:
            return
        name = self.pick()
        done = bool(name and self.run(name))
        # 没做成（贴边、全在冷却）就很快再试，别为一次空手等上一整段空闲
        self.next_in = self._gap() if done else config.ticks(config.BEHAVIOR_RETRY_S)
