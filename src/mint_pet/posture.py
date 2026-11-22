"""姿态：坐下、趴下、打瞌睡与趴着睡。

姿态和朝向是**两个正交的维度** —— 坐着/趴着/睡着时她仍然有朝向，只是这些姿态
素材大多只有正面，所以进入前必须先转回正面，期间也不许换朝向。

`PostureHold` 按姿态注册表跑 enter → holding → leave 三段：坐下有过渡素材，
趴下、打瞌睡与趴睡没有，进入就是硬切。驻留段循环哪一段由 manifest 标定，
可以是**按朝向对齐的列表**（趴着只有正面那一段会动；趴睡左右各一段）。
"""
from __future__ import annotations

import random

from . import config


def settle_side_prone(pet) -> None:
    """关掉"左右趴着"开关时的收场。

    在飞的爬走必须停 —— 它用的四足步态和侧趴朝向就是被开关管着的那套东西。
    左右趴着睡同理：它的两个朝向就是"侧躺"，开关关掉后这个姿态不该继续存在，
    所以让她起身（没有过渡素材，下一 tick 直接回到站立）。

    但**只是正面趴着她不用起来**：正面趴下不在这个开关的管辖范围内。
    重新 `enter_posture("prone")` 会按当前开关重装朝向表（关掉时只剩正面）
    并画回正面趴，所以"从侧趴退回正趴"和"表里根本没有侧趴"是同一个动作。
    """
    if pet.crawler.busy:
        pet.walker.stop()
        pet.crawler.stop()
    sleeping = pet.holders.get("sleeping")
    if sleeping is not None and sleeping.busy:
        sleeping.stop()
    if pet.posture == "prone":
        pet.enter_posture("prone")
        # 换表之后 holder 还停在原来那一帧上（可能是已经不存在的侧趴朝向），
        # 驻留还开着就按新表重画。趴姿的微动素材只有正面那一段，重画就是正趴呼吸帧。
        holder = pet.holders.get("prone")
        if holder is not None and holder.phase == "holding":
            holder.redraw()


class PostureWants:
    """姿态、移动和抱抱共用一个待执行意图，后来的手动请求覆盖前一个。"""

    def __init__(self, pet):
        self.pet = pet
        self.key = None
        self.left = 0
        self._start = None

    def clear(self) -> None:
        self.key, self.left, self._start = None, 0, None

    def request(self, key: str, manual: bool = False) -> bool:
        pet = self.pet
        holder = pet.holders.get(key)
        actions = {
            "hug": pet.oneshot.start,
            "walk_left": lambda: pet.walker.start(-1),
            "walk_right": lambda: pet.walker.start(1),
            "roam_walk": lambda: pet.walker.roam(gait="walk"),
            "roam_run": lambda: pet.walker.roam(gait="run"),
        }
        start = holder.start if holder else actions.get(key)
        if start is None or (key == "sleeping" and not pet.side_prone):
            return False
        if holder and holder.busy and holder.phase != "leaving":
            self.clear()
            return True
        self.clear()
        if manual:
            pet.follower._halt()
            pet.walker.stop()
            pet.crawler.stop()
            pet.oneshot.stop()
            for h in pet.holders.values():
                if h is not holder:
                    h.stop()
        if start():
            return True
        self.key, self._start = key, start
        self.left = config.ticks(config.POSTURE_WAIT_S)
        return True

    def drain(self) -> None:
        if self.key is None:
            return
        holder = self.pet.holders.get(self.key)
        self.left -= 1
        if (holder and holder.busy and holder.phase != "leaving") or self.left <= 0:
            self.clear()
            return
        if self._start():
            self.clear()


class PostureHold:
    """一种姿态的驻留状态机：转回正面 → 进入 → 驻留微动 → 退出。

    进入/退出各是一段一次性动画（`enter` 的前半与倒放），驻留是循环动画。
    没有过渡素材时 enter=None，直接切静帧。

    **过渡帧号和驻留循环帧号必须分开两个计数器。** 共用一个的话，驻留期间它一路
    递增到几百，起身倒放就从几百往下减：既慢到几十秒起不来，又会去请求
    `frame_250.png` 这种根本不存在的文件（实测就是这么把两条断言打挂的）。
    """

    def __init__(self, pet, key: str, hold_s: tuple, enter_anim: str | None = None,
                 loop_anim: str | None = None, rng: random.Random | None = None):
        self.pet = pet
        self.key = key
        self.hold_s = hold_s
        self.enter_anim = enter_anim
        self.loop_anim = loop_anim
        self.rng = rng or random.Random()
        self.phase = "off"          # off / turning / entering / holding / leaving
        self.enter_frame = 0        # 过渡段（正放=坐下，倒放=起身）的帧号
        self.enter_from = 0         # 过渡段的起点帧：素材开头那几帧她只是站着没事干
        self.enter_step = 1         # 过渡段每次往前跳几帧（= 素材帧距，24fps 配 80ms 节拍是 2）
        self.loop_frame = 0         # 坐着微动循环的帧号
        self.left = 0
        self.settle = 0
        self.total = 0
        self.done = None

    @property
    def busy(self) -> bool:
        return self.phase != "off"

    def start(self, done=None) -> bool:
        pet = self.pet
        # 左右趴着睡是侧躺的身体，只在"左右趴着"开关打开时存在。
        # 菜单已经藏了入口，这里再挡一道：随机池和排队意图也不该绕过开关。
        if self.key == "sleeping" and not pet.side_prone:
            return False
        # 爬走期间她是趴着的，但那是 Crawler 在用姿态，不是驻留：
        # 这里必须直接问 crawler，不能只靠"walker 忙"间接挡住——
        # 那是个恰好成立的前提，不是这条规则的理由。
        crawler = getattr(pet, "crawler", None)
        oneshot = getattr(pet, "oneshot", None)
        if (self.busy or pet.facing.turning or pet.walker.busy
                or pet.active_posture_busy() or (crawler is not None and crawler.busy)
                or (oneshot is not None and oneshot.busy)):
            return False
        anim = pet.manifest.anim(self.enter_anim) if self.enter_anim else None
        self.done = done
        self.loop_frame = 0
        self.left = config.ticks(self.rng.uniform(*self.hold_s))
        if anim:
            self.settle = int(anim.get("settle_at", 0))
            self.total = int(anim["frames"])
            # 过渡段按素材自己的帧距播（24fps 素材 → 80ms 节拍取 2）。
            # 逐帧播等于放慢一倍：坐下视频里"坐下去之后挪两下定住"那一段会被拉成
            # 两秒的磨蹭，整段坐下要五秒才结束。管线把交接点定在驻留窗口起点，
            # 这一段必须放得完又不能拖，就按帧距走。
            self.enter_step = max(1, int(anim.get("stride", 2)))
            # 起点由管线按"最像站立姿势图"标定，不是第 0 帧：素材开头她只是站着，
            # 而且比标准站姿图大 3.3%，从第 0 帧开播就是"先长高一下再坐下"。
            self.enter_from = min(int(anim.get("enter_at", 0)), self.settle - 1)
        else:
            self.settle = self.total = 0
            self.enter_from = 0
            self.enter_step = 1
        self.enter_frame = self.enter_from
        self.phase = "turning"
        if pet.facing.value != config.FRONT_FACING:
            pet.facing.turn_to(config.FRONT_FACING, done=self._begin)
        else:
            self._begin()
        return True

    def _begin(self):
        if self.phase != "turning":
            return
        pet = self.pet
        pet.enter_posture(self.key)
        faces = pet.posture_faces(self.key)
        if len(faces) > 1 and pet.manifest.posture_turn(self.key) is None:
            # 左右趴着睡没有转身素材（侧躺之间没有过渡画面），进入时一律落在 faces[0]
            # 就等于"永远只朝左躺"，第二条素材白接。这里随机挑一侧；
            # 有转身素材的姿态（趴着）不能这么干 —— 硬 apply 会跳过那一转。
            pet.facing.apply(self.rng.randrange(len(faces)))
        if self.enter_anim and self.settle:
            self.phase = "entering"
        else:
            self.phase = "holding"
            self._show_loop()

    def _loop_anim(self):
        """当前朝向下该播哪一段驻留微动；None = 这个朝向没有微动，停在静姿图上。

        清单里的 `idle` 可以是单个名字（坐着、打瞌睡），也可以是**按朝向对齐的列表**
        （趴着 = [正面微动, 无, 无]；左右躺睡 = [向左睡, 向右睡]）。
        列表必须由朝向索引来取：拿整段循环当"这个朝向的微动"会把她画成另一个身体。
        """
        idle = self.pet.manifest.posture_idle(self.key)
        if isinstance(idle, (list, tuple)):
            i = self.pet.facing.value
            return idle[i] if 0 <= i < len(idle) else None
        return idle if idle else self.loop_anim

    def _show_loop(self):
        """画驻留段里当前那一帧。循环范围由管线标定在 manifest 的 `loop` 里。

        不能想当然地拿 `[settle, 末)` 当"坐着微动"：坐下视频坐定之后其实还接了
        一整段抬头伸展、一段冻结、一段两姿态来回抖，直接循环就是"坐着坐着自己扭"。
        管线找不到与坐定帧连续的稳定段时会给出单帧循环，这里就老老实实停住。

        稳定段是**来回播**而不是播完接回起点：那一段本来就是一次起伏，
        硬接回去会在端点"顿"一下；原路返回天然接得上。
        """
        pet = self.pet
        name = self._loop_anim()
        anim = pet.manifest.anim(name) if name else None
        if not anim:
            pet.show_posture(self.key)
            return
        lo, hi = anim.get("loop", [self.settle, int(anim["frames"])])
        n = max(1, hi - lo)
        if n == 1:
            pet.show_anim_frame(name, lo)
            return
        # 步长由管线按"来回一轮要落在 2.5~4.5 秒"标定（坐着那段窗口只有 18 帧，
        # 走 1 帧一跳就是 2.72s；趴着的呼吸段有 45 帧，走 1 会慢到 7 秒看不出在动）。
        # 清单没给才退回全局默认值。
        stride = max(1, int(anim.get("loop_stride", config.POSTURE_LOOP_STRIDE)))
        span = 2 * n - 2
        k = (self.loop_frame * stride) % span
        pet.show_anim_frame(name, lo + (k if k < n else span - k))

    def redraw(self) -> None:
        """改尺寸之后重画当前那一帧。**不许推进状态机** —— 换个尺寸不该让她多走一步。"""
        if self.phase in ("entering", "leaving") and self.enter_anim:
            self.pet.show_anim_frame(self.enter_anim,
                                     max(0, min(self.enter_frame, self.total - 1)))
        elif self.phase == "holding":
            self._show_loop()

    def stop(self):
        """用户上手：立刻起身。留在原地比"演完再起身"更诚实。"""
        if self.busy:
            self._leave()

    def _leave(self):
        if self.phase in ("holding", "entering"):
            self.phase = "leaving"
        elif self.phase == "turning":
            self._finish()

    def _finish(self):
        self.phase = "off"
        self.pet.enter_posture(config.BASE_POSTURE)   # 起身也要把朝向表和调度器换回来，
        self.pet.facing.apply(0)                      # 否则站着却只有趴着那 3 个朝向
        done, self.done = self.done, None
        if done:
            done()

    def tick(self) -> None:
        if self.phase == "off":
            return
        pet = self.pet
        if self.phase == "turning":
            if not pet.facing.turning:
                pet.facing.turn_to(config.FRONT_FACING, done=self._begin)
            return
        if pet.facing.turning:
            return
        if self.phase == "entering":
            pet.show_anim_frame(self.enter_anim, self.enter_frame)
            self.enter_frame += self.enter_step
            if self.enter_frame >= self.settle:
                self.enter_frame = self.settle - 1
                self.loop_frame = 0
                self.phase = "holding"
            return
        if self.phase == "leaving":
            if self.enter_anim and self.enter_frame > self.enter_from:
                # 倒放过渡段 = 站起来。按同一帧距往回退，起步那一下才不会突然变慢一倍。
                self.enter_frame = max(self.enter_frame - self.enter_step, self.enter_from)
                pet.show_anim_frame(self.enter_anim, self.enter_frame)
            else:
                self._finish()
            return
        # holding
        self.left -= 1
        if self.left <= 0:
            self._leave()
            return
        self.loop_frame += 1
        self._show_loop()
