"""一次性动作：转回正面 → 正放一遍素材 → 交还。

和 `PostureHold` 的区别是**没有驻留**：坐下要一直坐着直到时间到、再倒放起身；
拥抱（以及将来的伸懒腰、舔毛）播完就完了。素材本身也循环不了 ——
`微笑拥抱` 首尾接缝 11.40，是相邻帧差中位（0.92）的 12.4 倍（见 seam_check），
接成 loop 每圈硬跳一次，所以只能正放一遍。
（绝对值一直如此；倍数从 2.9× 涨到 12.4× 是因为头部锚定把逐帧抖动治掉了，
 分母变小，不是接缝变差。）

单独一个执行者而不是塞进 `PostureHold`：那个的语义是"驻留期间独占，走路必须让路"，
而一次性动作占的时间短得多，硬套会让 `enter/leave/settle/loop` 四个概念里有两个变成死的。
"""
from __future__ import annotations

from . import config


class OneShot:
    """一个一次性动作的播放器。宿主（PetWindow / 测试替身）持有一个实例。"""

    def __init__(self, pet, anim: str):
        self.pet = pet
        self.anim = anim
        self.phase = "off"          # off / turning / playing
        self.frame = 0
        self.total = 0
        self.step = 1
        self.done = None

    @property
    def busy(self) -> bool:
        return self.phase != "off"

    def start(self, done=None) -> bool:
        pet = self.pet
        # 互斥要问全：任何一条不满足就完全不让，否则两个执行者会抢同一个绘制目标。
        # 反过来，别的执行者起步时也必须问 `pet.oneshot.busy`（见 behavior/stage）。
        if (self.busy or pet.facing.turning or pet.walker.busy
                or pet.active_posture_busy() or pet.crawler.busy):
            return False
        anim = pet.manifest.anim(self.anim)
        if not anim:
            return False            # 素材没到位：让池子换一件事做，不要原地空转
        self.done = done
        self.total = int(anim["frames"])
        self.step = max(1, int(anim.get("stride", 2)))
        self.frame = 0
        self.phase = "turning"
        if pet.facing.value != config.FRONT_FACING:
            pet.facing.turn_to(config.FRONT_FACING, done=self._begin)
        else:
            self._begin()
        return True

    def _begin(self):
        if self.phase != "turning":
            return
        self.phase = "playing"
        self.pet.show_anim_frame(self.anim, self.frame)

    def tick(self) -> None:
        if self.phase == "turning":
            # 与 `Walker._guard_stolen_callback` 同一条兜底：用户中途单击换朝向会把挂在
            # 这一转上的 done 顶掉（那是 Facing 故意的语义），只等回调就永远卡在这。
            # 实测：随机长链里"单击循环→抱抱→单击循环"之后她再也不会有任何一次性动作。
            if not self.pet.facing.turning:
                self.pet.facing.turn_to(config.FRONT_FACING, done=self._begin)
            return
        if self.phase != "playing":
            return
        if self.pet.facing.turning or self.pet.facing.value != config.FRONT_FACING:
            self._finish()
            return
        self.frame += self.step
        if self.frame >= self.total - 1:
            self._finish()          # 末帧不单独停一拍：交还给待机，由静帧收尾
            return
        self.pet.show_anim_frame(self.anim, self.frame)

    def stop(self) -> None:
        """用户上手：当场结束，不演完。一次性动作没有"必须播完"的道理。"""
        if self.busy:
            self._finish()

    def _finish(self) -> None:
        self.phase = "off"
        done, self.done = self.done, None
        if done:
            done()

    def redraw(self) -> None:
        """改尺寸之后重画当前那一帧。**不许推进状态机。**"""
        if self.phase == "playing":
            self.pet.show_anim_frame(self.anim,
                                     max(0, min(self.frame, self.total - 1)))
