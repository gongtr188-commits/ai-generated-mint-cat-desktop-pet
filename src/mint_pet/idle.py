"""待机微动：正面静立时的呼吸 / 摇摆循环，速度带随机漂移。"""
from __future__ import annotations

import random

from . import config


class IdlePlayer:
    """只在"朝向=正面、不在转身、不在拖拽"时推进，其余状态一律让位。

    让位必须是彻底的：转身和 idle 抢同一个 QLabel 的话，表现就是画面抽搐。
    重新入场时回到第 0 帧 —— 那一帧最接近静态姿势，接得上也不会跳。

    倍速策略是**每圈抽一个新目标、逐 tick 限幅逼近**，而不是逐 tick 随机游走：
    游走看着平滑，实际一圈内就能从 0.5 抖到 1.0，观感就是忽快忽慢。
    限幅保证任意相邻两 tick 的倍速差都不超过 IDLE_SPEED_RATE，换速过程被摊到
    整圈里，听不出拐点。
    """

    def __init__(self, frames: int, stride: int, render, active,
                 speed_range=config.IDLE_SPEED, rate: float = config.IDLE_SPEED_RATE,
                 rng: random.Random | None = None):
        self.steps = list(range(0, max(0, frames), max(1, stride)))
        self.render = render
        self.active = active          # callable() -> bool，由宿主判定当前该不该动
        self.lo, self.hi = float(speed_range[0]), float(speed_range[1])
        self.rate = float(rate)
        self.rng = rng or random.Random()
        self.pos = 0.0
        self.speed = self.rng.uniform(self.lo, self.hi)
        self.target = self.speed
        self.shown = None

    def reset(self) -> None:
        """回到第 0 帧重播，但**保留当前倍速**。

        重新入场时重抽倍速会让"转身结束→待机"这一下突然变调；
        倍速只该在每圈结束时自然换目标。
        """
        self.pos = 0.0
        self.shown = None

    def _advance_speed(self) -> None:
        step = self.rate
        if self.speed < self.target:
            self.speed = min(self.target, self.speed + step)
        elif self.speed > self.target:
            self.speed = max(self.target, self.speed - step)

    def tick(self) -> None:
        if not self.steps or not self.active():
            if self.shown is not None:
                self.reset()
            return
        self._advance_speed()
        index = int(self.pos) % len(self.steps)
        self.render(self.steps[index])
        self.shown = self.steps[index]
        # 速度 <1 时 int(pos) 会连续两 tick 取到同一帧，这就是"慢放"本身，
        # 不需要额外的补帧——补帧反而会把素材里的呼吸节奏抹平。
        self.pos += self.speed
        if self.pos >= len(self.steps):        # 一圈跑完，给下一圈换个 tempo
            self.pos -= len(self.steps)
            self.target = self.rng.uniform(self.lo, self.hi)

    @property
    def running(self) -> bool:
        return self.shown is not None
