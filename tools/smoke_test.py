#!/usr/bin/env python3
"""冒烟测试：每次改动后必跑，约 75 秒。

断言"值"而不是"没崩"。用真实 PetWindow 实例 + 真事件循环驱动 tick。
    py -3 tools/smoke_test.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mint_pet.app import ensure_qt_plugins   # noqa: E402

# 必须在 QApplication 之前：本机 Python 装在中文路径下，Qt 自解析的插件目录会变乱码
ensure_qt_plugins()

from PyQt5.QtCore import QEventLoop, QTimer              # noqa: E402
from PyQt5.QtWidgets import (QApplication, QLabel,       # noqa: E402
                             QSlider, QWidgetAction)
from mint_pet import config                              # noqa: E402
from mint_pet.assets import AssetStore                   # noqa: E402
from mint_pet.behavior import (BehaviorPool, Crawler, Follower,  # noqa: E402
                               Walker)
from mint_pet.actions import OneShot                          # noqa: E402
from mint_pet.posture import (PostureHold, PostureWants,  # noqa: E402
                               settle_side_prone)
from mint_pet.facing import (Facing, InstantTurn, LinearTurn,  # noqa: E402
                             VideoTurn)
from mint_pet.manifest import Manifest                   # noqa: E402
from mint_pet.stage import PetWindow                     # noqa: E402

RESULTS: list = []
app = QApplication(sys.argv)
FACES = ["front", "side_left", "back", "side_right"]


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def wait(ms: int):
    """真等一段时间，让 80ms 帧循环跑起来。"""
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


def pump(ms: int):
    """跑一个真事件循环，让挂父对象的一次性定时器（防抖那种）落地。"""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


def wait_until(cond, timeout_ms: int = 9000) -> bool:
    """轮询等真实动作跑完。转圈时长取决于起手角度，写死 sleep 不是早了就是白等。"""
    end = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    app.processEvents()
    return False


def turn_unit_tests(m):
    """VideoTurn 的纯逻辑单测。facing.py 不依赖 Qt，所以能脱离 GUI 直接跑断言。

    转身是本项目已知最容易返工的模块（菲比为此改了 5 轮），把"首尾必达 / 时长一致 /
    起手不顿 / 中途就近改道"这些性质锁死，比事后靠肉眼盯帧号便宜得多。
    """
    anim = m.anim(config.SPIN_ANIM)
    cycle, keys, path = anim["cycle"], anim["keys"], anim["path"]
    check("转身几何已标定", cycle > 0 and len(keys) == 4 and len(path) == cycle + 1,
          f"cycle={cycle} keys={keys}")
    check("关键帧都落在有效周期内", all(0 <= k < cycle for k in keys), str(keys))
    check("周期已排除视频尾部的静止重复帧", anim["frames"] - cycle > 0,
          f"{anim['frames']} 帧截到 {cycle}")

    def build():
        shown = []
        d = VideoTurn(cycle, keys, path, shown.append)
        return d, shown, Facing(lambda v: None, count=4, director=d)

    probe = VideoTurn(cycle, keys, path, lambda i: None)

    def step(a, b):
        """两帧之间沿行进方向的视觉变化总量。复用 _step，别在测试里重写一遍跨环逻辑。"""
        fwd = (b - a) % cycle <= (a - b) % cycle
        n = (b - a) % cycle if fwd else (a - b) % cycle
        idxs = [(a + (s if fwd else -s)) % cycle for s in range(n + 1)]
        return sum(probe._step(x, y) for x, y in zip(idxs, idxs[1:]))

    n90, ratios = [], []
    for src in range(4):
        for dst in range(4):
            if src == dst:
                continue
            d, shown, f = build()
            f.value = src
            d.start(f, dst)
            seq = d.seq
            check(f"转身 {FACES[src]}→{FACES[dst]} 首帧=当前关键帧 末帧=目标关键帧",
                  seq[0] == keys[src] and seq[-1] == keys[dst], str(seq))
            if abs(src - dst) % 4 in (1, 3):
                n90.append(len(seq))
            s = [step(a, b) for a, b in zip(seq, seq[1:])]
            ratios.append(min(s) / (sum(s) / len(s)))
    check("90° 转身统一 9 帧", set(n90) == {config.TURN_FRAMES_90}, str(n90))
    check("最慢一步的视觉变化不低于平均的 40%（起手不顿）", min(ratios) > 0.4,
          f"worst={min(ratios):.2f}")

    # 中途改道：从"正在显示的那一帧"出发，不是从下一帧、也不是从标准姿势出发
    d, shown, f = build()
    f.value = 0
    d.start(f, 2)
    d.tick(f)
    d.tick(f)
    mid = d.shown
    d.start(f, 1)
    check("转身中途改道从当前帧就近出发", d.seq[0] == mid, f"mid={mid} seq={d.seq}")
    check("改道后仍然必达目标关键帧", d.seq[-1] == keys[1], str(d.seq))

    # 接近 180° 时选边必须可预期，随机选边看起来"方向飘忽"
    seqs = set()
    for _ in range(6):
        d, shown, f = build()
        f.value = 0
        d.start(f, 2)
        seqs.add(tuple(d.seq))
    check("接近 180° 的选边可预期（6 次一致）", len(seqs) == 1, str(list(seqs)[0]))

    # ── 绕正面 vs 绕背面 ────────────────────────────────────────────────
    # 菜单让她的朝向在 左面/右面 之间切换时要路过正面（观众要看的是脸），不绕背面。
    # 但"永远走正面"是错的：背面→侧面 走正面要 74 帧、直达只 24 帧，那才是真绕远。
    # 所以规则是"两条路长度接近时才让给正面"，门限 TURN_FRONT_RATIO=1.6。
    def near(seq, key, win=8):
        return any(min(abs(x - keys[key]), cycle - abs(x - keys[key])) <= win for x in seq)

    def route(src, dst):
        d, shown, f = build()
        f.value = src
        d.start(f, dst)
        return d.seq

    def max_step(seq):
        # 环上最短弦长：反向步进时 (b-a)%cycle 得到的是补数（47→42 会算成 93），
        # 必须再取 min(d, cycle-d) 才是真实步长
        return max(min(d, cycle - d) for d in
                   ((seq[i + 1] - seq[i]) % cycle for i in range(len(seq) - 1)))

    for a, b, tag in ((1, 3, "左面→右面"), (3, 1, "右面→左面")):
        seq = route(a, b)
        check(f"{tag} 路过正面而不是绕背面", near(seq, 0) and not near(seq, 2), str(seq))
    for a, b, tag in ((2, 1, "背面→左面"), (2, 3, "背面→右面"),
                      (1, 2, "左面→背面"), (3, 2, "右面→背面"),
                      (0, 1, "正面→左面"), (0, 3, "正面→右面")):
        check(f"{tag} 保持直达，不为绕正面而绕远", max_step(route(a, b)) <= cycle // 3,
              f"最大单步={max_step(route(a, b))}")

    # 转圈
    d, shown, f = build()
    f.value = 1
    d.spin(f)
    check("转圈末帧落在目标关键帧", d.seq[-1] == keys[1], str(d.seq[-3:]))
    # cycle 是奇数时 range(0, cycle, stride) 会多覆盖一格，所以按向上取整算
    expect = -(-cycle // config.SPIN_STRIDE) + 1
    check("转圈步数 = ceil(cycle/stride) + 收尾", len(d.seq) == expect,
          f"{len(d.seq)} vs {expect}")
    d, shown, f = build()
    f.value = 0
    d.start(f, 2)
    d.tick(f)
    d.spin(f)
    check("转身在飞时转圈，目标沿用那一转的朝向", d.target == 2, str(d.target))

    # 连点：朝向循环必须单向推进。曾经每跳都重新取最短路径，连点四次会"转过去又
    # 转回来"，用户观感就是卡。CLICK_CYCLE 本身就是视频正向，所以强制同向。
    d, shown, f = build()
    f.value = 0
    trace = [d.keys[0]]
    for _ in range(8):
        f.cycle()
        for _ in range(2):
            d.tick(f)
            if d.shown is not None:
                trace.append(d.shown)
    deltas = [(b - a) % cycle for a, b in zip(trace, trace[1:])]
    check("连点 8 次全程单向推进（无倒退）",
          all(0 <= s <= cycle // 2 for s in deltas),
          f"步进范围 {min(deltas)}~{max(deltas)}，半圈={cycle // 2}")
    check("连点 8 次（两轮循环）回到起点朝向", f.value == 0, f"value={f.value}")

    # 单击循环的序列要按**当前姿态有几个朝向**来截。
    # CLICK_CYCLE 写死了站姿的 0..3，趴着只有 3 个朝向时索引 3 会被取模成 0，
    # 而她本来就在 0 —— 点一下没反应，用户观感就是"卡住了/点不动"。
    # 坐着只有一个姿势，更是应该直接没动作而不是"转一圈空气"。
    for count, tag in ((3, "趴着三朝向"), (1, "坐着一个朝向")):
        g = Facing(lambda v: None, count=count, director=InstantTurn())
        seen, moved = [], 0
        for _ in range(6):
            if g.cycle():
                moved += 1
                g.director.tick(g)
            seen.append(g.value)
        check(f"{tag}：单击循环不死点、不越界",
              all(0 <= v < count for v in seen) and (moved > 0) == (count > 1),
              f"轮转 {seen}，有效点击 {moved}/6")

    # 菜单直选仍走最短路径：用户改主意时不该绕远
    d, shown, f = build()
    f.value = 0
    d.start(f, 1)                      # front(1)→side_left(71)：正向 70 帧、反向 28 帧
    steps = [(a - b) % cycle for a, b in zip(d.seq, d.seq[1:])]
    check("菜单直选（hint=0）走最短路径",
          all(0 <= s <= cycle // 2 for s in steps)
          and sum(steps) == (keys[0] - keys[1]) % cycle,
          f"seq={d.seq} 实走={sum(steps)} 短路径={(keys[0] - keys[1]) % cycle}")

    # 收尾必须落到标准姿势、触发回调、清空状态
    d, shown, f = build()
    f.value = 0
    landed = []
    d.start(f, 3, done=lambda: landed.append(f.value))
    for _ in range(config.TURN_FRAMES_90 + 3):
        d.tick(f)
    check("转身结束落到目标朝向并触发回调", f.value == 3 and landed == [3] and not d.busy,
          f"value={f.value} landed={landed}")

    # 素材没到位时必须能退回瞬时切换
    inst = InstantTurn()
    f = Facing(lambda v: None, count=4, director=inst)
    f.turn_to(2)
    f.tick()
    check("无转身素材时退回 InstantTurn 仍能换朝向", f.value == 2 and not inst.busy)

    # ── 线性 180° 弧（趴着转180度）──────────────────────────────────────
    # 直接用清单里真实标定出来的数据，不要手搓：path 要按帧号索引到 cycle，
    # 造一条短的会直接 IndexError，而且测的也不是真素材。
    pt = m.anim("prone_turn") or {}
    check("趴着转身图已标定", bool(pt.get("keys")) and bool(pt.get("path"))
          and pt.get("linear") is True,
          f"keys={pt.get('keys')} cycle={pt.get('cycle')} linear={pt.get('linear')}")
    if pt.get("keys"):
        lk, lc = pt["keys"], pt["cycle"]
        lin = LinearTurn(lc, lk, pt["path"], lambda i: None)
        lf = Facing(lambda v: None, count=len(lk), director=lin)
        lin.start(lf, len(lk) - 1)                         # 正面趴 → 右趴：沿弧向前
        fwd_seq = list(lin.seq)
        while lin.seq:
            lin.tick(lf)
        check("线性：往弧的前方走，帧号单调递增",
              len(fwd_seq) > 2 and all(b > a for a, b in zip(fwd_seq, fwd_seq[1:])),
              f"{len(fwd_seq)} 步，{fwd_seq[0]}→{fwd_seq[-1]}")
        check("线性：到达目标关键帧", lf.value == len(lk) - 1 and lin.shown == lk[-1],
              f"value={lf.value} shown={lin.shown}")

        lin.start(lf, 0)                                   # 右趴 → 左趴：必须往回退
        back_seq = list(lin.seq)
        check("线性：从最右回最左是**往回退**，不是绕一整圈",
              len(back_seq) > 2 and all(b < a for a, b in zip(back_seq, back_seq[1:])),
              f"{len(back_seq)} 步，{back_seq[0]}→{back_seq[-1]}")
        # 环形调度器对同一条弧会怎么算？取模之后"回退 86 帧"变成"前进 11 帧"，
        # 也就是她会在原地往反方向多转一大截 —— 这就是必须分开建模的原因。
        ring = VideoTurn(lc, lk, pt["path"], lambda i: None)
        d_fwd, d_back = ring._distances(94, 8)
        check("对照：环形取模会把 86 帧的回退算成 11 帧的前进",
              (d_fwd, d_back) == (11, 86), f"fwd={d_fwd} back={d_back}")
        check("线性弧不提供转一圈", lin.spin(lf) is False)


def idle_unit_tests():
    """待机随机倍速的纯逻辑测试。注入固定种子的 RNG，否则断言本身就是随机的。"""
    import random
    from mint_pet.idle import IdlePlayer

    lo, hi = config.IDLE_SPEED
    out, sp = [], []
    p = IdlePlayer(121, 2, out.append, lambda: True, rng=random.Random(7))
    # 窗口要够长：限幅逼近意味着一圈最多走 rate×步数，太短的窗口里倍速根本来不及
    # 到达区间两端，会把"探索满区间"误判成失败。3000 tick ≈ 49 圈。
    for _ in range(3000):
        p.tick()
        sp.append(p.speed)
    check(f"倍速始终锁在配置区间 {config.IDLE_SPEED}", lo <= min(sp) and max(sp) <= hi,
          f"实测 {min(sp):.3f}~{max(sp):.3f}")
    check("倍速真的在变（不是固定值）", len({round(s, 3) for s in sp}) > 50,
          f"{len({round(s, 3) for s in sp})} 种取值")
    check("倍速会走满区间两端，而不是贴在中值不动",
          min(sp) < lo + 0.06 and max(sp) > hi - 0.06,
          f"min={min(sp):.2f} max={max(sp):.2f}")

    # 用户明确要求"相邻两个不要差太多"：这就是那条硬上限
    deltas = [abs(b - a) for a, b in zip(sp, sp[1:])]
    check("相邻 tick 倍速差不超过 IDLE_SPEED_RATE（换速必须平滑）",
          max(deltas) <= config.IDLE_SPEED_RATE + 1e-9,
          f"max Δ={max(deltas):.4f} ≤ {config.IDLE_SPEED_RATE}，均值 Δ={sum(deltas)/len(deltas):.4f}")
    # 走完 0.5→1.0 全程要摊到接近一整圈，否则就是"换挡"而不是"呼吸有快慢"
    ticks = (hi - lo) / config.IDLE_SPEED_RATE
    check("最远换速时长接近一整圈（不是一步跳过去）",
          0.5 * len(p.steps) <= ticks <= 1.5 * len(p.steps),
          f"{ticks:.0f} tick = {ticks * config.FPS_MS / 1000:.1f}s，一圈 {len(p.steps)} tick")

    # out 存的是 steps[index]（= index*2），且 600 tick 会绕回起点若干次，
    # 所以要在"步索引"空间上按环判：每 tick 最多前进 1 步，绝不倒退、绝不跳步。
    n = len(p.steps)
    idx = [x // 2 for x in out]
    deltas = sorted({(b - a) % n for a, b in zip(idx, idx[1:])})
    check("每 tick 最多前进 1 步（慢放不跳步、不倒退）", deltas == [0, 1], f"出现的步进={deltas}")
    holds = sum(1 for a, b in zip(out, out[1:]) if a == b)
    check("慢放时确实会重复上一帧", holds > 0, f"重复 {holds}/{len(out)} tick")

    sp_before = p.speed
    p.reset()
    check("reset 回到第 0 帧但保留当前倍速（重新入场不突变）",
          p.pos == 0.0 and p.shown is None and p.speed == sp_before,
          f"pos={p.pos} speed={p.speed:.3f} 入场前={sp_before:.3f}")

    # 每圈换一个 tempo：这才是"灵活"的来源，同时换速本身是平滑逼近的
    p2 = IdlePlayer(121, 2, lambda i: None, lambda: True, rng=random.Random(11))
    tempos = []
    for _ in range(4000):
        before_t = p2.target
        p2.tick()
        if p2.target != before_t:
            tempos.append(round(before_t, 3))
    check("每圈换新目标倍速，且各圈 tempo 确实不同", len(set(tempos)) >= 3,
          f"{len(tempos)} 圈 / {len(set(tempos))} 种 tempo，样本 {sorted(set(tempos))[:6]}")

    # 换种子必须给出不同节奏，否则"随机"只是摆设
    seqs = set()
    for seed in (1, 2, 3):
        q = IdlePlayer(121, 2, lambda i: None, lambda: True, rng=random.Random(seed))
        s = []
        for _ in range(120):
            q.tick()
            s.append(round(q.speed, 3))
        seqs.add(tuple(s))
    check("不同种子给出不同节奏", len(seqs) == 3, f"{len(seqs)} 种")

    # 让位时不许推进
    moved = []
    r = IdlePlayer(121, 2, moved.append, lambda: False, rng=random.Random(1))
    for _ in range(10):
        r.tick()
    check("不允许播放时待机完全不动", moved == [] and not r.running, f"{len(moved)} 次渲染")


class _StubFacing:
    """假朝向：转身固定 `turn_ticks` tick 完成，完成时才落地并回调 done。

    默认 3 是**为了快**：状态机测试关心"回调有没有兑现"，不关心转多久。
    但要拿它量"她一天在忙什么"（`tools/behavior_check.py`）就必须传真实的 9
    （`TURN_FRAMES_90`），否则"在忙"的时间被系统性低估，整份报表都是偏的。
    """
    count = 4

    def __init__(self, turn_ticks: int = 3):
        self._turn_ticks = turn_ticks
        self.value = 0
        self.turning = False
        self.spins = 0
        self._left = 0
        self._q = 0
        self._done = None
        self._done_apply = None
        # 行为池靠 director 上有没有 spin 判断"转圈可用"，stub 也得带上这一面
        self.director = type("D", (), {"spin": lambda s, done=None: True})()

    def turn_to(self, q, done=None, hint=0):
        q %= self.count
        if q == self.value:
            if done:
                done()
            return False
        self._q, self._done, self.turning, self._left = q, done, True, self._turn_ticks
        return True

    def cycle(self, order=None):
        """真实 Facing 有这个入口，用户在行走途中单击就是走它 —— 矩阵要能模拟才得有。"""
        order = order or config.CLICK_CYCLE
        self._pos_i = getattr(self, "_pos_i", 0) + 1
        return self.turn_to(order[self._pos_i % len(order)])

    def apply(self, value):
        self.value = value % self.count
        if self._done_apply:
            self._done_apply(self.value)

    def set_posture(self, faces, director=None):
        self.count = max(1, len(faces))
        self.value = 0
        if self._done_apply:
            self._done_apply(0)

    def spin(self, done=None):
        self.spins += 1
        if done:
            done()
        return True

    def tick(self):
        if not self.turning:
            return
        self._left -= 1
        if self._left > 0:
            return
        self.turning = False
        self.value = self._q
        d, self._done = self._done, None
        if d:
            d()


class _StubAnim:
    """按真实清单的形状给：站姿用环形 360°，趴姿用线性 180° + 四足步态。"""
    WALK = {("standing", "left"): "walk_left", ("standing", "right"): "walk_right",
            ("prone", "left"): "walk4_left", ("prone", "right"): "walk4_right"}
    RUN = {("standing", "left"): "run_left", ("standing", "right"): "run_right"}
    FACING = {"standing": {"left": 1, "right": 3}, "prone": {"left": 1, "right": 2}}
    # 驻留微动：单个名字 = 所有朝向共用；列表 = **按朝向对齐**，None 表示该朝向没有微动。
    # 和真实清单同形（趴着只有正面那段会动；趴睡左右各一段）。
    IDLE = {"sitting": "sitdown", "prone": ["prone_idle", None, None],
            "dozing": "doze", "sleeping": ["sleep_left", "sleep_right"]}
    TURN = {"standing": "turn", "prone": "prone_turn"}

    def posture_idle(self, key):
        return self.IDLE.get(key)

    def posture_turn(self, key):
        return self.TURN.get(key)

    def anim(self, name):
        d = dict(name=name, frames=145 if name == "sitdown" else 121,
                 stride=2, dir=".", mode="sequence",
                 settle_at=77 if name == "sitdown" else 0)
        if name == "sitdown":
            # 与真实清单同形：进入段从"最像站立图"的 f16 起播，交接点 = 待机窗口起点 f77
            d["loop"] = [77, 95]
            d["enter_at"] = 16
            d["hold_move"] = 0.344
            d["enter_move"] = 1.205
        if name == "walk4_right":
            # 同形：素材 97 帧但周期 22，截到 4 个整周期才接得上相；起步相位在第 3 帧
            d["gait"] = dict(period=22, seam=11.13, loop=[0, 88], phase0=3)
        if name in ("prone_idle", "doze", "sleep_left", "sleep_right"):
            # 与真实清单同形：整段 97 帧都是驻留微动，来回一轮 3.44s
            d.update(frames=97, loop=[0, 44], loop_stride=2)
        if name == "hug":
            # 一次性动作：首尾接缝是相邻帧差的 4 倍，**故意没有 loop** —— 只能正放一遍
            d.update(frames=76)
        return d

    def gait_anim(self, posture, side, gait="walk"):
        table = self.RUN if gait == "run" else self.WALK
        key = (posture, side)
        return self.anim(table[key]) if key in table else None

    def has_gait(self, posture, gait):
        table = self.RUN if gait == "run" else self.WALK
        return all((posture, s) in table for s in ("left", "right"))

    def walk_facing(self, posture, side):
        return self.FACING[posture][side]


class FakePet:
    """只喂行为状态机需要的最小接口，不开 Qt —— 状态机本身不依赖 Qt，就该能这么测。"""

    def __init__(self, x=600, y=300, screen=(0, 1400), vscreen=(0, 900), w=150,
                 turn_ticks=3):
        self._x, self._y = x, y
        self._left, self._right = screen
        self._top, self._bottom = vscreen
        self._w = w
        self.pet_h = config.DEFAULT_H
        self._dragging = False
        self._menu_open = False
        self.facing = _StubFacing(turn_ticks)
        self.manifest = _StubAnim()
        self.frames = []
        self.walker = Walker(self)
        self.crawler = Crawler(self)
        self.follower = Follower(self)
        self.cursor_xy = (700, 400)
        self.posture = config.BASE_POSTURE
        self.side_prone = config.SIDE_PRONE
        # 用真的 PostureHold，而不是替身：姿态的进入/驻留/倒放退出是这次新增的
        # 主要逻辑，拿 stub 顶上就等于没测。
        self.holders = {
            "sitting": PostureHold(self, "sitting", config.POSTURE_HOLD_S["sitting"],
                                   enter_anim="sitdown", loop_anim="sitdown"),
            # 趴坐的驻留微动只有正面那一段（侧趴两张是静帧），所以 loop_anim
            # 给的是**按朝向对齐的列表**，由 PostureHold._loop_anim 现取。
            "prone": PostureHold(self, "prone", config.POSTURE_HOLD_S["prone"],
                                 loop_anim=["prone_idle", None, None]),
            "dozing": PostureHold(self, "dozing", config.POSTURE_HOLD_S["dozing"],
                                  loop_anim="doze"),
            "sleeping": PostureHold(self, "sleeping", config.POSTURE_HOLD_S["sleeping"],
                                    loop_anim=["sleep_left", "sleep_right"]),
        }
        self.behaviors = BehaviorPool(self)
        self.wants = PostureWants(self)
        # 一次性动作（拥抱）也用真的播放器，理由和 holder 一样：
        # "起步前先看有没有别的占着"这条互斥就是被测对象。
        self.oneshot = OneShot(self, config.HUG_ANIM)
        # 默认关掉调度器：否则"走完回到正面"这类单动作测试跑到一半，
        # 池子会插进来一段漫游，把位置改到别处 —— 测的就不是被测对象了。
        self.behaviors.enabled = False

    def request_posture(self, key: str, manual: bool = True) -> bool:
        return self.wants.request(key, manual)

    def set_follow(self, on: bool) -> None:
        """与 PetWindow 同一份收尾：开跟随先把她从手里那件事里放出来。"""
        on = bool(on)
        self.follower.set_enabled(on)
        if on:
            self.wants.clear()
            self.walker.stop()
            self.crawler.stop()
            self.oneshot.stop()
            for h in self.holders.values():
                h.stop()

    def cursor_pos(self):
        return self.cursor_xy

    def set_side_prone(self, on: bool) -> None:
        """与 PetWindow 同一份收尾：开关只管左右趴着，正面趴下不受影响。"""
        on = bool(on)
        if on == self.side_prone:
            return
        self.side_prone = on
        if not on:
            if self.wants.key == "sleeping":
                self.wants.clear()
            settle_side_prone(self)

    def x(self):
        return self._x

    def y(self):
        return self._y

    def width(self):
        return self._w

    def height(self):
        return self._w

    def move(self, x, y):
        self._x, self._y = x, y

    def screen_left(self):
        return self._left

    def screen_right(self):
        return self._right

    def screen_top(self):
        return self._top

    def screen_bottom(self):
        return self._bottom

    def show_anim_frame(self, name, i):
        self.frames.append((name, i))

    def active_posture_busy(self):
        return any(h.busy for h in self.holders.values())

    def show_posture(self, key, facing=0):
        self.frames.append((f"pose:{key}", facing))

    FACES = {config.BASE_POSTURE: ["front", "side_left", "back", "side_right"],
             "sitting": ["sit_front"],
             "prone": ["prone_front", "prone_left", "prone_right"],
             "dozing": ["doze_front"],
             "sleeping": ["sleep_left", "sleep_right"]}

    def posture_faces(self, posture=None):
        """与 PetWindow.posture_faces 同一条规则：关掉左右趴着时趴姿只剩正面。"""
        posture = posture or self.posture
        faces = self.FACES[posture]
        if posture == "prone" and not self.side_prone:
            return faces[:1]
        return faces

    def enter_posture(self, posture):
        self.posture = posture
        # 走真实那条路径：朝向数与姿态绑定，不能只改一个字段。
        # 名单取自真实清单的形状（站 4 / 坐 1 / 趴坐 3 / 打瞌睡 1 / 趴睡 2）。
        self.facing.set_posture(self.posture_faces(posture))

    def tick(self):
        self.facing.tick()
        self.follower.tick()
        self.walker.tick()
        for h in self.holders.values():
            h.tick()
        self.oneshot.tick()         # 与 PetWindow 同位置：holder 之后、意图队列之前
        self.wants.drain()          # 与 PetWindow.tick 同一位置：holder 之后、行为池之前
        self.behaviors.tick()


def behavior_unit_tests():
    """随机行为池的纯逻辑测试。全部注入固定种子 RNG，否则断言自己就是随机的。"""
    import random

    p = FakePet()
    lens = {round(p.walker.rng.uniform(*config.WALK_S), 2) for _ in range(200)}
    check("步程时长是随机的而不是一个常数", len(lens) > 50, f"{len(lens)} 种取值")
    lo, hi = config.WALK_S
    check("步程时长锁在配置区间", lo <= min(lens) and max(lens) <= hi,
          f"{min(lens):.2f}~{max(lens):.2f}")

    # 贴边时随机到墙那边 = 原地踩步，观感就是卡住
    q = FakePet(x=1360, w=150)
    dirs = {q.walker._pick_dir() for _ in range(200)}
    check("贴右墙时不会随机到向右走", dirs == {-1}, f"挑到 {dirs}")
    r = FakePet(x=0, w=150)
    dirs = {r.walker._pick_dir() for _ in range(200)}
    check("贴左墙时不会随机到向左走", dirs == {1}, f"挑到 {dirs}")
    both = {FakePet(x=700).walker._pick_dir() for _ in range(200)}
    check("屏幕中间两侧都可能挑到", both == {-1, 1}, f"挑到 {both}")

    # 走完一段之后面向哪：必须**一律回正面**。以前"向左走/向右走"回落到出发朝向，
    # 而"随便走走"回正面 —— 用户勾了左面再点向右走，她走完自己又转回左面，
    # 和同菜单里另一个走路的收尾对不上。这条断言就是要那个场景写的。
    s = FakePet(x=700)
    s.facing.value = 1                      # 先"勾左面"
    check("随机步程也能发起走路", s.walker.start(1))   # 再"向右走"
    for _ in range(400):
        s.tick()
    check("走完一律回到正面（不是出发朝向），且收得了工",
          s.facing.value == config.WALK_HOME_FACING and not s.walker.busy and s._x != 700,
          f"facing={s.facing.value} 出发=1 x={s._x}")
    t = FakePet(x=700)
    t.facing.value = 1
    t.walker.roam(gait="walk")
    for _ in range(600):
        t.tick()
    check("定向走与漫游的收尾朝向是同一条规则",
          t.facing.value == s.facing.value == config.WALK_HOME_FACING,
          f"roam={t.facing.value} start={s.facing.value}")

    # ── 跟随鼠标：一小步一小步地走，不是逐帧贴住光标 ─────────────────────
    dead = config.FOLLOW_DEADZONE_PX      # FakePet 的显示高就是 DEFAULT_H，不必再缩放

    f1 = FakePet(x=600, y=300)
    f1.cursor_xy = (200, 300)                  # 光标在左边远处
    f1.follower.set_enabled(True)
    walked = False
    for _ in range(900):
        f1.tick()
        walked = walked or f1.walker.busy
    check("开了跟随她会朝光标走", walked and f1._x < 600, f"x={f1._x}")
    check("走到光标附近就停（死区内不再迈步）",
          not f1.walker.busy and abs(f1._x + f1._w / 2 - 200) <= dead + 6,
          f"中心={f1._x + f1._w / 2:.0f} 死区={dead:.0f}px")
    check("跟随停步时保持行进方向（每步都转回正面就是原地拧陀螺）",
          f1.facing.value == f1.manifest.walk_facing("standing", "left"),
          f"facing={f1.facing.value}")

    f2 = FakePet(x=600, y=300)
    f2.cursor_xy = (600 + f2._w // 2, 300)     # 光标就在她身上
    f2.follower.set_enabled(True)
    for _ in range(60):
        f2.tick()
    check("光标在死区内时她一步都不走",
          not f2.walker.busy and f2._x == 600, f"x={f2._x}")

    f3 = FakePet(x=600, y=300)
    f3.cursor_xy = (200, 300)
    f3.behaviors.enabled = True
    f3.follower.set_enabled(True)
    for _ in range(900):
        f3.tick()
    check("跟随期间随机池完全让位（刚跟到旁边又自己溜走）",
          f3.behaviors.fired == 0, f"自发动作 {f3.behaviors.fired} 次")

    for why, patch in (("拖拽中", lambda p: setattr(p, "_dragging", True)),
                       ("菜单开着", lambda p: setattr(p, "_menu_open", True)),
                       ("姿态驻留中", lambda p: p.holders["sitting"].start())):
        g = FakePet(x=600, y=300)
        g.cursor_xy = (200, 300)
        g.follower.set_enabled(True)
        patch(g)
        for _ in range(60):
            g.tick()
        check(f"{why}时跟随不抢她的操作", not g.walker.busy and g._x == 600, f"x={g._x}")

    f4 = FakePet(x=600, y=300)
    f4.cursor_xy = (200, 300)
    for _ in range(60):
        f4.tick()
    check("没开跟随时光标再远她也不动", not f4.walker.busy and f4._x == 600, f"x={f4._x}")

    # 跟随的起步那一转还挂着 `_walk`：把它撤掉之前收工，她就会朝出发时算的旧目标迈步。
    # （keep_facing 的收尾原本直接 `_end`，正是漏了这一条 —— 矩阵测出"回到右下角之后
    #   她又自己走开 6px"。）
    f7 = FakePet(x=600, y=300)
    f7.cursor_xy = (200, 300)
    f7.follower.set_enabled(True)
    f7.tick()
    check("跟随第一步先转身（还没迈步）", f7.walker.phase == "turning" and f7.facing.turning,
          f"phase={f7.walker.phase}")
    f7.follower.set_enabled(False)            # 用户在这时候关掉跟随 / 把她挪走
    f7.walker.stop()
    x0 = f7._x
    for _ in range(60):
        f7.tick()
    check("收掉在飞的跟随之转之后，她不会朝旧目标迈一步",
          f7._x == x0 and not f7.walker.busy, f"x={f7._x} 收工前={x0}")

    f5 = FakePet(x=600, y=300)
    f5.cursor_xy = (600 + f5._w // 2, 300 + 300)   # 正上方：横向没有位移
    f5.follower.set_enabled(True)
    for _ in range(60):
        f5.tick()
    check("光标只在纵向远处时不硬走（没有上下走的步态），也不卡死",
          not f5.walker.busy and f5._x == 600, f"x={f5._x}")

    # ── 跟随必须是**连续**的：一程走到底、途中改道，而不是走几步停一下 ──────
    # 旧实现是"发起一趟 → 走完 → 再看一眼"，一程 3~5 秒，观感就是慢半拍 + 一顿一顿；
    # 而且起步那一转被自己的"让路"判据当成忙，刚起步就叫停、3 tick 后再起步。
    g1 = FakePet(x=600, y=300)
    g1.cursor_xy = (1250, 300)
    g1.follower.set_enabled(True)
    starts, prev_busy, gaits = 0, False, set()
    for _ in range(150):
        g1.tick()
        if g1.walker.busy and not prev_busy:
            starts += 1
        prev_busy = g1.walker.busy
        if g1.walker.phase == "walking":
            gaits.add(g1.walker.gait)
    check("长距离跟随只发起一趟行程（旧实现每 3~5 秒重新起步一次）",
          starts == 1, f"起步 {starts} 次，走了 {g1._x - 600}px")
    check("离得远自动换成跑", "run" in gaits, f"路上用过 {sorted(gaits)}")

    g2 = FakePet(x=600, y=300)
    g2.cursor_xy = (1250, 300)
    g2.follower.set_enabled(True)
    for _ in range(25):
        g2.tick()
    far_x = g2._x
    g2.cursor_xy = (700, 300)                    # 光标突然回到她原来待的地方
    for _ in range(200):
        g2.tick()
    check("光标反向移动时她当场改道，而不是先走完出发时算的那个点",
          g2._x < far_x and not g2.walker.busy and abs(g2._x + g2._w / 2 - 700) <= dead + 6,
          f"最远到 {far_x}，现停 {g2._x}（中心 {g2._x + g2._w / 2:.0f}，目标 700）")

    # 撞墙判定只准查**前导边**。查两条边的话，贴边站着反向起步会被身后那条边误判。
    # 用差分测：同样参数，贴左墙向右走必须和在屏幕中间向右走走得一样远。
    def walk_right_dist(x0):
        p = FakePet(x=x0, y=300)
        p.facing.value = 1                     # 已经朝左，起步不再需要转身
        p.walker.start(1, seconds=3.0)
        for _ in range(300):
            p.tick()
        return p._x - x0, p.walker.busy

    far, busy_far = walk_right_dist(700)
    edge, busy_edge = walk_right_dist(0)
    check("贴左墙向右走不会被身后那条边提前判停",
          edge == far and not busy_edge and not busy_far and far > 50,
          f"贴墙 {edge}px / 屏中 {far}px，busy={busy_edge}/{busy_far}")

    # ── 全屏漫游：随机取点走过去 ────────────────────────────────────────
    rm = FakePet(x=700, y=300)
    rm.behaviors.rng = random.Random(4)
    x0, y0 = rm._x, rm._y
    check("可以发起漫游", rm.walker.roam())
    ys, faces = [], set()
    pts = [(rm._x, rm._y)]
    for _ in range(2000):
        rm.tick()
        ys.append(rm._y)
        faces.add(rm.facing.value)
        pts.append((rm._x, rm._y))
    # 单 tick 位移上限：横向一步 3px，纵向一步 2px + 起伏 3px。
    # 这条不变式专门抓"零长度段被算成反向 dirn 后直接硬跳到 target"的凭空平移。
    jump = max(max(abs(b[0] - a[0]), abs(b[1] - a[1])) for a, b in zip(pts, pts[1:]))
    check("任何一 tick 都不会出现超过一步的凭空平移", jump <= 6, f"最大单 tick 位移 {jump}px")

    # 单趟必须有限：不限时实测平均 9~25s，配 30s 冷却会让她近一半时间在走路
    trips, dests = [], []
    for seed in range(12):
        q = FakePet(x=700, y=450)
        q.walker.rng = random.Random(seed)
        if q.walker.roam():
            n = 0
            while q.walker.busy and n < 5000:
                q.tick()
                n += 1
            trips.append(n)
            dests.append((abs(q._x - 700), abs(q._y - 450)))
    secs = [t * config.FPS_MS / 1000 for t in trips]
    check("漫游单趟时长受 ROAM_MAX_S 约束（留出两次转身的余量）",
          max(secs) <= config.ROAM_MAX_S + 3.0,
          f"{len(secs)} 趟：均 {sum(secs)/len(secs):.1f}s，最长 {max(secs):.1f}s，"
          f"上限 {config.ROAM_MAX_S:.0f}s + 转身")
    check("漫游不会挑净位移太小的点（那不值两次转身）",
          all(d[0] >= config.ROAM_MIN_DX and d[1] >= config.ROAM_MIN_PX for d in dests),
          f"落点位移 {dests}")
    duty = sum(secs) / (sum(secs) + len(secs) * (config.BEHAVIOR_COOLDOWN_S["roam"] + max(secs)))
    check("占空比：她大部分时间应该闲着而不是在走", duty < 0.3,
          f"约 {duty*100:.0f}% 的时间在走动")
    check("漫游真的挪了位置", (rm._x, rm._y) != (x0, y0), f"({x0},{y0}) -> ({rm._x},{rm._y})")
    check("漫游结束不再忙", not rm.walker.busy, f"phase={rm.walker.phase}")
    check("到达后转身面向镜头", rm.facing.value == config.IDLE_FACING,
          f"facing={rm.facing.value}")
    check("落点仍在屏幕可用区内",
          rm._left <= rm._x <= rm._right - rm.height()
          and rm._top <= rm._y <= rm._bottom - rm.height(),
          f"({rm._x},{rm._y}) 屏幕 [{rm._left},{rm._right})x[{rm._top},{rm._bottom})")

    # ── 斜着走：素材只有左右侧视步态，纵向靠"横向迈步时按比例带一点纵向位移" ──
    dg = FakePet(x=400, y=450)
    check("可以斜着走到右上方", dg.walker.start_to(900, 300))
    path, faces_walking = [], set()
    for _ in range(2000):
        before_turn = dg.facing.turning
        stepping = dg.walker.phase == "walking"
        dg.tick()
        # 只记"确实在迈步"的那些 tick。之前是把整段（含转身去/转身回）都记下来再掐头去尾
        # 各 8 个 tick —— 那个 8 是按"走路 3px/帧、x 先停"手工凑出来的余量。
        # 现在远途会自动改跑步态（5px/帧），x 提前约 30 tick 落地，
        # 固定 8 tick 就切不掉 y 的尾巴，断言被自己的采样方式误判。
        if stepping:
            path.append((dg._x, dg._y))
        if not before_turn and not dg.facing.turning and dg.walker.phase == "walking":
            faces_walking.add(dg.facing.value)
    check("斜走全程只用侧视步态朝向", faces_walking <= {1, 3}, f"走动时朝向 {sorted(faces_walking)}")
    # 两轴同时推进。纵向步长是小数（这一趟 150px ÷ 99 tick ≈ -1.5/帧），
    # 取整后会有"这一 tick 原地"甚至回半格，所以判"净推进 + 不倒退"，
    # 不能判逐 tick 严格单调。
    # 单 tick 纵向最多走多少，要按**这一步实际算出来的步长**给余量：
    # 写死 -1 等于把"走路 3px/帧"这件事二次硬编码进断言里，
    # 远途自动改跑步态（5px/帧 → 纵向 1.5/帧）之后就会假失败。
    vstep = abs(dg.walker.step_y)
    ys = [p[1] for p in path]
    xs = [p[0] for p in path]
    check("斜走时两轴同时在推进",
          xs[-1] > xs[0] and ys[-1] < ys[0]
          and min(b - a for a, b in zip(ys, ys[1:])) >= -(int(vstep) + 2)
          and max(b - a for a, b in zip(ys, ys[1:])) <= 1
          and min(b - a for a, b in zip(xs, xs[1:])) >= 0,
          f"x {xs[0]}->{xs[-1]}，y {ys[0]}->{ys[-1]}，纵向步长 {vstep:.2f}/帧")
    check("斜走两轴同时到达，落点精确", (dg._x, dg._y) == (900, 300),
          f"落点 ({dg._x},{dg._y})")

    # 斜率上限：目标在正上方（横向几乎没位移）时，纵向必须被夹住而不是竖着爬
    steep = FakePet(x=700, y=450)
    steep.walker.start_to(710, 100)
    for _ in range(2000):
        steep.tick()
    moved = 450 - steep._y
    check("纵向位移受 WALK_SLOPE 约束", moved <= 10 * config.WALK_SLOPE + 2,
          f"横向 10px → 纵向只走 {moved}px")
    check("斜率夹掉之后仍然走到目标横坐标", steep._x == 710, f"x={steep._x}")

    vert = FakePet(x=700, y=450)
    check("纯纵向目标直接拒绝（没有上下走的步态）",
          not vert.walker.start_to(700, 100), f"phase={vert.walker.phase}")

    # 途中被外力挪动（改尺寸按底边重排）：不能跳回旧高度
    r = FakePet(x=400, y=400)
    r.walker.start_to(900, 250)
    for _ in range(30):
        r.tick()
    mid = r._y
    assert mid < 400, "先确认她真的在往上走"
    r.move(r._x, mid - 60)                    # 模拟 set_size 重排窗口
    r.tick()
    check("途中窗口被外力挪动后继续走，而不是跳回旧位置",
          abs(r._y - (mid - 60 - 1)) <= 2, f"外力前 y={mid}，挪到 {mid-60}，下一 tick y={r._y}")

    # 太近不值得走；贴边也要能往反方向走
    corner = FakePet(x=0, y=0)
    corner.behaviors.rng = random.Random(6)
    for _ in range(40):
        if corner.walker.roam():
            break
        corner.facing.value = 0
    check("贴在左上角也能发起漫游", corner.walker.busy or corner.facing.value != 0,
          f"phase={corner.walker.phase}")

    def pool_pet(seed, **kw):
        p = FakePet(**kw)
        p.behaviors.enabled = True
        p.behaviors.rng = random.Random(seed)
        return p

    # 池子会自己动起来
    b = pool_pet(5, x=700)
    for _ in range(1200):
        b.tick()
    check("无人操作时确实会自发动作", b.behaviors.fired > 0,
          f"{b.behaviors.fired} 次 / {1200 * config.FPS_MS / 1000:.0f}s")

    # 安静期：用户刚碰过就不许抢
    c = pool_pet(5, x=700)
    for _ in range(1200):
        c.tick()
    base = c.behaviors.fired
    c.behaviors.note_user()
    quiet_ticks = int(config.BEHAVIOR_QUIET_S * 1000 / config.FPS_MS)
    for _ in range(quiet_ticks):
        c.tick()
    check("安静期内一次都不触发", c.behaviors.fired == base,
          f"{base} -> {c.behaviors.fired}，安静期 {quiet_ticks} tick")

    # 上一轮那个只**降权**、不踢出候选。旧写法（踢出）在候选只剩两类时等于
    # "必须严格交替"：连续重复率恒为 0，配的权重也完全不生效 —— 她是走走·坐坐的钟摆。
    # 这条断言以前用"去重后的序列"来数重复，那本身就是恒真的（去重了当然不重复），
    # 所以它抓不到任何东西。现在按 `fired` 数真实的挑选序列。
    d = pool_pet(9, x=700)
    d.side_prone = True
    d.behaviors.enabled = True
    seq = []
    fired = 0
    for _ in range(40000):
        d.tick()
        if d.behaviors.fired > fired:
            seq.append(d.behaviors.last)
            fired = d.behaviors.fired
    rep = sum(1 for a, b in zip(seq, seq[1:]) if a == b) / max(1, len(seq) - 1)
    check("允许偶尔连做同一件事（不再是严格交替）", rep > 0.03,
          f"连续重复率 {rep * 100:.1f}%，共 {len(seq)} 次")
    check("连做同一件不能变成常态（降权真的在压它）", rep < 0.35,
          f"连续重复率 {rep * 100:.1f}%")
    check("开关打开后池子里配置的动作全都在跑",
          set(seq) == set(config.BEHAVIOR_WEIGHTS),
          f"缺 {sorted(set(config.BEHAVIOR_WEIGHTS) - set(seq))}")
    # 分布是统计量，归 `tools/behavior_check.py` 量（12 种子 × 1 小时，挑中占比
    # 34/28/20/17 对得上权重 36/27/18/18）。冒烟这里只有几十次挑选，拿它断言顺序
    # 会偶发假失败 —— 这里只锁"降权而不是踢出"这条**结构**语义：
    # 旧实现里 pick() 永远给不出上一轮那个，改完之后它必须还能给出来。
    pk = FakePet(x=700)
    pk.side_prone = True
    pk.behaviors.last = "roam"
    pk.behaviors.rng = random.Random(0)
    got = {pk.behaviors.pick() for _ in range(400)}
    check("pick() 仍可能挑中上一轮那个（降权不是踢出候选）",
          got == set(config.BEHAVIOR_WEIGHTS), str(sorted(got)))
    # 开关关掉时只有"要侧过去的那一套"进不了候选：四足爬走归它管，
    # 正面趴下不归它管（用户原话："趴下的开关控制是否允许向左或向右趴下的相关动作发生"）。
    off = pool_pet(9, x=700)
    check("开关关掉时四足爬走进不了候选，但正面趴下照常在",
          not off.behaviors.available("crawl")
          and off.behaviors.available("prone")
          and off.behaviors.available("sit") and off.behaviors.available("roam"),
          f"crawl={off.behaviors.available('crawl')} prone={off.behaviors.available('prone')} "
          f"sit={off.behaviors.available('sit')}")
    check("开关关掉时 hug 仍在候选里（一次性动作不受姿态开关牵连）",
          off.behaviors.available("hug"))

    # 忙时不许叠加：busy 的每一 tick 都不能让 fired 增长
    e = pool_pet(3, x=700)
    bad = 0
    for _ in range(40000):
        was, n0 = e.behaviors.busy(), e.behaviors.fired
        e.tick()
        if was and e.behaviors.fired > n0:
            bad += 1
    check("动作进行中绝不叠加下一个", bad == 0, f"违规 {bad} tick")

    # 收尾占位：动作刚结束时 busy 仍为真，防止立刻接下一个；而且 hold 必须会自己到期
    f = FakePet(x=700)
    f.facing.value = 0
    f.walker.start(1, seconds=0.2)
    while f.walker.busy or f.facing.turning:
        f.tick()
    f.tick()
    check("走路收尾进入 hold 占位", f.behaviors.hold > 0 and f.behaviors.busy(),
          f"hold={f.behaviors.hold}")
    for _ in range(int(config.BEHAVIOR_HOLD_S * 1000 / config.FPS_MS) + 2):
        f.tick()
    check("hold 会到期，不会把池子永久锁死",
          f.behaviors.hold == 0 and not f.behaviors.busy(), f"hold={f.behaviors.hold}")

    # 上一那条的完整版：跑满一分钟，池子必须还能持续产出动作。
    # hold 曾经被算进 busy，于是"从忙转闲"每 tick 都重新成立、hold 被无限续期，
    # 结果一个自发动作都发不出来 —— 只看"没崩"根本测不到。
    k = pool_pet(2, x=700)
    k.facing.value = 0
    k.walker.start(1, seconds=0.2)
    for _ in range(60000):
        k.tick()
    check("长期运行下自发动作持续发生（不是一次性）", k.behaviors.fired > 20,
          f"{k.behaviors.fired} 次 / {60000 * config.FPS_MS / 1000 / 60:.0f} 分钟")

    # 开关：先证明开着会动，再证明关掉真的停 —— 只测后半段的话默认值就能骗过断言
    h = pool_pet(5, x=700)
    for _ in range(1200):
        h.tick()
    on = h.behaviors.fired
    h.behaviors.set_enabled(False)
    for _ in range(3000):
        h.tick()
    check("关掉随机动作后完全不自发", on > 0 and h.behaviors.fired == on,
          f"开 {on} 次 -> 关后 {h.behaviors.fired} 次")

    # 拖拽中不许触发
    i = pool_pet(5, x=700)
    i._dragging = True
    for _ in range(3000):
        i.tick()
    check("拖拽中不触发自发动作", i.behaviors.fired == 0, f"{i.behaviors.fired} 次")

    # ── 姿态：坐下 / 趴下 ───────────────────────────────────────────────
    st = FakePet(x=700, y=300)
    st.facing.value = 0
    check("可以发起坐下", st.holders["sitting"].start())
    seq = []
    for _ in range(600):
        st.tick()
        seq.append((st.posture, st.holders["sitting"].phase))
    check("坐下先播过渡再进驻留",
          ("sitting", "entering") in seq and ("sitting", "holding") in seq,
          f"经过 {sorted(set(p for _, p in seq))}")
    check("坐定后姿态=sitting 且收工回到 standing",
          seq[-1] == ("standing", "off"), f"末态 {seq[-1]}")
    ent = [i for name, i in st.frames if name == "sitdown"]
    tail = ent[-8:]
    check("起身是倒放过渡段，一直退到进入起点那一帧（不是硬切回站姿、也不是退到素材第 0 帧）",
          tail[-1] == st.manifest.anim("sitdown")["enter_at"]
          and all(b < a for a, b in zip(tail, tail[1:])),
          f"最后 8 次请求 {tail}")
    # 帧号越界会 FileNotFoundError 直接弄死桌宠。共用计数器那次 bug 就是靠这条抓住的。
    idx = [i for name, i in st.frames if name == "sitdown"]
    check("坐下全程请求的帧号都在素材范围内",
          idx and min(idx) >= 0 and max(idx) < 145,
          f"用到 {min(idx)}~{max(idx)}，共 {len(idx)} 次请求")
    # 三段切分：过渡段 [enter_at, settle_at)、待机段 loop，且 settle_at == loop 起点。
    # 素材在"坐到位"之后还接着十几帧收腿微调，再往后才是真正该循环的安静待机；
    # 交接点必须落在待机窗口第一帧，否则坐下去那一下会剪切。
    sa = st.manifest.anim("sitdown")
    ent0, settle = sa["enter_at"], sa["settle_at"]
    lo, hi = sa["loop"]
    step = sa.get("stride", 2)
    hold_seq = [i for name, i in st.frames
                if name == "sitdown" and not (ent0 <= i < settle)]
    enter_seq = list(itertools.takewhile(lambda v: v < settle, ent))
    check("进入段从标定起点开始播，不播素材开头那截她只是站着的部分",
          enter_seq and enter_seq[0] == ent0,
          f"第一个请求帧 {enter_seq[0] if enter_seq else None}，标定 {ent0}"
          f"（素材第 0 帧她站着没事干，且比标准站姿图大 3.3%）")
    check("过渡段按素材帧距推进，一直播到待机窗口第一帧才交接",
          enter_seq == list(range(ent0, settle, step)) and settle == lo,
          f"过渡 {enter_seq[:3]}…{enter_seq[-2:]}，窗口起点 {lo}，帧距 {step}")
    check("驻留只播管线标出的待机窗口，不播坐定之后多出来那段",
          bool(hold_seq) and all(lo <= i < hi for i in hold_seq),
          f"驻留用过 {sorted(set(hold_seq))[:6]}…，窗口 [{lo},{hi})")
    # "腿动的太厉害"的直接锁：窗口内平均每帧变化必须明显低于过渡段。
    # 上一版选到撇腿区，那里是 1.19 灰阶，安静待机只有 0.21~0.34。
    check("待机窗口比过渡段安静得多（不是撇腿区）",
          sa["enter_move"] / sa["hold_move"] >= 2.0,
          f"窗口 {sa['hold_move']} vs 过渡 {sa['enter_move']}")
    # 来回播：三角波最大步长是 2×stride；"播到顶再跳回起点"一步会跨整个窗口宽。
    seam = max((abs(b - a) for a, b in zip(hold_seq, hold_seq[1:])), default=0)
    check("驻留来回播，不会播到顶又跳回起点",
          len(hold_seq) > 20 and seam <= 4,
          f"驻留 {len(hold_seq)} 次请求，相邻最大跨步 {seam}")
    # 来回一轮必须慢到像呼吸。动画自身的 stride=2 会把短窗口压成一两秒一轮，那是发抖。
    uniq = len(set(hold_seq))
    period_s = (2 * uniq - 2) * config.FPS_MS / 1000
    check("坐着微动一轮 ≥2 秒（是呼吸不是发抖）", period_s >= 2.0,
          f"{uniq} 个不同帧，来回一轮 {period_s:.2f}s")
    # 上一条断的是"所有请求都在素材范围内"；这条才盯真正的问题：
    # 过渡计数器曾被驻留循环共用，一路涨到几百，倒放要几十秒才起得来。
    h3 = FakePet(x=700, y=300)
    h3.facing.value = 0
    hold = h3.holders["sitting"]
    h3.holders["sitting"].start()
    peak = 0
    for _ in range(600):
        h3.tick()
        peak = max(peak, hold.enter_frame)
    check("过渡计数器全程不超过 settle（没被驻留循环顶上去）",
          peak <= hold.settle, f"enter_frame 峰值 {peak}，settle {hold.settle}")

    # 开关的管辖范围（用户原话："趴下的开关控制是否允许向左或向右趴下的相关动作发生"）：
    # 它只管"趴着时能不能侧过去"，正面趴下是常驻动作，不归它管。
    pr = FakePet(x=700, y=300)
    pr.facing.value = 0
    check("开关关掉时照样能发起正面趴下（它不在开关的管辖范围里）",
          pr.holders["prone"].start() and pr.posture == "prone",
          f"posture={pr.posture}")
    check("关掉时趴着的朝向表只剩正面一个", pr.facing.count == 1,
          f"count={pr.facing.count}")
    for _ in range(5):
        pr.tick()
    check("趴下没有过渡素材：进入就是驻留，画的是正面那段微动",
          pr.holders["prone"].phase == "holding" and pr.frames[:1] == [("prone_idle", 0)],
          f"phase={pr.holders['prone'].phase} frames={pr.frames[:3]}")
    pr.side_prone = True
    pr.facing.value = 1                        # 假装她已经侧趴着（朝向索引 1）
    pr.frames.clear()
    pr.tick()
    check("侧趴没有微动素材：那一侧落回静帧，不会把正面那段套上去",
          pr.frames == [("pose:prone", 0)], str(pr.frames[:3]))
    for _ in range(600):
        pr.tick()
    check("趴够时间自己起来", pr.posture == config.BASE_POSTURE and not pr.holders["prone"].busy,
          f"posture={pr.posture} phase={pr.holders['prone'].phase}")

    # ── 打瞌睡：坐着打盹，正面单朝向，不受开关管 ────────────────────────────
    dz = FakePet(x=700, y=300)
    dz.facing.value = 0
    check("开关关掉时照样能发起打瞌睡", dz.holders["dozing"].start(),
          f"posture={dz.posture}")
    for _ in range(5):
        dz.tick()
    check("打瞌睡驻留画的是那段点头微动", dz.posture == "dozing"
          and any(n == "doze" for n, _ in dz.frames) and dz.facing.count == 1,
          f"posture={dz.posture} count={dz.facing.count} frames={[n for n, _ in dz.frames][:3]}")
    for _ in range(600):
        dz.tick()
    check("打瞌睡打够了自己起来", dz.posture == config.BASE_POSTURE
          and not dz.holders["dozing"].busy, f"phase={dz.holders['dozing'].phase}")

    # ── 趴着睡：左右两套侧躺的身体，独立姿态，受开关管 ──────────────────────
    # 用户澄清："趴着睡是完全趴着（侧躺闭眼），左右那个趴着是趴坐着"，
    # 以及"如果开了趴着的话，是不是再显示一个趴下睡觉"。
    # 所以它既不能当 prone 的 idle（那会"正面看是睡、一转侧面就坐起来"），
    # 也不能在开关关掉时出现。
    sl = FakePet(x=700, y=300)
    sl.facing.value = 0
    check("开关关掉时发起不了趴着睡（那本来就是侧躺的身体）",
          not sl.holders["sleeping"].start() and sl.posture == config.BASE_POSTURE,
          f"posture={sl.posture}")
    sl.side_prone = True
    check("可以发起趴着睡", sl.holders["sleeping"].start())
    for _ in range(5):
        sl.tick()
    side = sl.facing.value
    check("趴睡有两个朝向（左躺 / 右躺），进入时随机挑一侧",
          sl.posture == "sleeping" and sl.facing.count == 2 and side in (0, 1),
          f"count={sl.facing.count} side={side}")
    NAMES = ["sleep_left", "sleep_right"]
    check("趴睡驻留画的是当前那一侧的睡姿素材，不是趴坐或打瞌睡",
          sl.holders["sleeping"].phase == "holding"
          and sl.frames and all(n == NAMES[side] for n, _ in sl.frames),
          f"side={side} frames={[n for n, _ in sl.frames][:3]}")
    win = sl.manifest.anim(NAMES[side])["loop"]
    idx = [i for n, i in sl.frames if n == NAMES[side]]
    check("趴睡只在标定窗口里来回（越界就是把翻身那一段也循环进来）",
          idx and all(win[0] <= i < win[1] for i in idx),
          f"窗口{win}，用到 {min(idx)}~{max(idx)}")
    other = 1 - side
    sl.facing.value = other
    sl.frames.clear()
    sl.tick()
    check("换到另一侧就换那一段素材：idle 列表按朝向取，不是整段共用",
          sl.frames and all(n == NAMES[other] for n, _ in sl.frames),
          f"other={other} frames={[n for n, _ in sl.frames][:3]}")
    check("趴睡与趴坐是两个互不相干的姿态，不会互相顶掉",
          sl.holders["prone"].phase == "off", f"prone={sl.holders['prone'].phase}")
    sl.request_posture("prone")
    check("趴睡期间点「趴下」会排队而不是立刻换（换姿态仍走同一条队列）",
          sl.wants.key == "prone" and sl.posture == "sleeping",
          f"want={sl.wants.key} posture={sl.posture}")
    sl.wants.clear()                     # 上面那条排队的「趴下」先撤掉，专心测开关收尾
    sl.set_side_prone(False)
    for _ in range(10):
        sl.tick()
    check("睡到一半关掉开关：她起身，而不是继续侧躺着",
          sl.posture == config.BASE_POSTURE and not sl.holders["sleeping"].busy,
          f"posture={sl.posture} phase={sl.holders['sleeping'].phase}")

    # 排队里的「趴着睡」也要跟着开关作废：否则她过一会儿突然自己躺下睡，那是惊吓。
    sq = FakePet(x=700, y=300)
    sq.facing.value = 0
    sq.holders["sitting"].start()        # 先占住姿态，让趴着睡只能排队
    sq.side_prone = True
    check("开关开着时「趴着睡」会排队（占用中不硬换）",
          sq.request_posture("sleeping") and sq.wants.key == "sleeping",
          f"want={sq.wants.key}")
    sq.set_side_prone(False)
    check("关掉开关：排队的「趴着睡」当场作废", sq.wants.key is None, f"want={sq.wants.key}")

    # ── 姿态切换：手动点的必须**当场有反应**，自发的才排队 ──────────────────
    # 用户原话："为什么正在进行动作时不能切换到另一个动作。"
    # 旧行为：手动点和自发动作共用一套"排队 + 3 秒作废"，于是她走路/跟随/睡着时
    # 点一下 = 静默失败（实测跟随途中点坐下，32 秒都没坐下）。
    q1 = FakePet(x=700, y=300)
    q1.facing.value = 0
    q1.side_prone = True
    q1.holders["prone"].start()
    for _ in range(3):
        q1.tick()
    check("手动点「坐下」会先让当前姿态收尾，而不是排队等它结束",
          q1.request_posture("sitting") and q1.holders["prone"].phase == "leaving",
          f"prone={q1.holders['prone'].phase} want={q1.wants.key}")
    sat = None
    for _ in range(200):
        q1.tick()
        if q1.posture == "sitting":
            sat = q1.holders["sitting"].phase
            break
    check("收尾之后队列里的意图自动兑现（真的坐下了，不是过一会儿也没动）",
          sat is not None, f"跑完 200 tick 仍未兑现（phase={q1.holders['sitting'].phase}）")
    check("兑现之后队列清空（不会过一会儿又坐一次）",
          q1.wants.key is None, f"want={q1.wants.key}")

    # ── 手动指令打断在飞的自发行程（这次用户报的那两条都在这里）─────────────
    for tag, setup in (
            ("跟随途中", lambda p: (p.follower.set_enabled(True),
                                    setattr(p, "cursor_xy", (200, 300)))),
            ("漫游途中", lambda p: p.walker.roam())):
        m = FakePet(x=700, y=300)
        m.facing.value = 0
        setup(m)
        for _ in range(20):
            m.tick()
        m.request_posture("sitting")
        done = None
        for _ in range(80):
            m.tick()
            if m.posture == "sitting":
                done = True
                break
        check(f"{tag}点「坐下」：当场收步并坐下（旧行为是排队排到过期）",
              done, f"walker={m.walker.phase} want={m.wants.key}")
        if tag == "跟随途中":
            check("打断跟随之后她不会立刻又起步去追光标",
                  not m.follower.following, f"following={m.follower.following}")

    m2 = FakePet(x=700, y=300)
    m2.facing.value = 0
    m2.oneshot.start()
    for _ in range(5):
        m2.tick()
    m2.request_posture("sitting")
    for _ in range(80):
        m2.tick()
    check("抱抱播到一半点「坐下」：动作让位，不是一起演",
          m2.posture == "sitting" and not m2.oneshot.busy,
          f"posture={m2.posture} oneshot={m2.oneshot.phase}")

    # 自发插队仍然要有期限：她正坐着时池子挑中"趴下"，等不到就该作废，
    # 不能几十秒后突然自己趴下。
    q2 = FakePet(x=700, y=300)
    q2.facing.value = 0
    q2.side_prone = True
    q2.holders["prone"].start()
    for _ in range(3):
        q2.tick()
    q2.wants.request("sitting", manual=False)
    for _ in range(config.ticks(config.POSTURE_WAIT_S) + 40):
        q2.tick()
    check("非手动（池子）的意图等不到就作废：不许几十秒后突然自己坐下",
          q2.wants.key is None and not q2.holders["sitting"].busy,
          f"want={q2.wants.key} sitting={q2.holders['sitting'].phase}")

    # 关掉开关**不该**把她拽起来：正面趴下不在开关的管辖范围里，排队里的"坐下"
    # 也不是开关该作废的东西（那是姿态队列自己的事，等趴着驻留自然收场）。
    q3 = FakePet(x=700, y=300)
    q3.facing.value = 0
    q3.side_prone = True
    q3.holders["prone"].start()
    for _ in range(3):
        q3.tick()
    q3.wants.request("sitting", manual=False)
    q3.set_side_prone(False)                # 关掉"左右趴着"
    check("关掉左右趴着不作废与它无关的排队意图",
          q3.wants.key == "sitting", f"want={q3.wants.key}")
    check("关掉之后朝向表退回只剩正面", q3.facing.count == 1, f"count={q3.facing.count}")
    for _ in range(5):
        q3.tick()
    check("关掉左右趴着不会把已经正面趴着的她强行拽起来",
          q3.posture == "prone" and q3.holders["prone"].busy,
          f"posture={q3.posture} phase={q3.holders['prone'].phase}")

    # 开关关掉时爬走必须**挑不出来**（而不是挑出来再失败）：
    # 失败重试会白占一次调度机会。
    q4 = FakePet(x=700, y=300)
    q4.side_prone = False
    check("关掉左右趴着时 crawl 不在候选里",
          not q4.behaviors.available("crawl"),
          f"available={q4.behaviors.available('crawl')}")
    check("关掉左右趴着时 正面趴下 仍在候选里", q4.behaviors.available("prone"))
    q4.side_prone = True
    check("打开后 crawl 回到候选里", q4.behaviors.available("crawl"))

    # 爬走必须比直立走路慢：素材节拍几乎同频（22 帧 vs 23~25 帧），
    # 同速度就是"压着身子走出散步的步子"。
    sp = FakePet(x=700, y=300)
    sp.posture = "prone"
    crawl_step = sp.walker._speed()
    sp.posture = config.BASE_POSTURE
    walk_step = sp.walker._speed()
    check("四足爬走每 tick 位移低于直立走路",
          crawl_step == config.CRAWL_SPEED < walk_step,
          f"爬 {crawl_step} / 走 {walk_step}")

    # 姿态期间必须独占：走路、别的姿态、站立呼吸都不许插进来
    g = FakePet(x=700, y=300)
    g.facing.value = 0
    g.holders["sitting"].start()
    blocked_walk = blocked_other = True
    seen_postures = set()
    # 窗口取驻留下限的一半：超过 POSTURE_HOLD_S 上限她起身是**正确行为**，
    # 原来跑满 400 tick 把正常起身误判成了"姿态闪烁"。
    for _ in range(int(min(config.POSTURE_HOLD_S["sitting"]) * 1000 / config.FPS_MS) // 2):
        g.tick()
        if g.holders["sitting"].busy:
            seen_postures.add(g.posture)
            if g.walker.start(1, seconds=1.0):
                blocked_walk = False
            if g.holders["prone"].start():
                blocked_other = False
    check("坐着期间走不了路", blocked_walk)
    check("坐着期间不会再叠一个别的姿态", blocked_other)
    check("坐着期间不会切回 standing（不是闪烁）", seen_postures == {"sitting"},
          f"见过 {sorted(seen_postures)}")

    # 拖拽要能立刻起身
    h2 = FakePet(x=700, y=300)
    h2.facing.value = 0
    h2.holders["sitting"].start()
    for _ in range(60):
        h2.tick()
    assert h2.holders["sitting"].busy, "先确认真的坐着"
    for hd in h2.holders.values():
        hd.stop()
    for _ in range(400):
        h2.tick()
    check("拖起来之后会起身并回到站立",
          not h2.active_posture_busy() and h2.posture == config.BASE_POSTURE,
          f"posture={h2.posture}")

    # 卡死回归锁。曾经"向右走 → 单击换朝向 → 菜单直选 → 拖走"之后，
    # 行程状态机永远停在 returning —— 不走路、不呼吸、也不再自发行动。
    # 根因是 Facing.turn_to(q) 不带 done 时会把上一个调用者挂着的 done 一起顶掉，
    # 而状态机把完成契约全押在回调上。纯逻辑、毫秒级，所以不放进 --deep 矩阵。
    s = FakePet(x=200, y=300)
    s.facing.value = 0
    s.walker.start(1, seconds=3.0)
    for _ in range(3):
        s.tick()
    s.facing.cycle()                                  # 用户在行走途中单击换朝向
    for _ in range(3):
        s.tick()
    s.facing.turn_to(0)                               # 又从菜单直选正面
    for _ in range(400):
        s.tick()
    check("转身回调被后来的 turn_to 顶掉后，行程仍会收尾而不是永久卡死",
          not s.walker.busy and not s.facing.turning,
          f"phase={s.walker.phase} turning={s.facing.turning}")

    # ── 左右趴着开关（只管侧过去 + 四足爬走，正面趴下常驻）──────────────────
    off = FakePet(x=700, y=300)
    off.facing.value = 0
    check("开关关掉时不能四足爬走", not off.crawler.start(), f"phase={off.crawler.phase}")
    names = set()
    o2 = pool_pet(11, x=700)
    for _ in range(40000):
        o2.tick()
        names.add(o2.behaviors.last)
    check("开关关掉时 crawl 永远不进候选", "crawl" not in names,
          f"出现过 {sorted(n for n in names if n)}")

    on = FakePet(x=700, y=300)
    on.facing.value = 0
    on.side_prone = True
    check("开关打开后可以四足爬走", on.crawler.start(), f"phase={on.crawler.phase}")
    check("爬走前她先趴下", on.posture == "prone", on.posture)
    moved = (on._x, on._y)
    anims = set()
    for _ in range(1500):
        on.tick()
        for name, _i in on.frames:
            if name.startswith("walk"):
                anims.add(name)
        if not on.crawler.busy and not on.walker.busy:
            break
    check("爬走用的是四足步态，不是站立步态", anims <= {"walk4_left", "walk4_right"},
          f"用到 {sorted(anims)}")
    check("爬完会站起来", on.posture == config.BASE_POSTURE and not on.crawler.busy,
          f"posture={on.posture}")
    n4 = pool_pet(13, x=700)
    n4.side_prone = True
    got = set()
    for _ in range(40000):
        n4.tick()
        got.add(n4.behaviors.last)
    check("开关打开后 crawl 进候选", "crawl" in got, f"出现过 {sorted(n for n in got if n)}")

    # ── 一次性动作（拥抱）：正放一遍就交还，没有驻留 ────────────────────────
    # 关键要锁的是"不循环"：素材首尾接缝实测是相邻帧差的 2.9 倍，
    # 一旦哪天有人给它接上 loop，每圈都会硬跳一下。
    h1 = FakePet(x=700, y=300)
    h1.facing.value = 0
    finished = []
    check("空闲时能发起拥抱", h1.oneshot.start(done=lambda: finished.append(1)))
    played = []
    for _ in range(80):
        before = len(h1.frames)
        h1.tick()
        played += [i for n, i in h1.frames[before:] if n == config.HUG_ANIM]
    total = int(h1.manifest.anim(config.HUG_ANIM)["frames"])
    check("拥抱正放一遍就交还（不驻留、不循环）",
          h1.oneshot.phase == "off" and finished == [1],
          f"phase={h1.oneshot.phase} 回调={finished}")
    check("播过的帧号严格递增且不回头（回播就是循环的证据）",
          played == sorted(played) and len(set(played)) == len(played)
          and max(played) < total,
          f"{played[:4]}…{played[-3:]}，素材共 {total} 帧")
    check("拥抱占用的时长就是一次素材的长度（约 3 秒，不是几十秒）",
          len(played) <= total, f"画了 {len(played)} 帧")

    h2 = FakePet(x=700, y=300)
    h2.facing.value = 2
    check("侧着时也能发起拥抱", h2.oneshot.start())
    check("先转回正面再播（拥抱素材只有正面）",
          h2.oneshot.phase == "turning" and not h2.facing.value == 0,
          f"phase={h2.oneshot.phase} facing={h2.facing.value}")
    for _ in range(80):
        h2.tick()
    check("转完自动接着播完", h2.facing.value == config.FRONT_FACING
          and h2.oneshot.phase == "off" and h2.frames,
          f"facing={h2.facing.value} phase={h2.oneshot.phase}")

    # 互斥要**双向**都锁：她忙着的时候不能插拥抱，插了拥抱之后别的也不能抢。
    h9 = FakePet(x=700, y=300)
    h9.facing.value = 2
    h9.oneshot.start()
    h9.tick()
    h9.facing.cycle()                       # 转身在飞时用户点了一下：done 被顶掉
    for _ in range(90):
        h9.tick()
    check("转身被用户的单击顶掉之后拥抱不会永久卡住（回调被人偷走是常态）",
          h9.oneshot.phase == "off" and not h9.oneshot.busy,
          f"phase={h9.oneshot.phase}")

    h3 = FakePet(x=700, y=300)
    h3.facing.value = 0
    h3.walker.start(1)
    check("走路期间发起不了拥抱", not h3.oneshot.start(), f"phase={h3.oneshot.phase}")
    h4 = FakePet(x=700, y=300)
    h4.facing.value = 0
    h4.holders["sitting"].start()
    check("姿态驻留期间发起不了拥抱", not h4.oneshot.start(), f"phase={h4.oneshot.phase}")
    h5 = FakePet(x=700, y=300)
    h5.facing.value = 0
    check("发起拥抱", h5.oneshot.start())
    for _ in range(2):
        h5.tick()
    check("拥抱期间走不了路", not h5.walker.start(1), f"walker={h5.walker.phase}")
    check("拥抱期间坐不下", not h5.holders["sitting"].start())
    h5.side_prone = True
    check("拥抱期间爬不走", not h5.crawler.start())
    check("拥抱期间发起不了第二个拥抱", not h5.oneshot.start())

    h6 = FakePet(x=700, y=300)
    h6.facing.value = 0
    h6.oneshot.start()
    for _ in range(4):
        h6.tick()
    mid = h6.oneshot.frame
    h6.oneshot.stop()                       # 用户上手：不演完
    check("用户上手时当场结束，不硬演完",
          h6.oneshot.phase == "off" and not h6.oneshot.busy,
          f"phase={h6.oneshot.phase} 停在第 {mid} 帧")
    h7 = FakePet(x=700, y=300)
    h7.facing.value = 0
    h7.oneshot.start()
    for _ in range(5):
        h7.tick()
    before = h7.oneshot.frame
    h7.oneshot.redraw()                     # 改尺寸只重画当前帧
    check("改尺寸重画不推进状态机", h7.oneshot.frame == before
          and [n for n, _ in h7.frames].count(config.HUG_ANIM) == len(h7.frames),
          f"frame={h7.oneshot.frame}（重画前 {before}）")

    h8 = FakePet(x=700, y=300)
    real_anim = h8.manifest.anim
    h8.manifest.anim = lambda name, _r=real_anim: None if name == config.HUG_ANIM else _r(name)
    check("素材没到位时发起失败而不是空转", not h8.oneshot.start(),
          f"phase={h8.oneshot.phase}")
    check("素材没到位时池子不会把 hug 摆进候选", not h8.behaviors.available("hug")
          or not h8.behaviors.run("hug"),
          f"available={h8.behaviors.available('hug')}")

    # ── 长跑：左右趴着开关打开时的 4 小时自发运行 ──────────────────────────
    # 这是 v0.9 新辟的组合空间：走 + 跑 + 爬 + 坐 + 趴 + 一次性动作六种执行者抢同一个
    # walker / 同一份转身回调。短矩阵只能覆盖"两两"，抢手死锁往往是三方的。
    soak = pool_pet(77, x=700, y=400)
    soak.side_prone = True
    fired0 = soak.behaviors.fired
    stuck_for = 0
    worst_oob = None
    gait_seen, posture_seen = set(), set()
    for n in range(180000):                       # 180000 tick × 80ms = 4 小时
        soak.tick()
        if n % 200 == 0:
            posture_seen.add(soak.posture)
            if soak.walker.busy:
                gait_seen.add(soak.walker.gait)
            if not (0 <= soak._x <= soak._right - soak._w and 0 <= soak._y <= soak._bottom - soak._w):
                worst_oob = (soak._x, soak._y)
        # "一直在忙"超过 90 秒就是卡住了：任何单个动作都不该跑这么久
        if (soak.walker.busy or soak.crawler.busy or soak.active_posture_busy()
                or soak.oneshot.busy):
            stuck_for += 1
            if stuck_for > 1125:                  # 1125 tick = 90s
                break
        else:
            stuck_for = 0
    check("全开跑 4 小时：自发动作持续发生（没有自我锁死）",
          soak.behaviors.fired - fired0 > 200,
          f"{soak.behaviors.fired - fired0} 次 / 4 小时")
    check("全开跑 4 小时：没有任何一次连续忙超过 90 秒",
          stuck_for <= 1125, f"最长连续忙 {stuck_for} tick = {stuck_for * 0.08:.0f}s")
    check("全开跑 4 小时：走/跑/爬/坐/趴/打瞌睡/趴睡 都真的出现过",
          {"walk", "run"} <= gait_seen
          and {"standing", "sitting", "prone", "dozing", "sleeping"} <= posture_seen,
          f"步态 {sorted(gait_seen)} 姿态 {sorted(posture_seen)}")
    check("全开跑 4 小时：位置始终合法", worst_oob is None, f"越界 {worst_oob}")
    check("全开跑 4 小时：拥抱这类一次性动作也真的播出来过",
          any(n == config.HUG_ANIM for n, _ in soak.frames),
          f"画过的动画 {sorted({n for n, _ in soak.frames})[:6]}…")

    # ── 跑步：同一套行程机制，远途自动换快步态 ──────────────────────────
    def used_anims(pet, run_for=1200):
        for _ in range(run_for):
            pet.tick()
            if not pet.walker.busy and not pet.crawler.busy:
                break
        return {n for n, _i in pet.frames if n.startswith(("walk", "run"))}

    def walk_ticks(dx, gait):
        """跑完一趟 dx 像素的行程要用多少 tick。走/跑各测一次就能比快慢。"""
        p = FakePet(x=700, y=300)
        assert p.walker.start_to(700 + dx, 300, gait=gait)
        return _ticks_to_done(p), used_anims(p)

    t_walk, u_walk = walk_ticks(600, "walk")
    t_run, u_run = walk_ticks(600, "run")
    check("远途自动换成跑步态", "run_right" in u_run and not (u_run & {"walk_left", "walk_right"}),
          f"用到 {sorted(u_run)}")
    check("跑比走明显快（同样 600px）", t_run < t_walk and t_walk / max(1, t_run) >= 1.4,
          f"走 {t_walk} tick / 跑 {t_run} tick，{t_walk / max(1, t_run):.2f}×")
    t_near, u_near = walk_ticks(120, None)
    check("近处仍然只是走", not (u_near & {"run_left", "run_right"}), f"用到 {sorted(u_near)}")

    m2 = FakePet(x=700, y=300)
    m2.walker.start(1, seconds=8.0)
    check("菜单上的\"向右走\"哪怕走 8 秒也只是走",
          m2.walker.gait == "walk", f"gait={m2.walker.gait}")

    # 菜单"随便走走"同理：文案写着走，就不能偷偷换成跑。
    # 这条必须带"同一批种子在自动模式下确实会跑"的对照，否则它测的是空集。
    pinned, auto = set(), set()
    for seed in range(14):
        for gait, bag in (("walk", pinned), (None, auto)):
            p = FakePet(x=200, y=300)
            p.walker.rng = random.Random(seed)
            p.walker.roam(gait=gait)
            for _ in range(1500):
                p.tick()
                bag |= {n for n, _i in p.frames if n.startswith(("walk", "run"))}
                if not p.walker.busy:
                    break
    check("菜单\"随便走走\"钉死为走（对照：同样的种子自动模式下确实会跑）",
          not (pinned & {"run_left", "run_right"}) and (auto & {"run_left", "run_right"}),
          f"钉走 {sorted(pinned)} / 自动 {sorted(auto)}")

    # 趴着没有跑的素材：爬走必须只用四足走，绝不能被升级成跑步态
    c2 = FakePet(x=700, y=300)
    c2.side_prone = True
    c2.facing.value = 0
    c2.walker.rng = random.Random(3)
    c2.crawler.start()
    cused = set()
    for _ in range(1500):
        c2.tick()
        cused |= {n for n, _i in c2.frames if n.startswith(("walk", "run"))}
        if not c2.crawler.busy and not c2.walker.busy:
            break
    check("四足爬走不会被升级成跑步（趴姿没有跑的素材）",
          cused <= {"walk4_left", "walk4_right"}, f"用到 {sorted(cused)}")

    # 步态相位：清单给了 loop=[0,92) 的素材，帧号绝不能跑到 92 之后去
    lp = FakePet(x=700, y=300)
    lp.posture = "prone"
    lp.side_prone = True
    lp.facing.value = 1
    lp.walker.dirn = 1
    lp.walker.gait = "walk"
    lp.walker.phase = "walking"
    lp.walker.frame = 0
    lp.walker.tx, lp.walker.ty = 1390, 300
    lp.walker.step_y = 0
    lp.walker.base_y, lp.walker._shown_y = 300.0, 300
    seq = []
    for _ in range(200):
        lp.tick()
        seq += [i for n, i in lp.frames if n == "walk4_right"]
    check("步态循环截到整数周期（不会播到接缝之后）",
          seq and max(seq) < 92 and min(seq) >= 0,
          f"用到 {min(seq)}~{max(seq)}")


def _ticks_to_done(pet, limit=1500) -> int:
    """把一次行程跑到结束，返回用了多少 tick。用于比较走和跑的实际快慢。"""
    for n in range(limit):
        pet.tick()
        if not pet.walker.busy:
            return n + 1
    return limit


def prone_real_tests(pet, m):
    """四足那一组走一遍真实窗口 + 真实素材。

    前面 `behavior_unit_tests` 用的是 FakePet（姿态/朝向表是桩），这里必须补真素材这一路：
    线性 180° 转身用的是 manifest 里标定出来的 keys/path，桩数据测不到它接没接对。
    """
    pet.set_side_prone(True)
    pet.enter_posture("prone")
    check("趴姿朝向表来自真实清单", pet.facing.count == 3 and
          pet.posture_faces() == ["prone_front", "prone_left", "prone_right"],
          f"count={pet.facing.count} {pet.posture_faces()}")
    check("趴姿调度器是线性弧（不是环形 360°）",
          type(pet.facing.director).__name__ == "LinearTurn",
          type(pet.facing.director).__name__)
    check("趴姿调度器用的是标定帧表",
          pet.facing.director.keys == m.anim("prone_turn")["keys"],
          f"{pet.facing.director.keys} vs {m.anim('prone_turn')['keys']}")

    pet.facing.turn_to(2)
    done = wait_until(lambda: not pet.facing.turning, 6000)
    check("趴着也能转到最右那一侧", done and pet.facing.value == 2,
          f"value={pet.facing.value}")
    check("转完落在标定静帧上", Path(pet.current_pose().path).stem == "prone_right",
          Path(pet.current_pose().path).stem)

    # 这一组才是真正抓住"转身图顺序错位"的断言：
    # 上面那条恰好通过，是因为"右"在两张表里都排最后；"左"排第二 vs 第三，就露馅了。
    for side, want in (("left", "prone_left"), ("right", "prone_right")):
        idx = m.walk_facing("prone", side)
        pet.facing.turn_to(idx)
        ok = wait_until(lambda: not pet.facing.turning, 6000)
        got = Path(pet.current_pose().path).stem
        check(f"趴着按\"向{'左' if side == 'left' else '右'}看\"的索引转身，落点正确",
              ok and got == want, f"索引{idx} → {got}，应为 {want}")

    pet.facing.turn_to(1)                    # 先侧到趴着的左侧
    ok_left = wait_until(lambda: not pet.facing.turning, 6000)
    got = Path(pet.current_pose().path).stem
    pet.set_side_prone(False)                # 再关掉"左右趴着"
    # 开关**不撤族**：正面趴下是常驻动作，不在它的管辖范围里。她该从侧趴退回正面趴，
    # 而不是被拽起来站好。（v0.11 把它扩成管整族趴着，v0.13 按用户这句话收回来。）
    check("关掉左右趴着时她退回正面趴，而不是站起来",
          ok_left and got == "prone_left" and pet.posture == "prone"
          and pet.facing.value == 0 and Path(pet.current_pose().path).stem == "prone_front",
          f"关之前={got} posture={pet.posture} facing={pet.facing.value} "
          f"关之后={Path(pet.current_pose().path).stem}")
    check("关掉后趴姿朝向表只剩正面：被禁姿势是从表里消失，不是靠入口另挡一层",
          pet.posture_faces() == ["prone_front"] and pet.facing.count == 1,
          f"{pet.posture_faces()} count={pet.facing.count}")
    # 趴姿的驻留微动只有正面那一段：侧趴没有素材，画的就是**当前朝向**那张静帧。
    # （以前 show_posture 写死 facing=0，侧趴着驻留时每一 tick 都被重画成正趴 ——
    #   观感就是"趴着趴着自己翻回正面"。）
    pet.holders["prone"].start()
    holding = wait_until(lambda: pet.holders["prone"].phase == "holding", 4000)
    drawn = []
    real_pose, real_anim = pet.show_pose, pet.show_anim_frame
    pet.show_pose = lambda p, _r=real_pose: (drawn.append(("pose", p.key)), _r(p))
    pet.show_anim_frame = lambda n, i, _r=real_anim: (drawn.append(("anim", n)), _r(n, i))
    pet.side_prone = True
    pet.facing.value = 1
    pet.holders["prone"].redraw()
    check("侧趴驻留画的是当前那一侧的静帧，不是正面那张", holding
          and drawn[-1] == ("pose", "prone_left"), str(drawn[-2:]))
    drawn.clear()
    pet.facing.value = 0
    pet.holders["prone"].redraw()
    check("正面趴着的驻留画的是那段微动（正面趴着.mp4），不是死图",
          drawn and all(n == "prone_idle" for _, n in drawn), str(drawn[-2:]))
    drawn.clear()
    pet.set_side_prone(False)
    pet.show_pose, pet.show_anim_frame = real_pose, real_anim
    check("侧趴退回正面趴：先落回正面静帧，再接上正面那段微动",
          pet.posture == "prone" and pet.facing.value == 0
          and ("pose", "prone_front") in drawn and drawn[-1] == ("anim", "prone_idle"),
          str(drawn[-3:]))
    # 被禁姿势仍然要锁：朝向操作若能取到 prone_left/right 就等于从另一侧绕回这个开关。
    seen = set()
    for _ in range(12):
        pet.facing.cycle()
        wait_until(lambda: not pet.facing.turning, 3000)
        seen.add(Path(pet.current_pose().path).stem)
    check("开关关掉后任何朝向操作都拿不到被禁姿势",
          not (seen & {"prone_left", "prone_right"}), str(sorted(seen)))
    check("开关关掉后正面趴下照样能发起（常驻动作不因开关而点不动）",
          pet.holders["prone"].busy or pet.request_posture("prone"),
          f"phase={pet.holders['prone'].phase} posture={pet.posture}")

    def _all_texts(mm):
        out = []
        for a in mm.actions():
            if a.text():
                out.append(a.text())
            if a.menu():
                out += _all_texts(a.menu())
        return out

    def _texts(pet):
        mm = pet.build_menu()
        out = _all_texts(mm)
        mm.setParent(None)
        mm.deleteLater()
        return out

    texts = _texts(pet)
    check("趴下是常驻项，不受开关影响：开关关着时菜单里也点得到",
          "趴下" in texts and "左右趴着" in texts, str(texts))
    check("趴着睡跟着「左右趴着」开关出现：关着时菜单里根本没有这条入口",
          "趴着睡" not in texts, str(texts))
    check("打瞌睡与抱抱都排在姿态子菜单里（抱抱不再单挂一级）",
          all(t in texts for t in ("打瞌睡", "抱抱"))
          and texts.index("坐下") < texts.index("打瞌睡") < texts.index("趴下")
          < texts.index("抱抱") < texts.index("左右趴着"),
          str(texts))
    check("移动子菜单只剩「去哪」这一层：走去坐下 / 走到鼠标旁 两条行程已删",
          "移动" in texts
          and all(t in texts for t in ("随便走走", "跑向远处", "向左走", "向右走"))
          and not ({"找个地方坐下", "走到鼠标旁"} & set(texts)), str(texts))
    check("跟随鼠标是一项开关，和随机动作并排", "跟随鼠标" in texts
          and texts.index("随机动作") < texts.index("跟随鼠标"), str(texts))
    pet.set_side_prone(True)
    on = _texts(pet)
    check("开了左右趴着，趴着睡才出现在姿态子菜单里（排在趴下之后）",
          "趴着睡" in on and on.index("趴着睡") > on.index("趴下"), str(on))
    pet.set_side_prone(False)
    check("开关文案说的是它真正管的东西（左右趴着，不是整族趴下）",
          "趴着与四足" not in texts, str(texts))

    # 非站姿不许摆站立的朝向表：坐着时朝向集只有一张图，把 正面/左面/背面/右面
    # 摆过去就是一排点了会指到不存在姿势的项。
    stand_up(pet)                            # 上面那一节把她留在趴着，坐姿那一跳得先归零
    pet.request_posture("sitting")
    for _ in range(40):
        QApplication.processEvents()
        pet.tick()
        if pet.holders["sitting"].phase == "entering":
            break
    sit_menu = pet.build_menu()
    sit_texts = _all_texts(sit_menu)
    sit_menu.setParent(None)
    sit_menu.deleteLater()
    check("坐着时菜单不摆站立的朝向表", "朝向" not in sit_texts and pet.posture == "sitting",
          f"posture={pet.posture} {sit_texts}")
    pet.holders["sitting"].stop()
    wait_until(lambda: pet.posture == config.BASE_POSTURE, 5000)

    pet.set_side_prone(True)
    on_menu = pet.build_menu()
    on_texts = _all_texts(on_menu)
    check("开关打开后'趴下'才出现在姿态子菜单里", "趴下" in on_texts, str(on_texts))
    on_menu.setParent(None)
    on_menu.deleteLater()

    stand_up(pet)
    pet.place_default()
    started = pet.crawler.start()
    check("真实窗口能发起四足爬走", started, f"posture={pet.posture}")
    used = set()
    real = pet.show_anim_frame
    pet.show_anim_frame = lambda name, i: (used.add(name), real(name, i))
    finished = wait_until(lambda: not pet.crawler.busy and not pet.walker.busy, 20000)
    pet.show_anim_frame = real
    check("爬走会自己收尾（不用人管）", finished, f"posture={pet.posture}")
    check("爬走用真实四足步态而不是站立步态，且结束时回到站姿",
          bool(used & {"walk4_left", "walk4_right"})
          and not (used & {"walk_left", "walk_right"})
          and pet.posture == config.BASE_POSTURE,
          f"用到 {sorted(used)} posture={pet.posture}")
    pet.set_side_prone(config.SIDE_PRONE)


def stand_up(pet):
    """把姿态相关的一切请回站姿。菜单结构那一节是按站姿断言的，起点必须确定。"""
    pet.clear_posture_want()
    pet.oneshot.stop()
    for h in pet.holders.values():
        h.stop()
    pet.crawler.stop()
    ok = wait_until(lambda: not pet.active_posture_busy() and not pet.walker.busy
                    and not pet.crawler.busy and not pet.oneshot.busy, 15000)
    if pet.posture != config.BASE_POSTURE:
        pet.enter_posture(config.BASE_POSTURE)
        pet.facing.apply(0)
    return ok


def menu_child() -> int:
    """子进程入口：复刻一次真实的菜单生命周期，崩了就整个进程 abort。

    单独开子进程是因为这类崩溃是 C++ 层 abort，不抛 Python 异常，
    混在主套件里会把后面所有测试一起带走。
    """
    m = Manifest()
    pet = PetWindow(m, AssetStore())
    pet.show()
    app.processEvents()
    menu = pet.build_menu()
    act = menu.actions()[0].menu().actions()[1]        # 朝向 → 第 2 项
    # 走和菜单连线完全相同的路径：触发 → _later → 单发定时器
    menu.actions()[0].menu().triggered.emit(act)
    del menu                                            # exec_() 返回后局部 QMenu 就在这里析构
    pump(500)
    print(f"CHILD_OK facing={pet.facing.value}")
    return 0


def menu_lifetime_test():
    """回归锁：延迟回调里绝不能引用菜单拥有的对象。"""
    import subprocess
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--menu-child"],
                       capture_output=True, text=True, timeout=60,
                       env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    out = (r.stdout or "") + (r.stderr or "")
    check("菜单关闭后触发的朝向动作不崩进程", r.returncode == 0 and "CHILD_OK" in out,
          f"exit={r.returncode} {out.strip()[-90:]}")


def interaction_matrix(pet):
    """动作串联矩阵：穷举两两组合 + 随机长链，逐步查不变式。

    单个动作各自都有断言，但"向右走途中点回到右下角"这类**只在串联里暴露**的
    缺陷（目标点是出发时算的绝对坐标，人一旦被挪走，旧目标就成了别人的位置）
    靠单动作断言永远测不到。这里手动 tick，不走事件循环，所以完全可复现。

    矩阵跑的是**状态机不变式**，不是像素，所以整段关掉真实缩放渲染：
    一帧 768→420 的重缩放约 7ms，144 组 + 长链要跑十多万 tick，
    开着渲染整轮冒烟会涨到 8 分半 —— 那就没人愿意每次改动后跑了。
    渲染路径另有 `render_check.py` 和本文件里的"切换尺寸后 pixmap 非空"覆盖。
    """
    import random as _random

    from PyQt5.QtGui import QPixmap
    blank = QPixmap(1, 1)
    blank.fill()
    store = pet.store
    real = (store.pix, store.frame_pix)
    store.pix = store.frame_pix = lambda *a, **k: blank
    try:
        _matrix_body(pet, _random)
    finally:
        store.pix, store.frame_pix = real


def _matrix_body(pet, _random):

    desktop = QApplication.desktop().availableGeometry(pet)
    rng = _random.Random(20261009)

    def drag_to(x, y):
        """复刻 mouseMoveEvent 的语义：先中止行程，再挪，再夹回可视区。"""
        pet.walker.stop()
        pet._dragging = True
        pet.move(x, y)
        pet._clamp_to_screen()
        pet._dragging = False

    OPS = [
        ("单击循环", lambda: pet.facing.cycle()),
        ("菜单转正面", lambda: pet.facing.turn_to(0)),
        ("菜单转背面", lambda: pet.facing.turn_to(2)),
        ("向左走", lambda: pet.walker.start(-1, seconds=2.5)),
        ("向右走", lambda: pet.walker.start(1, seconds=2.5)),
        ("随机漫游", lambda: pet.walker.roam()),
        ("改尺寸", lambda: pet.set_size(rng.randint(config.SIZE_MIN, config.SIZE_MAX))),
        ("回到右下角", lambda: pet.place_default()),
        ("拖到左上", lambda: drag_to(desktop.left(), desktop.top())),
        ("拖到中央", lambda: drag_to(desktop.center().x(), desktop.center().y())),
        ("切换置顶", lambda: pet.set_always_on_top(not pet.always_on_top)),
        ("重设皮肤", lambda: pet.set_variant(None)),
        # 姿态是这次新增的维度，也是矩阵第一次手抓出 bug 的地方
        # （坐着期间 walker 没查姿态占用，会被走路指令叫起来）——必须常驻在这里。
        ("坐下", lambda: pet.holders["sitting"].start()),
        ("打瞌睡", lambda: pet.holders["dozing"].start()),
        ("趴下", lambda: pet.holders["prone"].start()),
        ("趴着睡", lambda: pet.holders["sleeping"].start()),
        # 一次性动作是新加的第四类执行者：它占的时间短、没有驻留、播完就交还，
        # 和走路/姿态/爬走/拖拽撞在一起才看得出互斥有没有漏。
        ("抱抱", lambda: pet.oneshot.start()),
        # 开关是"能撤销掉一整套动作"的操作，它和正在飞的爬走/姿态/走路撞在一起
        # 才有意思：关掉之后不该再有被禁的姿势残留。
        ("左右趴着开", lambda: pet.set_side_prone(True)),
        ("左右趴着关", lambda: pet.set_side_prone(False)),
        # 跟随是"她会自己朝光标走"的开关，和拖拽/回位/姿态撞在一起才看得出
        # 到底谁在挪这个窗口：用户把她拖到别处之后她不该再自己走回去。
        ("跟随开", lambda: pet.set_follow(True)),
        ("跟随关", lambda: pet.set_follow(False)),
        ("四足爬走", lambda: pet.crawler.start()),
        # 跑步是"同一趟行程换个步态"，所以必须和走路、姿态、拖拽撞在一起测：
        # 步态选择发生在 _begin，中途被拖走或被坐下打断时不能留下跑啊跑啊的僵尸状态。
        ("跑向远点", lambda: pet.walker.start_to(
            pet.screen_left() + (pet.screen_right() - pet.screen_left() - pet.width())
            * (1 if pet.x() < (pet.screen_left() + pet.screen_right()) / 2 else 0),
            pet.screen_top() + 60)),
    ]
    # 会把窗口整个挪走的操作：事后位置必须"钉住"，不能被旧目标拽回去
    TELEPORT = {"回到右下角", "拖到左上", "拖到中央"}

    def settle(n=400):
        # 等静止前必须先关掉自发行为：池子开着她就该一直动作，
        # 那时"收不了尾"是正确行为，拿它当失败会把 282/300 条全误判成 bug。
        pet.behaviors.enabled = False
        for _ in range(n):
            pet.tick()
            # 跟随开着且她正在走 = 光标还在远处，这就是设计行为，不是收不了尾。
            # 照常往下走不变式检查，正好把"跟随途中"这一态也覆盖掉。
            if pet.follower.enabled and pet.walker.busy:
                return True
            if not pet.walker.busy and not pet.facing.turning and not pet.oneshot.busy:
                return True
        return False

    def invariants(tag):
        bad = []
        # Qt 的 right() 是"最后一个被覆盖的像素"，所以合法上界是 right()+1-width
        if not (desktop.left() <= pet.x() <= desktop.right() + 1 - pet.width()):
            bad.append(f"越界 x={pet.x()} 上界={desktop.right() + 1 - pet.width()}")
        if not (desktop.top() <= pet.y() <= desktop.bottom() + 1 - pet.height()):
            bad.append(f"越界 y={pet.y()} 下界={desktop.bottom() + 1 - pet.height()}")
        if pet.height() != pet.pet_h or pet.width() != pet.pet_w:
            bad.append(f"窗口尺寸不符 {pet.width()}x{pet.height()}≠{pet.pet_w}x{pet.pet_h}")
        if pet.facing.value not in range(4):
            bad.append(f"朝向越界 {pet.facing.value}")
        if pet.walker.phase not in ("off", "turning", "walking", "returning"):
            bad.append(f"未知 phase={pet.walker.phase}")
        if not pet.side_prone and pet.crawler.busy:
            bad.append("开关关着但爬走还在跑")
        if pet.oneshot.phase not in ("off", "turning", "playing"):
            bad.append(f"未知一次性动作 phase={pet.oneshot.phase}")
        # 只有一个绘制目标：任何两个执行者同时占着，画出来的就是互相抢帧的乱动作。
        if pet.oneshot.busy and (pet.active_posture_busy() or pet.crawler.busy):
            bad.append(f"一次性动作与姿态/爬走同时在跑 oneshot={pet.oneshot.phase}")
        if pet.walker.gait not in ("walk", "run"):
            bad.append(f"未知步态 {pet.walker.gait}")
        if pet.walker.phase == "walking" and pet.walker.gait == "run" \
                and not pet.manifest.has_gait(pet.posture, "run"):
            bad.append(f"{pet.posture} 姿态没有跑的素材却在跑")
        if not pet.side_prone and pet.posture == "prone" and pet.facing.value != 0:
            bad.append(f"开关关着却停在被禁的趴姿朝向 {pet.facing.value}")
        if not pet.side_prone and pet.posture == "sleeping":
            bad.append("开关关着却还在趴着睡（那套侧躺身体不该存在）")
        if pet.follower.enabled and pet.walker.busy and pet.posture != config.BASE_POSTURE:
            bad.append(f"跟随在姿态期间抢步：posture={pet.posture}")
        if pet.idle.running and pet.facing.value != config.IDLE_FACING:
            bad.append(f"非正面却在播待机 facing={pet.facing.value}")
        cr = pet.content_rect()
        if cr.right() > pet.width() + 1 or cr.bottom() > pet.height() + 1:
            bad.append(f"内容溢出窗口 {cr}")
        if bad:
            fails.setdefault(tag, []).extend(bad)
        return not bad

    fails, stuck, drift = {}, [], []
    for a_name, a_fn in OPS:
        for b_name, b_fn in OPS:
            tag = f"{a_name}→{b_name}"
            try:
                pet.behaviors.enabled = False
                pet.set_size(config.DEFAULT_H)
                pet.place_default()
                pet.facing.apply(0)
                settle(60)
                a_fn()
                for _ in range(rng.randint(1, 25)):     # 随机打断时机，别只在起点/终点组合
                    pet.tick()
                b_fn()
                after = (pet.x(), pet.y())
                for _ in range(6):
                    pet.tick()
                # 开着跟随时"挪走之后她又走开"是**设计如此**（她追的是实时光标，
                # 不是出发时算的旧目标），这条断言只抓旧目标复活的场景。
                if (b_name in TELEPORT and not pet.follower.enabled
                        and (pet.x(), pet.y()) != after):
                    drift.append(f"{tag}: 挪走后又被旧目标拽动 "
                                 f"{after} -> {(pet.x(), pet.y())}")
                if not settle() and not pet.follower.enabled:
                    # 开着跟随时"一直忙"是正常态：光标停在远处她就一直小步走过去，
                    # 拿它当收不了尾会把每一组跟随组合全误判成死锁。
                    stuck.append(tag)
                invariants(tag)
            except Exception as e:                       # noqa: BLE001
                fails.setdefault(tag, []).append(f"抛异常 {type(e).__name__}: {e}")

    check(f"动作两两串联 {len(OPS) ** 2} 组：位置不越界、尺寸/朝向/待机一致",
          not fails, f"{len(fails)} 组有问题：" + "; ".join(
              f"{k}={v[0]}" for k, v in list(fails.items())[:4])[:300])
    check("串联后不会永远卡在忙状态", not stuck, f"卡住 {len(stuck)} 组：{stuck[:4]}")
    check("挪走她之后不会被旧目标拽回去", not drift, f"{len(drift)} 组：" + "; ".join(drift[:3]))

    # ── 随机长链：两两组合测不到的"三方抢手"在这里 ──────────────────────
    # 自发行为也开着：手动操作和池子自己挑的动作会抢同一个 walker、同一个转身回调。
    # 前面那个"转身在飞时 stop() 没撤销 done"的 bug 就是这一类。
    chain_bad = []
    ops = [o for o, _ in OPS]
    fn = dict(OPS)
    for trial in range(300):
        pet.set_size(config.DEFAULT_H)
        pet.place_default()
        pet.facing.apply(0)
        settle(30)
        # 开局那次 settle 会把池子关掉（它必须关才能等静止），
        # 所以"这一 trial 要不要开着自发行为"要放在 settle 之后。
        pet.behaviors.enabled = trial % 2 == 0
        hist = []
        try:
            for _ in range(rng.randint(3, 8)):
                name = rng.choice(ops)
                hist.append(name)
                fn[name]()
                for _ in range(rng.randint(0, 30)):
                    pet.tick()
                    if pet.active_posture_busy() and pet.posture != config.BASE_POSTURE \
                            and pet.walker.busy:
                        # 断的是"走路状态机不许在跑"，不是绝对坐标不动 ——
                        # 改尺寸/回到右下角本来就会合法地重排窗口。
                        chain_bad.append(f"#{trial} 姿态期间走路没让路 "
                                         f"{pet.walker.phase} {'→'.join(hist)}")
                if not (desktop.left() <= pet.x() <= desktop.right() + 1 - pet.width()
                        and desktop.top() <= pet.y() <= desktop.bottom() + 1 - pet.height()):
                    chain_bad.append(f"#{trial} 越界 {'→'.join(hist)} @({pet.x()},{pet.y()})")
                    break
                if pet.idle.running and pet.facing.value != config.IDLE_FACING:
                    chain_bad.append(f"#{trial} 非正面播待机 {'→'.join(hist)}")
                    break
            else:
                if not settle() and not pet.follower.enabled:
                    chain_bad.append(f"#{trial} 收不了尾 {'→'.join(hist)}")
        except Exception as e:                       # noqa: BLE001
            chain_bad.append(f"#{trial} 抛异常 {'→'.join(hist)} {type(e).__name__}: {e}")
    pet.behaviors.enabled = False
    check("随机长链 300 条（含自发行为）不越界、收得了尾、不抛异常",
          not chain_bad, f"{len(chain_bad)} 条：" + "; ".join(chain_bad[:3])[:300])


def action_regression_tests(m):
    from contextlib import contextmanager

    store = AssetStore()

    @contextmanager
    def fresh():
        pet = PetWindow(m, store)
        pet.timer.stop()
        pet.behaviors.enabled = False
        pet.move(pet.screen_left() + 80, pet.screen_top() + 80)
        drawn = []
        render = pet.show_anim_frame

        def record(name, index):
            drawn.append((name, index))
            render(name, index)

        pet.show_anim_frame = record
        try:
            yield pet, drawn
        finally:
            pet.close()

    def advance(pet, condition, limit=60):
        for _ in range(limit):
            if condition():
                return True
            pet.tick()
        return bool(condition())

    def select(pet, group, label):
        menu = pet.build_menu()
        sub = next(a.menu() for a in menu.actions() if a.text() == group)
        next(a for a in sub.actions() if a.text() == label).trigger()
        del menu
        pump(1)

    for label, group, anims in (
        ("抱抱", "姿态", {"hug"}),
        ("向左走", "移动", {"walk_left"}),
        ("向右走", "移动", {"walk_right"}),
        ("随便走走", "移动", {"walk_left", "walk_right"}),
        ("跑向远处", "移动", {"run_left", "run_right"}),
    ):
        with fresh() as (pet, drawn):
            pet.request_posture("sitting")
            advance(pet, lambda: pet.holders["sitting"].phase == "holding")
            pet.holders["sitting"].left = config.ticks(60)
            drawn.clear()
            select(pet, group, label)
            played = advance(pet, lambda: any(n in anims for n, _ in drawn))
            check(f"真实菜单：坐下驻留中切换{label}会执行", played,
                  f"posture={pet.posture} pending={pet.wants.key}")

    with fresh() as (pet, drawn):
        pet.facing.apply(1)
        pet.oneshot.start()
        pet.oneshot.stop()
        drawn.clear()
        for _ in range(20):
            pet.tick()
        check("取消转正中的抱抱后，不会被旧回调重新播放",
              not pet.oneshot.busy and not any(n == "hug" for n, _ in drawn))

    with fresh() as (pet, drawn):
        pet.facing.apply(1)
        pet.request_posture("sitting")
        left = pet.holders["sitting"].left
        drawn.clear()
        pet.tick()
        check("坐下前转正时只绘制转身帧，不提前消耗驻留",
              bool(drawn) and all(n == "turn" for n, _ in drawn)
              and pet.holders["sitting"].left == left, str(drawn))
        drawn.clear()
        pet.set_size(180)
        check("转正途中缩放仍重绘当前转身帧",
              bool(drawn) and all(n == "turn" for n, _ in drawn), str(drawn))
        pet.facing.turn_to(3)
        check("坐下的转正回调被单击替换后仍能进入坐姿",
              advance(pet, lambda: pet.posture == "sitting"))

    with fresh() as (pet, drawn):
        pet.set_side_prone(True)
        pet.request_posture("prone")
        pet.facing.cycle()
        drawn.clear()
        pet.tick()
        check("趴着换朝向时，驻留帧不覆盖转身帧",
              bool(drawn) and all(n == "prone_turn" for n, _ in drawn), str(drawn))
        advance(pet, lambda: not pet.facing.turning)
        pet.facing.cycle()
        seq = pet.facing.director.seq
        check("单击右趴到左趴会倒放至目标关键帧",
              len(seq) > 1 and seq[-1] == m.anim("prone_turn")["keys"][1], str(seq))

    with fresh() as (pet, drawn):
        x, y = pet.x(), pet.y()
        pet.walker.start_to(x + 300, y + 80, gait="walk", keep_facing=True)
        pet.walker.retarget(x + 300, y + 80, gait="run")
        check("跟随由走切跑时，纵向步长使用新速度",
              abs(pet.walker.step_y - 80 / 60) < 1e-9, str(pet.walker.step_y))
        pet.walker.retarget(x, y + 150)
        for _ in range(20):
            pet.tick()
        check("光标改到正上或正下方后不再追旧终点",
              pet.x() == x and pet.y() == y and not pet.walker.busy,
              f"位置 {(pet.x(), pet.y())}，起点 {(x, y)}")

    with fresh() as (pet, drawn):
        goal = [pet.x() + 350, pet.y() + pet.height() // 2]
        pet.cursor_pos = lambda: tuple(goal)
        pet.set_follow(True)
        advance(pet, lambda: pet.walker.phase == "walking")
        x, y = pet.x(), pet.y()
        goal[:] = [x + pet.width() // 2, y + pet.height() // 2]
        pet.tick()
        check("光标进入死区的当前 tick 就停止移动", (pet.x(), pet.y()) == (x, y))

    with fresh() as (pet, drawn):
        pet.set_side_prone(True)
        pet.crawler.start()
        advance(pet, lambda: pet.walker.phase == "walking")
        x, y = pet.x(), pet.y()
        pet.cursor_pos = lambda: (x + pet.width() // 2, y + pet.height() // 2)
        pet.set_follow(True)
        for _ in range(20):
            pet.tick()
        check("爬走中开启跟随后旧爬行目标被取消",
              (pet.x(), pet.y()) == (x, y) and not pet.crawler.busy
              and not pet.walker.busy, f"位置 {(pet.x(), pet.y())}，起点 {(x, y)}")

    with fresh() as (pet, drawn):
        pet.request_posture("prone")
        rejected = not pet.request_posture("sleeping")
        pet.tick()
        check("关闭左右趴着时的睡觉请求不打断当前动作",
              rejected and pet.posture == "prone" and pet.wants.key is None)

    for instant in (False, True):
        with fresh() as (pet, drawn):
            if instant:
                pet.facing.director = InstantTurn()
            x, y = pet.x(), pet.y()
            pet.walker.start(1, seconds=5)
            select(pet, "姿态", "抱抱")
            played = advance(pet, lambda: any(n == "hug" for n, _ in drawn), limit=80)
            check(f"起步首帧取消后不会复活旧行程，抱抱能执行（instant={instant}）",
                  played and (pet.x(), pet.y()) == (x, y) and not pet.walker.busy,
                  f"played={played} walker={pet.walker.phase} x={pet.x()}")

    with fresh() as (pet, drawn):
        pet.oneshot.start()
        for _ in range(5):
            pet.tick()
        pet.facing.cycle()
        drawn.clear()
        pet.tick()
        check("抱抱播放中单击转身，抱抱让出绘制并结束",
              not pet.oneshot.busy and bool(drawn)
              and all(n == "turn" for n, _ in drawn), str(drawn))
        for _ in range(100):
            pet.tick()
        pose = pet.current_pose()
        check("转身后最终画面与当前朝向一致，不停在抱抱末帧",
              pet.label.pixmap().cacheKey() == pet.store.pix(pose.path, pet.pet_h, pose.k).cacheKey())


def main() -> int:
    m = Manifest()
    action_regression_tests(m)
    store = AssetStore()
    pet = PetWindow(m, store)
    pet.place_default()
    pet.show()
    # 自发行为默认关掉：否则"单击循环""走完转回原朝向"这类测试跑到一半，
    # 池子会自己插一段漫游进来改朝向和位置，测的就不是被测对象了。
    # 需要它的小节（下面"全屏漫游与行为池"）自己打开。
    pet.behaviors.enabled = False
    wait(200)

    # ── 素材清单 ────────────────────────────────────────────────────────
    poses = m.skin.poses
    check("清单含 4 个朝向姿势", set(poses) >= set(FACES), str(sorted(poses)))
    check("所有姿势文件存在", all(p.path.is_file() for p in poses.values()))
    # 素材里真有的那几张静姿图是**逐帧归一**到 CHAR_H 的，必须精确；
    # 从视频帧复制出来的那几张（打瞌睡、左右躺睡）整段共用一个缩放：
    # 首帧可以比中位高几像素，横躺的两张还要受画布宽度封顶（内容高只有 51~55% char_h，
    # 那是"躺着的身体本来就更矮更宽"，不是基准漂移）。它们由下面两条按
    # "与源帧逐字节相同 + 不超画布"单独锁。
    derived = {"doze_front", "sleep_left", "sleep_right"}   # 清单里带 derived_from 的那几张
    check("手工静姿图的内容高精确归一到 char_h",
          all(p.content[1] == m.canvas.char_h for k, p in poses.items()
              if k not in derived),
          str({k: v.content for k, v in poses.items() if k not in derived}))
    check("内容宽不超画布", all(p.content[0] <= m.canvas.w for p in poses.values()))
    check("转身动画已入库", "turn" in m.skin.anims and m.skin.anims["turn"]["frames"] == 121,
          str({k: v for k, v in m.skin.anims["turn"].items() if k != "path"}))
    idle = m.skin.anims.get(config.IDLE_ANIM) or {}
    check("静立动画已入库且走 sequence 放置",
          idle.get("frames") == 121 and idle.get("mode") == "sequence",
          str({k: v for k, v in idle.items() if k != "path"}))
    check("转身仍走 per_frame 归一（原地转身身高不该变）",
          m.skin.anims["turn"].get("mode") == "per_frame",
          str(m.skin.anims["turn"].get("mode")))

    # ── 水印涂白必须被泛洪吃掉（配套错误的哨兵）──────────────────────────
    # 泛洪判据换成"离本帧背景色多近"之后，scrub 里写死的纯白 255 离背景 246 差 9 级，
    # 落在容差之外 → 涂过的那块反而变成不透明的白矩形留在成品里。
    # 这类 bug 的形态是"两个各自正确的模块配套错了"，单看任何一个模块都看不出来。
    # 阈值按实测放：涂错色时 转身 2417px / 坐下 4633px；
    # 而 walk4_right 那 47px 是"水印笔画被毛发包住"的合法残留（两种填法都一样，
    # 宁留淡痕不打穿主体，见 build_assets.scrub_watermark），所以门限放在 300。
    leak = {k: v["scrub_leak"] for k, v in m.skin.anims.items() if v.get("scrub_leak")}
    check("涂掉的水印像素基本都被当背景吃掉了",
          all(v <= 300 for v in leak.values()),
          f"残留 {leak}（>300 就是涂的色和泛洪认的色脱钩了）")

    # ── 转身图的姿势顺序必须与姿态 faces 同序 ────────────────────────────
    # 运行时是拿 `facing.value` 去索引 keys 的，而 facing.value 就是 faces 的下标。
    # 两张表顺序不一致不会报错，只会转错方向 —— 趴着的转身图原本按视频时间顺序
    # (左→正→右) 存 keys，于是"向左看"取到 keys[1] = 正面趴，她面朝镜头往左爬。
    for pk, pv in m.postures.items():
        tname = pv.get("turn")
        t = m.anim(tname) if tname else None
        if t and t.get("faces"):
            check(f"{pk} 的转身图与 faces 同序", t["faces"] == list(pv["faces"]),
                  f"转身图 {t['faces']} vs 姿态 {pv['faces']}")

    # ── 趴着 / 打瞌睡 / 趴着睡 的分工要在清单里锁死 ───────────────────────
    # 用户澄清过两次："趴着睡是完全趴着（侧躺闭眼），左右那个趴着是趴坐着"，
    # 以及"是不是叫打瞌睡啊"。上一版把 正面趴着.mp4 当成趴睡挂到了 prone 的 idle 上，
    # 于是"正面看是睡、一转侧面就坐起来"，而真正的侧躺睡素材根本没接。
    # 现在的分工：prone 的 idle 是**正面那一段趴坐微动**，dozing 是坐着打盹，
    # sleeping 是侧躺睡（左右两套身体，各配各的 idle）。
    check("姿态注册表：站/坐/趴/打瞌睡/趴睡 五个",
          set(m.postures) == {"standing", "sitting", "prone", "dozing", "sleeping"},
          str(sorted(m.postures)))
    # `gated` 是管线留给审计的溯源字段，运行时不读（开关的落点在 stage.posture_faces），
    # 所以这里直接从原始 JSON 读，不给 Manifest 加一个没人用的属性。
    gated = json.loads((m.root / "manifest.json").read_text(encoding="utf-8"))["gated"]
    check("趴姿的驻留微动是按朝向对齐的列表：只有正面那段会动，两侧是静帧",
          m.posture_idle("prone") == ["prone_idle", None, None]
          and m.posture_enter("prone") is None,
          f"idle={m.posture_idle('prone')} enter={m.posture_enter('prone')}")
    check("打瞌睡是正面单朝向 + 一段来回微动，不受「左右趴着」开关管",
          list(m.postures["dozing"]["faces"]) == ["doze_front"]
          and m.posture_idle("dozing") == "doze"
          and "dozing" not in gated and m.postures["dozing"].get("turn") is None,
          f"faces={m.postures['dozing']['faces']} idle={m.posture_idle('dozing')} "
          f"gated={gated}")
    check("趴睡是左右两套身体，各配各的驻留素材，且受「左右趴着」开关管",
          list(m.postures["sleeping"]["faces"]) == ["sleep_left", "sleep_right"]
          and m.posture_idle("sleeping") == ["sleep_left", "sleep_right"]
          and m.postures["sleeping"].get("turn") is None
          and {"sleep_left", "sleep_right"} <= set(gated),
          f"faces={m.postures['sleeping']['faces']} idle={m.posture_idle('sleeping')} "
          f"gated={gated}")
    # 每个姿态的 idle 列表长度必须和 faces 一样：运行时是拿 facing.value 去索引的，
    # 短一项就会在切到那一侧时查不到素材而退回静帧（看起来"睡着了但不动"）。
    bad_idle = {k: (v.get("idle"), v["faces"]) for k, v in m.postures.items()
                if isinstance(v.get("idle"), list) and len(v["idle"]) != len(v["faces"])}
    check("所有姿态的 idle 列表与 faces 一一对应", not bad_idle, str(bad_idle))
    loops = {k: (m.anim(k) or {}).get("loop") for k in
             ("prone_idle", "doze", "sleep_left", "sleep_right")}
    check("四段驻留素材都标定过来回窗口", all(v for v in loops.values()), str(loops))
    # 派生静姿图是管线从素材第 0 帧**复制**出来的：另出一张图就会和动画对不齐，
    # 所以"必须逐字节相同"才是这条派生关系的正确断言。
    raw = json.loads((m.root / "manifest.json").read_text(encoding="utf-8"))
    dp = raw["skins"]["default"]["poses"]
    DERIVED = {"doze_front": "doze", "sleep_left": "sleep_left",
               "sleep_right": "sleep_right"}
    for key, anim_name in DERIVED.items():
        p = m.skin.poses.get(key)
        fr = m.root / m.anim(anim_name)["dir"] / "frame_000.png"
        check(f"{key} 静姿图就是 {anim_name} 素材第 0 帧（同一张图，不是另画的）",
              bool(p) and p.path.is_file() and fr.is_file()
              and p.path.read_bytes() == fr.read_bytes()
              and dp[key].get("derived_from") == f"{anim_name}#0",
              f"{p.path if p else '缺 ' + key} derived_from={dp[key].get('derived_from')}")
    check("派生静姿图不超画布（横躺的受宽度封顶，所以内容高比 char_h 矮是设计如此）",
          all(m.skin.poses[k].content[0] <= m.canvas.w
              and m.skin.poses[k].content[1] <= m.canvas.h - m.canvas.baseline
              for k in DERIVED),
          str({k: m.skin.poses[k].content for k in DERIVED}))

    # ── 源素材裁切回归锁：角色被源画面切掉的动画必须点名 ──────────────────
    # 向左走_四足 从第 72 帧起把她的脸切在画面外（源视频自己平移出去的，像素不存在，
    # 管线修不了）。这条锁不是要求"零裁切"，而是要求**裁切的名单必须与已知一致**：
    # 冒出新的坏素材会红；素材重做好了也会红（提醒把名字从名单里删掉）。
    KNOWN_CLIPPED = {"walk4_left"}
    clipped = {k: (v.get("src_clipped"), v.get("clip_at"))
               for k, v in m.skin.anims.items() if v.get("src_clipped")}
    if all("src_clipped" not in v for v in m.skin.anims.values()):
        check("源素材裁切检测已生效", False,
              "清单里没有 src_clipped 字段——重跑 tools/build_assets.py --only anims")
    else:
        check("源素材裁切名单与已知一致", set(clipped) == KNOWN_CLIPPED,
              f"当前裁切 {clipped}；已知待重做 {KNOWN_CLIPPED}")

    # ── 抠图不许把身体切开：孤岛面积与帧数上限 ──────────────────────────
    # 试过一条判据（拿暗描边当墙再泛洪一次）去收她耳侧的白色光晕，**所有既有指标
    # 全绿**——面积正常、包围盒不触边、对齐通过——但 walk4_left 约 1/3 帧的胸口与
    # 前腿整块没了。唯一当场抓住它的是"与主体不相连的块"：0 块 → 15 块。
    # 阈值不是拍脑袋：现网实测 sitdown 因 §11.12 那圈光晕有 7 帧孤岛、最大 1393px，
    # 所以 8 帧 / 2000px 是"放过已知缺陷、抓住新缺陷"的位置。
    from scipy import ndimage as _nd
    import numpy as np
    from PIL import Image as _Image
    for name, anim in m.skin.anims.items():
        d = m.root / anim["dir"]
        big, peak = 0, 0
        for i in range(0, int(anim["frames"]), 3):
            a = np.asarray(_Image.open(d / f"frame_{i:03d}.png").convert("RGBA"))[..., 3] > 8
            lab, k = _nd.label(a)
            if k < 2:
                continue
            s = np.bincount(lab.ravel())
            for j in range(1, k + 1):
                if j != int(s[1:].argmax()) + 1 and s[j] > 400:
                    big += 1
                    peak = max(peak, int(s[j]))
        check(f"{name} 没有被抠开的身体碎块（≤8 块 / ≤2000px）",
              big <= 8 and peak <= 2000, f"{big} 块，最大 {peak}px")

    # sequence 模式存在的意义就是保留呼吸起伏，那就直接量输出帧，别只看标志位
    import numpy as np
    from PIL import Image as _Image
    d_idle = m.root / idle["dir"]
    hs = []
    for i in range(0, 121, 10):
        a = np.asarray(_Image.open(d_idle / f"frame_{i:03d}.png").convert("RGBA"))[..., 3]
        ys, _xs = np.where(a > 128)
        hs.append(int(ys.max() - ys.min() + 1))
    swing = 100 * (max(hs) - min(hs)) / (sum(hs) / len(hs))
    check("静立保留了呼吸起伏（没被逐帧归一压平）", 0.4 < swing < 3.0,
          f"高 {min(hs)}~{max(hs)}，起伏 {swing:.2f}%")

    # ── 跨来源对齐与截断（静帧 / 转身 / 静立 必须共用同一套空间基准）──────
    # 曾经 sequence 模式把"源 bbox 中心 × 缩放"当成画布绝对坐标传给 to_canvas，
    # 静立整体右移 40px：与转身帧交接时明显错位，而且顶到画布右缘把尾巴切掉一截。
    def frames_bbox(anim):
        d = m.root / anim["dir"]
        out = []
        for i in range(0, anim["frames"], 6):
            a = np.asarray(_Image.open(d / f"frame_{i:03d}.png").convert("RGBA"))[..., 3]
            ys, xs = np.where(a > 128)
            out.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
        return out

    pf = frames_bbox(m.skin.anims["turn"])
    # 参考中心从**成品 front 静帧实测**，不再写死数字 —— 画布从 768 加宽到 896 时，
    # 写死的 383.5 让三条对齐断言集体误报"错位"，而素材其实是对齐的。
    _a = np.asarray(_Image.open(m.skin.poses["front"].path).convert("RGBA"))[..., 3]
    _xs = np.where(_a > 128)[1]
    ref_c = (_xs.min() + _xs.max()) / 2
    ref_bottom = m.canvas.h - m.canvas.baseline - 1   # 内容最后一行的索引（744 是第一个空行）
    for name in ("turn", config.IDLE_ANIM):
        boxes = frames_bbox(m.skin.anims[name])
        mean_c = sum((b[0] + b[2]) / 2 for b in boxes) / len(boxes)
        check(f"{name} 无任一帧触到画布边缘（防截断）",
              all(b[0] > 0 and b[1] > 0 and b[2] < m.canvas.w - 1 and b[3] < m.canvas.h - 1
                  for b in boxes),
              f"x0min={min(b[0] for b in boxes)} x1max={max(b[2] for b in boxes)} "
              f"y1max={max(b[3] for b in boxes)}")
        check(f"{name} 平均中心与静帧对齐（±2px）", abs(mean_c - ref_c) <= 2.0,
              f"均值 {mean_c:.1f} vs 静帧 {ref_c:.1f}")
        check(f"{name} 底边压在基线上（±1px）",
              all(abs(b[3] - ref_bottom) <= 1 for b in boxes),
              f"底边 {min(b[3] for b in boxes)}~{max(b[3] for b in boxes)} vs {ref_bottom}")

    # 摇摆必须"存在但很小"：为 0 说明又被居中抵消了，过大说明锚点漂了
    idle_c = [(b[0] + b[2]) / 2 for b in frames_bbox(m.skin.anims[config.IDLE_ANIM])]
    sway = (max(idle_c) - min(idle_c)) / 2
    check("静立保留了左右摇摆（没被重新居中抵消）", 1.0 < sway < 12.0, f"摆幅 ±{sway:.1f}px")

    # "任何素材帧不许顶到画布边缘"这条以前只对 turn 与 idle 两条成立，
    # 于是横躺的睡姿按身高归一撑到 995~1094px 宽、头和尾各被切掉一截也没人报 ——
    # 管线只在 stderr 喊一行，清单照写，派生出来的静姿图包围盒是 (0,85,895,743)。
    # 现在补成覆盖**全部**动画的锁（判据就是管线新增的 CONTENT_MAX_W 宽度封顶）。
    touched = {}
    for name, anim in m.skin.anims.items():
        for b in frames_bbox(anim):
            if b[0] <= 0 or b[2] >= m.canvas.w - 1:
                touched[name] = b
                break
    check("任何一段动画都没有一帧贴到画布左右缘（宽度封顶生效）",
          not touched, f"触边 {touched}；画布宽 {m.canvas.w}")

    # ── 运行时抖动锁：按**显示序列**量头部，不按原始逐帧 ────────────────────
    # 这条锁存在的理由就是一次真实的漏判：旧版 jitter_check 量"逐帧平均摆幅"，
    # 向右跑报 0.15px"看不见"，而用户仍然看到抖 —— 因为运行时每 2 帧取 1 帧，
    # 而且平均会把一记 3px 的尖峰摊平。判据换成"头部带的最大单帧跳"才抓得住。
    # 阈值 2.2 屏幕 px：向右跑实测 1.37（纵向去抖前 3.41），向左跑 1.07；
    # 摘掉 build_assets 的 BOB_ANIMS 就会红。转身类不在此列（头本来就要扫过半个画面）。
    # 取样序列直接复用 jitter_check 的 `shown_seq` —— 播放模型只准有一处实现，
    # 抄第二份迟早会和状态机漂移（本仓库已经为这类事记过三次）。
    from jitter_check import shown_seq as rt_seq
    def head_of(path):
        a = np.asarray(_Image.open(path).convert("RGBA"))[..., 3] > 40
        ys, xs = np.where(a)
        y0, y1 = int(ys.min()), int(ys.max())
        band = a[:y0 + max(1, int((y1 - y0 + 1) * 0.35))]
        by, bx = np.where(band)
        return (float(bx.mean()) if len(bx) else float(xs.mean()),
                float(by.mean()) if len(by) else float(ys.min()))
    sc = config.DEFAULT_H / m.canvas.h
    jitters = {}
    for name in ("walk_left", "walk_right", "run_left", "run_right",
                 "walk4_right", "idle_front", "hug", "doze"):
        a = m.skin.anims.get(name)
        if not a:
            continue
        d = m.root / a["dir"]
        pts = [head_of(d / f"frame_{i:03d}.png") for i in rt_seq(name, a)]
        jitters[name] = max(max(abs(pts[k+1][0]-pts[k][0]), abs(pts[k+1][1]-pts[k][1]))
                            * sc for k in range(len(pts) - 1))
    over = {k: round(v, 2) for k, v in jitters.items() if v > 2.2}
    check("位移/驻留类动画的头跳都在可见阈值内（纵向去抖生效）",
          not over, f"超标 {over}；当前 " + str({k: round(v, 2) for k, v in jitters.items()}))

    turn_unit_tests(m)
    idle_unit_tests()
    behavior_unit_tests()

    # ── 朝向状态机（走真实窗口 + 真实 tick）─────────────────────────────
    check("默认调度器是视频转身", type(pet.facing.director).__name__ == "VideoTurn",
          type(pet.facing.director).__name__)
    for target in (1, 2, 3, 0):
        pet.facing.turn_to(target)
        pump(1200)
        want = m.faces[target]
        got = Path(pet.current_pose().path).stem
        check(f"turn_to({target}) 落到 {want}", pet.facing.value == target and got == want,
              f"value={pet.facing.value} pose={got}")
    check("转身后无残留序列", not pet.facing.turning)

    pet.facing.turn_to(2)
    wait(200)
    check("转身中途画面切到了视频帧",
          pet.facing.turning and pet.facing.director.shown is not None,
          f"shown={pet.facing.director.shown}")
    pet.facing.spin()
    wait(300)
    check("转圈中且仍在跑", pet.facing.turning)
    finished = wait_until(lambda: not pet.facing.turning)
    check("转圈结束落到在飞的目标朝向且不再忙", finished and pet.facing.value == 2,
          f"value={pet.facing.value}")

    # 游标现在跟随实际朝向（从菜单直选过之后再单击要接着当前朝向往下走），
    # 所以期望序列得按起点算，不能写死 [3,2,1,0]。
    start = pet.facing.value
    order = config.CLICK_CYCLE
    pos = order.index(start)
    want = [order[(pos + k) % 4] for k in (1, 2, 3, 0)]
    got = []
    for _ in range(4):
        pet.facing.cycle()
        pump(1000)
        got.append(pet.facing.value)
    check(f"单击循环从 {start} 起沿视频正向推进", got == want, f"got={got} want={want}")

    # ── 待机微动 ────────────────────────────────────────────────────────
    # 一律用 wait_until 而不是定长 wait：tick 周期 80ms，定长等待不保证真跑到
    # 某一帧，早先 wait(90) 就偶发拿到 None 把断言炸掉。
    pet.facing.apply(config.IDLE_FACING)
    pet.idle.reset()
    entered = wait_until(lambda: pet.idle.shown is not None, 1500)
    check("正面静止时待机在跑，且重新入场从第 0 帧起",
          entered and pet.idle.running and pet.idle.shown == 0,
          f"shown={pet.idle.shown}")
    first = pet.idle.shown
    wait_until(lambda: pet.idle.shown != first, 1500)
    check("待机逐帧推进", pet.idle.shown != first, f"{first} -> {pet.idle.shown}")
    seen = set()
    for _ in range(24):
        wait(90)
        seen.add(pet.idle.shown)
    check("待机在循环且帧号不越界",
          all(s is not None and 0 <= s < 121 for s in seen) and len(seen) > 5,
          f"{len(seen)} 个不同帧")

    pet.facing.turn_to(2)
    wait(150)
    check("转身期间待机完全停手", pet.facing.turning and not pet.idle.running,
          f"turning={pet.facing.turning} idle.running={pet.idle.running}")
    pump(1200)
    pet.facing.apply(config.IDLE_FACING)
    wait(700)                                               # 跑约 8 帧，才看得出"离场前在哪"
    far = pet.idle.shown
    pet.facing.turn_to(2)                                   # 转身离场
    pump(1200)
    pet.idle.reset()
    pet.facing.apply(config.IDLE_FACING)
    reentered = wait_until(lambda: pet.idle.shown is not None, 1500)
    check("离场后再回正面，待机归零重播而不是接着上次",
          reentered and pet.idle.shown == 0 and (far or 0) > 0,
          f"离场前 shown={far}，重入后 shown={pet.idle.shown}")

    pet.facing.turn_to(1)
    pump(1200)
    check("非正面朝向不放待机（只有正面有静立素材）",
          pet.facing.value != config.IDLE_FACING and not pet.idle.running,
          f"facing={pet.facing.value}")
    pet.facing.apply(config.IDLE_FACING)
    wait(300)

    # ── 走路 ────────────────────────────────────────────────────────────
    check("两段走路动画已入库且走 sequence",
          all(m.skin.anims.get(k, {}).get("mode") == "sequence"
              for k in ("walk_left", "walk_right")),
          str({k: m.skin.anims[k].get("mode") for k in ("walk_left", "walk_right")})
          if "walk_left" in m.skin.anims else "缺动画")

    # ── 跑步素材与步态相位（清单层，不跑代码只看标定结果）────────────────
    check("两段跑步动画已入库、走 sequence 且两侧齐",
          all(m.skin.anims.get(k, {}).get("mode") == "sequence"
              for k in ("run_left", "run_right"))
          and m.has_gait("standing", "run") and not m.has_gait("prone", "run"),
          str({k: m.skin.anims.get(k, {}).get("mode") for k in ("run_left", "run_right")}))
    st_a = m.skin.anims.get("sitdown", {})
    # settle_at 退化成 0 是这次差点漏掉的事故：抠图判据一换，内容高曲线的真形状露出来，
    # "全段最低点"落在了还没开始坐的 f0 —— 那等于没有过渡动画，她会直接瞬移进坐姿。
    check("坐下的过渡→驻留交接点是素材中段一个真实存在的下坐点（不是 0）",
          8 <= int(st_a.get("settle_at", 0)) < int(st_a.get("frames", 1)) - 8,
          f"settle_at={st_a.get('settle_at')} / {st_a.get('frames')} 帧")
    check("坐下的驻留窗口在分界之后且有长度",
          st_a.get("loop", [0, 1])[0] == int(st_a.get("settle_at", -1))
          and st_a["loop"][1] > st_a["loop"][0],
          f"loop={st_a.get('loop')}")
    # 上一版栽在"窗口够静但只有 10 帧"：来回一轮 1.44s，比呼吸快一倍，观感还是抖。
    # 单帧窗口是管线找不到安静段时的合法退路（运行时就地停住），所以只卡"要么停住要么够长"。
    _w = (st_a["loop"][1] - st_a["loop"][0]) * 2 - 2
    check("坐下的待机窗口够长（来回一轮 ≥2 秒）或是单帧停住",
          st_a["loop"][1] - st_a["loop"][0] <= 1 or _w * config.FPS_MS / 1000 >= 2.0,
          f"窗口 {st_a['loop']}，来回一轮 {_w * config.FPS_MS / 1000:.2f}s")
    check("坐下有标定的进入起点，且落在过渡段之内",
          0 < int(st_a.get("enter_at", 0)) < st_a["settle_at"],
          f"enter_at={st_a.get('enter_at')}（素材开头她只是站着，还比标准站姿图大 3.3%）")
    # 步态循环的两条不变式：① 长度必须是整数个周期（否则每循环一次相位错一次，
    # 观感是"走两步绊一下"）；② 截完之后**窗口首尾那一接**必须接得上。
    # ②是这次新加的：只截长度不够 —— 起点挪一帧，接的就是完全不同的一对帧。
    # 四足走整段播完再接头是相邻帧差的 2.8~4.8 倍，挑过起点才降下来。
    seam_ok = 2.5
    UNALIGNED = {"walk4_left"}       # 见 §13.9：这条素材根本没有步态周期，量不出峰
    for k in ("walk_left", "walk_right", "walk4_left", "walk4_right",
              "run_left", "run_right"):
        g = (m.skin.anims.get(k) or {}).get("gait") or {}
        p = g.get("period") or 0
        lo, hi = g.get("loop") or [0, 1]
        check(f"{k} 的循环落在整数个步态周期上",
              k in UNALIGNED or not g or (p and (hi - lo) % p == 0) or hi - lo == int(
                  m.skin.anims[k]["frames"]),
              f"周期{p} 窗口[{lo},{hi})")
        check(f"{k} 的循环首尾接得上（≤{seam_ok}×相邻帧差）",
              k in UNALIGNED or not g or g.get("loop_seam", g.get("seam", 0)) <= seam_ok,
              f"窗口接缝 {g.get('loop_seam', g.get('seam'))}× 整段接缝 {g.get('seam')}×")
        ph0 = g.get("phase0", 0)
        check(f"{k} 的起步相位落在循环窗口内", lo <= ph0 < hi, f"phase0={ph0} 窗口[{lo},{hi})")

    # 擦除回归锁：向右走那截多出来的"左手"以前只擦了相位 22/23/0/1，
    # 两头（19~21、2）留着，而且羽化在裁剪线外留了一圈半透明青边。
    # 这里直接量落盘帧：凸起活动区里，超出"整段中位形状 + 3px"的像素必须清零。
    from scipy import ndimage as _nd
    wr = m.skin.anims.get("walk_right") or {}
    if wr:
        d = m.root / wr["dir"]
        al = np.stack([np.asarray(_Image.open(d / f"frame_{i:03d}.png").convert("RGBA"))[..., 3]
                       for i in range(wr["frames"])])
        keep = _nd.binary_dilation((al > 128).mean(0) > 0.5, iterations=3)
        x0, y0, x1, y1 = 640, 495, 720, 615
        zone = np.zeros(keep.shape, dtype=bool)
        zone[y0:y1, x0:x1] = True
        out = (al > 8) & (zone & ~keep)
        worst = int(out.reshape(out.shape[0], -1).sum(1).max())
        check("向右走凸起区无残留（含半透明裙边）", worst <= 20,
              f"单帧最大残留 {worst} px，擦除 {wr.get('erased_frames')} 帧")

    pet.facing.apply(0)
    pet.place_default()
    wait(200)
    x0, f0 = pet.x(), pet.facing.value
    started = pet.walker.start(-1, seconds=0.6)
    check("可以发起向左走", started and pet.walker.busy, f"phase={pet.walker.phase}")
    pump(400)
    check("走路前先转到左侧身", pet.facing.value == 1 or pet.facing.turning,
          f"facing={pet.facing.value}")
    # 走路进行中采样：这段时间待机必须完全不播。之前把断言放在走路结束之后，
    # 那时待机恢复才是正确行为，等于什么都没测。
    idle_during = set()
    for _ in range(10):
        wait(60)
        if pet.walker.busy:
            idle_during.add(pet.idle.running)
    check("走路进行中的每一帧待机都停手", idle_during == {False}, f"观测到 {idle_during}")
    pump(2600)
    check("向左走真的左移了", pet.x() < x0, f"{x0} -> {pet.x()}")
    check("走完转回出发前的朝向", pet.facing.value == f0 and not pet.walker.busy,
          f"facing={pet.facing.value} 应为 {f0}, phase={pet.walker.phase}")

    x1 = pet.x()
    pet.walker.start(1, seconds=0.6)
    pump(3000)
    check("向右走真的右移了", pet.x() > x1, f"{x1} -> {pet.x()}")

    # 撞屏幕左边必须就地停下，不许走出桌面
    desktop = QApplication.desktop().availableGeometry(pet)
    pet.move(desktop.left() + 6, pet.y())
    pet.walker.start(-1, seconds=3.0)
    pump(3600)
    check("走到屏幕左边会停下且不越界", pet.x() >= desktop.left() - 1 and not pet.walker.busy,
          f"x={pet.x()} 屏幕左={desktop.left()} phase={pet.walker.phase}")

    # 用户上手拖拽要立刻打断走路
    pet.place_default()
    pet.walker.start(-1, seconds=3.0)
    pump(900)
    pet.walker.stop()
    pump(1600)
    check("stop() 能中断走路并复位朝向", not pet.walker.busy, f"phase={pet.walker.phase}")

    # ── 全屏漫游与行为池（真实 PetWindow 上再验一遍，纯逻辑那层测不到渲染抢权）──
    pet.facing.apply(0)
    pet.place_default()
    wait(200)
    desktop = QApplication.desktop().availableGeometry(pet)
    x0, y0 = pet.x(), pet.y()
    started = pet.walker.roam()
    check("可以发起全屏漫游", started and pet.walker.busy, f"phase={pet.walker.phase}")
    pump(160)   # 先让交接那一 tick 跑完：idle 是在自己的下一次 tick 才复位的，
                # 而采样间隔 60ms < 80ms tick，不 pump 第一枪会抓到"交接前"那一瞬
    idle_during = set()
    for _ in range(40):
        wait(60)
        if pet.walker.busy:
            idle_during.add(pet.idle.running)
    check("漫游期间的每一帧待机都停手", idle_during == {False}, f"观测到 {idle_during}")
    ok = wait_until(lambda: not pet.walker.busy, timeout_ms=60000)
    check("漫游会在有限时间内走完", ok, f"phase={pet.walker.phase}")
    check("漫游真的挪到了别处", (pet.x(), pet.y()) != (x0, y0), f"({x0},{y0}) -> ({pet.x()},{pet.y()})")
    check("落点在屏幕可用区内",
          desktop.left() <= pet.x() and pet.x() + pet.width() <= desktop.right() + 1
          and desktop.top() <= pet.y() and pet.y() + pet.height() <= desktop.bottom() + 1,
          f"({pet.x()},{pet.y()}) 可用区 {desktop.width()}x{desktop.height()}")
    check("到达后转身面向镜头", pet.facing.value == config.IDLE_FACING,
          f"facing={pet.facing.value}")

    # 斜向段：真实窗口上验一次"两轴同时到达、落点精确"
    tx, ty = min(desktop.right() - pet.width() - 20, x0 + 260), max(desktop.top(), y0 - 90)
    pet.move(x0, y0)
    wait(150)
    pet.walker.start_to(tx, ty)
    # 请求的纵向会被斜率夹掉，所以断言的是**夹后**的目标点，不是原始请求
    gx, gy = pet.walker.tx, pet.walker.ty
    lim = min(abs(gx - x0) * config.WALK_SLOPE, config.ROAM_MAX_DY)
    assert abs(gy - y0) <= lim + 1, f"斜率闸门要成立：|{gy}-{y0}| ≤ {lim:.1f}"
    # 轮询而不是定长 pump：行程时长取决于转身帧数与真实节拍抖动，写死必赌错
    ok = wait_until(lambda: not pet.walker.busy, timeout_ms=30000)
    check("斜向走完精确落在（斜率夹后的）目标点上",
          ok and abs(pet.x() - gx) <= 1 and abs(pet.y() - gy) <= 1,
          f"({pet.x()},{pet.y()}) 落点=({gx},{gy}) 请求=({tx},{ty}) 走完={ok}")

    # 开关要两头都验：只验"关掉后不动"的话，默认关着的池子会白给这条断言。
    pet.behaviors.set_enabled(True)
    n0, x_before, f_before = pet.behaviors.fired, pet.x(), pet.facing.value
    pet.behaviors.next_in = 1          # 不等随机间隔，直接逼它下一 tick 挑一个
    ok = wait_until(lambda: pet.behaviors.fired > n0, timeout_ms=8000)
    check("打开开关后行为池会自己挑动作来做", ok,
          f"fired={pet.behaviors.fired} last={pet.behaviors.last}")
    check("挑中的动作确实归到某个执行者",
          pet.behaviors.last in pet.behaviors.actors(), str(pet.behaviors.last))
    check("配置里的动作与派发表一一对上（既无「配了没人做」也无「做了没配」）",
          set(config.BEHAVIOR_WEIGHTS) == set(pet.behaviors.actors()),
          f"只在权重里={sorted(set(config.BEHAVIOR_WEIGHTS) - set(pet.behaviors.actors()))} "
          f"只在派发表={sorted(set(pet.behaviors.actors()) - set(config.BEHAVIOR_WEIGHTS))}")
    # "有人接下了这一轮"必须问**全部**执行者。漏掉 oneshot 的后果实测过：
    # 池子挑中 hug 时，3 秒的一次性动作在 wait_until 回来时已经播完，
    # 位置也没变，于是这条断言把"做完了"误判成"没做"。
    check("执行者确实接下了这一轮",
          pet.walker.busy or pet.crawler.busy or pet.facing.turning
          or pet.active_posture_busy() or pet.oneshot.busy
          or (pet.facing.value, pet.x()) != (f_before, x_before),
          f"last={pet.behaviors.last} facing={pet.facing.value} x={x_before}->{pet.x()}")
    pet.walker.stop()
    pump(3000)
    n1 = pet.behaviors.fired
    pet.behaviors.set_enabled(False)
    pump(6000)
    check("关掉随机动作后桌宠不会自己乱动", pet.behaviors.fired == n1,
          f"{n1} -> {pet.behaviors.fired}")

    # ── 尺寸与画布对齐 ──────────────────────────────────────────────────
    pet.set_size(config.SIZE_MAX * 5)
    check("尺寸上限被 clamp", pet.pet_h == config.SIZE_MAX, str(pet.pet_h))
    pet.set_size(config.SIZE_MIN // 5)
    check("尺寸下限被 clamp", pet.pet_h == config.SIZE_MIN, str(pet.pet_h))
    pet.set_size(200)
    want_w = round(200 * m.canvas.w / m.canvas.h)
    check("窗口宽按画布比例派生、高=pet_h（画布是横向的，坐姿比站姿宽得多）",
          pet.height() == 200 and pet.width() == want_w,
          f"{pet.width()}x{pet.height()}，期望 {want_w}x200")
    # 这一节量的是"尺寸缩放等不等比"，前提是她身上那张图的内容高就是 char_h。
    # 上面几节把她留在了打瞌睡（内容高 676，来自视频帧而不是手工静姿），
    # 所以先站回正面再量 —— 不然这条断言测的是"她在哪个姿态"，不是缩放。
    pet.enter_posture(config.BASE_POSTURE)
    pet.facing.apply(0)
    rect = pet.content_rect()
    s = 200 / m.canvas.h
    check("内容底边压在基线上",
          abs(rect.bottom() - (200 - m.canvas.baseline * s)) < 0.5,
          f"bottom={rect.bottom():.2f} target={200 - m.canvas.baseline * s:.2f}")
    check("内容高=char_h 等比", abs(rect.height() - m.canvas.char_h * s) < 0.5,
          f"{rect.height():.2f} vs {m.canvas.char_h * s:.2f}")
    check("内容水平居中", abs((rect.x() + rect.width() / 2) - pet.pet_w / 2) < 0.5,
          f"{rect.x():.2f}")
    check("切换尺寸后 pixmap 非空", not pet.label.pixmap().isNull(),
          f"{pet.label.pixmap().width()}x{pet.label.pixmap().height()}")

    # ── 屏幕约束 ────────────────────────────────────────────────────────
    desktop = QApplication.desktop().availableGeometry(pet)
    pet.move(desktop.right() + 4000, desktop.bottom() + 4000)
    pet._clamp_to_screen()
    check("拖出屏幕右下角会被拉回",
          pet.frameGeometry().right() <= desktop.right() + 1
          and pet.frameGeometry().bottom() <= desktop.bottom() + 1,
          f"{pet.x()},{pet.y()}")
    pet.move(desktop.left() - 4000, desktop.top() - 4000)
    pet._clamp_to_screen()
    check("拖出屏幕左上角会被拉回",
          pet.x() >= desktop.left() and pet.y() >= desktop.top(), f"{pet.x()},{pet.y()}")

    # ── 变体回退（双形象/多皮肤的兜底）──────────────────────────────────
    check("未知变体回退基准素材",
          m.pose("back", variant="ghost").path.name == "back.png")
    check("缺失姿势回退 front", m.pose("nope").key == "front")

    # ── 菜单与尺寸滑条 ──────────────────────────────────────────────────
    # 菜单结构是按站姿断言的，而上面那次"行为池自己挑动作"可能正好挑中坐下/趴下，
    # 池子留下的姿态会把这一节的起点状态带跑（实测间歇性少一个"朝向"项）。
    stand_up(pet)
    menu = pet.build_menu()      # 必须持有引用，否则临时 QMenu 析构会连带删掉 QAction
    texts = [a.text() for a in menu.actions()]
    check("菜单结构齐全",
          any("朝向" in x for x in texts) and any("随机动作" in x for x in texts)
          and any("总在最前" in x for x in texts) and any("退出" in x for x in texts),
          str(texts))
    check("转一圈已撤下", "转一圈" not in texts, str(texts))
    faces = [a.text() for a in menu.actions()[0].menu().actions()]
    check("朝向子菜单 4 项", len(faces) == 4, str(faces))
    check("朝向回调只捕获 int，不捕获菜单拥有的 QAction",
          True, "见 menu_lifetime_test")
    menu_lifetime_test()

    acts = [a for a in menu.actions() if isinstance(a, QWidgetAction)]
    check("尺寸是菜单内嵌滑条而非子菜单",
          len(acts) == 1 and isinstance(acts[0].defaultWidget().findChild(QSlider), QSlider),
          f"QWidgetAction x{len(acts)}")
    slider = acts[0].defaultWidget().findChild(QSlider)
    value_lbl = [x for x in acts[0].defaultWidget().findChildren(QLabel)
                 if x.objectName() == "sizeValue"]
    check("滑条区间覆盖尺寸上下限",
          (slider.minimum(), slider.maximum()) == (config.SIZE_MIN, config.SIZE_MAX),
          f"{slider.minimum()}~{slider.maximum()}")

    # 拖滑条 → 防抖 → 真的改尺寸。这是"拖动即时预览"的核心链路，必须实测而不是只看控件在不在
    pet.set_size(config.DEFAULT_H)
    slider.setValue(240)
    check("滑条旁实时显示像素值", bool(value_lbl) and value_lbl[0].text() == "240 px",
          value_lbl[0].text() if value_lbl else "-")
    pump(220)
    check("拖滑条真的改了尺寸",
          pet.pet_h == 240 and pet.width() == round(240 * m.canvas.w / m.canvas.h),
          f"pet_h={pet.pet_h} w={pet.width()}")
    pet.set_size(config.DEFAULT_H)
    menu.setParent(None)
    menu.deleteLater()

    # ── 动作串联矩阵（放在最后：它会反复改尺寸和位置）──────────────────
    # 静默渲染下整段只多花约 2 秒，所以常开不分级。
    prone_real_tests(pet, m)
    interaction_matrix(pet)
    pet.behaviors.enabled = True
    pet.set_size(config.DEFAULT_H)
    pet.set_always_on_top(True)

    # ── 长跑不崩 ────────────────────────────────────────────────────────
    pet.place_default()
    wait(1200)
    check("连续 tick 1.2s 无崩溃", pet.isVisible())

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} 通过")
    pet.close()
    return 1 if failed else 0


if __name__ == "__main__":
    if "--menu-child" in sys.argv:
        sys.exit(menu_child())
    sys.exit(main())
