"""朝向状态机与转身调度。不依赖 Qt，可单独跑测试。

facing 四分位顺序取自 manifest.faces：0 正面 / 1 左侧 / 2 背面 / 3 右侧。
"""
from __future__ import annotations

from . import config


class TurnDirector:
    """转身策略接口。start 返回 False 表示目标朝向与当前一致、无需转身。

    hint：0 = 走最短路径（默认）；+1 / -1 = 强制沿视频正/反方向。
    连点换朝向必须给 +1，否则每一跳都重新取最短路径，方向会来回翻转。
    """

    def start(self, facing: "Facing", to: int, done=None, hint: int = 0) -> bool:
        raise NotImplementedError

    def tick(self, facing: "Facing") -> None:
        raise NotImplementedError

    def abort(self) -> None:
        """丢掉一切在飞状态与回调。窗口要析构了，回调再触发就是解引用死对象。"""

    @property
    def busy(self) -> bool:
        return False


class InstantTurn(TurnDirector):
    """v1 实现：下一帧直接切到目标静态姿势。

    v2 换成 VideoTurn（读 manifest.anims['turn'] + SPIN_KEY 关键帧表）时，
    调用方（菜单 / 单击 / 将来的行程机制）零改动——这是本模块唯一的扩展缝。
    """

    def __init__(self):
        self._target = None
        self._done = None

    def start(self, facing: "Facing", to: int, done=None, hint: int = 0) -> bool:
        to %= facing.count
        if to == facing.value:
            self.abort()
            facing.apply(to)
            if done:
                done()
            return False
        self._target, self._done = to, done
        return True

    def tick(self, facing: "Facing") -> None:
        if self._target is None:
            return
        target, done = self._target, self._done
        self._target = self._done = None
        facing.apply(target)
        if done:
            done()

    def abort(self) -> None:
        self._target = self._done = None

    @property
    def busy(self) -> bool:
        return self._target is not None


class VideoTurn(TurnDirector):
    """用原地 360° 转身视频的帧序列做朝向过渡。

    素材的三个实测特性决定了这里每一段的写法，数值来自
    `tools/build_assets.py:turn_geometry()`（标定过程见 docs/开发文档.md §9）：

    1. **转身不匀速**——一圈 cycle 帧里四个 90° 分别占 29/17/24/30 帧，
       所以不能把帧数按 4 等分当朝向，朝向→帧号写死在 manifest 的 `keys` 里。
    2. **尾部有冗余正面静止帧**——`cycle` 只取真正转完的那一段。若按整段视频长度
       循环，起手会先顿一下才动，就是菲比"转多了 / 有多余画面"的根因。
    3. **素材里有停滞区**——采样按**累计视觉路径**等分而不是按帧号等分，
       停滞区自然被跳过。菲比早期用"相邻帧号差"判重，均匀采样恰好跨过 3~4 帧的
       停滞区让重复帧漏网，改成路径距离才治住。
    """

    def __init__(self, cycle: int, keys: list, path: list, render,
                 frames_90: int = config.TURN_FRAMES_90,
                 tie: int = config.TURN_TIE_180,
                 front_ratio: float = config.TURN_FRONT_RATIO,
                 stride: int = config.SPIN_STRIDE):
        self.cycle = cycle
        self.keys = keys
        self.path = path
        self.render = render
        self.frames_90 = frames_90
        self.tie = tie
        self.front_ratio = front_ratio
        self.stride = stride
        self.seq: list = []
        self.i = 0
        self.target = None
        self.done = None
        self.shown = None          # 最后真正画出来的那一帧，供中途改尺寸后重绘

    def _show(self, idx: int) -> None:
        self.shown = idx
        self.render(idx)

    def _route(self, pos: int, tgt: int, fwd: bool) -> tuple:
        """走这条方向要经过多少帧，以及途中会不会路过正面。

        "途中"不含起点：从正面出发的任何一转都不该被算成"经过正面"，
        否则 front→back 这种两条路都从正面起步的转换会被误判成可以择优。
        """
        d = (tgt - pos) % self.cycle if fwd else (pos - tgt) % self.cycle
        rng = range(1, d + 1)
        path = {(pos + (s if fwd else -s)) % self.cycle for s in rng}
        return d, self.keys[0] in path

    # ── 采样 ────────────────────────────────────────────────────────────
    def _pos(self, facing_value: int) -> int:
        """当前在视频循环里的位置 = **屏幕上正在显示的那一帧**。

        不能用 `seq[self.i]`：`i` 在 tick 里已经自增，那是"下一帧"，
        拿它当改道起点会让新序列凭空跳过一帧，看起来就是转身上手一顿。
        """
        if self.seq and self.shown is not None:
            return self.shown
        return self.keys[facing_value % len(self.keys)]

    def _step(self, a: int, b: int) -> float:
        """相邻两帧之间的一步视觉变化。b 必须是 a 在行进方向上的下一帧。"""
        if (b - a) % self.cycle == 1:
            return self.path[a + 1] - self.path[a]
        prev = a if a else self.cycle        # a=0 的"上一步"是跨回环尾的那一段
        return self.path[prev] - self.path[prev - 1]

    def _sample(self, pos: int, dist: int, fwd: bool) -> list:
        """按**累计视觉路径**等分采样，不是按帧号等分。

        视频转身不匀速：back→side_left 这一段按帧号等分时，前 4 个采样帧几乎一模一样、
        视觉变化全挤在后半段，观感就是"顿了一下才开始转"。按路径等分之后，停滞区里
        的帧会因为累计进度没推进而自然撞车、被去重掉——不再需要单独的平台区阈值。

        首尾必达：rank=0 必落在 pos，rank=n-1 必落在目标关键帧，四舍五入不允许吃掉它们。
        """
        idxs = [(pos + s) % self.cycle for s in range(dist + 1)] if fwd             else [(pos - s) % self.cycle for s in range(dist + 1)]
        cum = [0.0]
        for a, b in zip(idxs, idxs[1:]):
            cum.append(cum[-1] + self._step(a, b))
        total = cum[-1]
        n = max(2, min(self.frames_90, dist + 1))
        out, seen = [], set()
        for rank in range(n):
            want = total * rank / (n - 1)
            j = min(range(len(idxs)), key=lambda i: abs(cum[i] - want))
            if j in seen:
                continue
            seen.add(j)
            out.append(idxs[j])
        return out

    # ── 可覆写钩子：环形 360° 与线性 180° 只差这两处 ──────────────────
    def _distances(self, pos: int, target_frame: int) -> tuple:
        """两个方向各要走多少帧。环形取模，所以往回走是"绕一大圈"的正数。"""
        return (target_frame - pos) % self.cycle, (pos - target_frame) % self.cycle

    def _choose(self, fwd: int, back: int) -> bool:
        """挑一个方向。默认走最短，并在两条路长度接近时优先路过正面。"""
        go_fwd = fwd <= back if abs(fwd - back) <= self.tie else fwd < back
        # 观感优先：能路过正面就别绕到背面去，观众要看的是她的脸。
        # 但只在"两条路长度接近"时才让路 —— 实测 左↔右 是 55 vs 43（比值 1.28），
        # 该改；而 背面→侧面 走正面要 74 帧、直接走只 24 帧（比值 3.08），
        # 那才是真的绕远，绝不能改。一刀切"永远走正面"会把后半圈转成 3 倍时长。
        return go_fwd

    def _prefer(self, pos: int, target_frame: int, go_fwd: bool) -> bool:
        """在两条路里改选"路过正面"的那条，前提是长度别差太多。"""
        d_f, thru_f = self._route(pos, target_frame, True)
        d_b, thru_b = self._route(pos, target_frame, False)
        if thru_f != thru_b:
            front_d, other_d = (d_f, d_b) if thru_f else (d_b, d_f)
            if front_d <= other_d * self.front_ratio:
                return thru_f
        return go_fwd

    # ── TurnDirector ────────────────────────────────────────────────────
    def start(self, facing: "Facing", to: int, done=None, hint: int = 0) -> bool:
        to %= len(self.keys)
        target_frame = self.keys[to]
        pos = self._pos(facing.value)
        fwd, back = self._distances(pos, target_frame)
        if fwd == 0 and back == 0:
            self.abort()
            facing.apply(to)
            if done:
                done()
            return False
        if hint > 0:
            go_fwd = True
        elif hint < 0:
            go_fwd = False
        else:
            # 接近 180° 时两边帧数差不多，固定走最短的一侧；随机选边会显得"方向飘忽"
            go_fwd = self._choose(fwd, back)
            go_fwd = self._prefer(pos, target_frame, go_fwd)
        seq = self._sample(pos, fwd if go_fwd else back, go_fwd)
        self.seq, self.i, self.target, self.done = seq, 1, to, done
        if seq:
            self._show(seq[0])         # 首帧就是当前角度，避免突跳；i 从 1 起不再多停一帧
        return True

    def spin(self, facing: "Facing", done=None) -> bool:
        """原地转一圈，末帧必达目标朝向的关键帧。

        转身在飞时接到转圈指令，目标沿用那一转的朝向——用户先点"背面"再点"转一圈"，
        他期待的是转完停在背面，而不是被就地就近停下或退回起点。
        """
        pos = self._pos(facing.value)
        target = (self.target if self.target is not None else facing.value) % len(self.keys)
        dist = (self.keys[target] - pos) % self.cycle
        if dist < self.cycle // 2:
            dist += self.cycle          # 保证真的转满一圈，而不是就近停住
        seq = [(pos + s) % self.cycle for s in range(0, dist, self.stride)]
        seq.append(self.keys[target])
        self.seq, self.i, self.target, self.done = seq, 1, target, done
        self._show(seq[0])
        return True

    def tick(self, facing: "Facing") -> None:
        if not self.seq:
            return
        if self.i < len(self.seq):
            self._show(self.seq[self.i])
            self.i += 1
            return
        target, done = self.target, self.done
        self.seq, self.i, self.target, self.done = [], 0, None, None
        facing.apply(target)           # 收尾落到标准姿势图，角度误差不带进下一个动作
        if done:
            done()

    def abort(self) -> None:
        self.seq, self.i, self.target, self.done = [], 0, None, None

    @property
    def busy(self) -> bool:
        return bool(self.seq)


class LinearTurn(VideoTurn):
    """线性 180° 弧的转身调度器，给 趴着转180度 用。

    那条素材是 左趴 → 正面趴 → 右趴 的**单向弧**：keys 单调递增（实测 [8,40,94]），
    末帧就是最右。它不是环形 360°，所以两处必须不同：

    1. **距离不能取模**。环形里"从右趴回左趴"是绕一大圈的正数，
       线性里它就是往回走 86 帧。取模会让她在原地多转一整圈。
    2. **"绕正面择优"没有意义**。站立时绕正面是为了让观众看到脸；
       趴着这条弧本来就经过正面趴，且只有三个姿势，没有"背面"可绕。

    其余（按累计视觉路径等分采样、首帧不突跳、中途改道从 shown 出发）与环形完全一致，
    所以直接继承。
    """

    def start(self, facing: "Facing", to: int, done=None, hint: int = 0) -> bool:
        return super().start(facing, to, done, hint=0)

    def _distances(self, pos: int, target_frame: int) -> tuple:
        d = target_frame - pos
        return (d if d > 0 else 0, -d if d < 0 else 0)

    def _choose(self, fwd: int, back: int) -> bool:
        return fwd > 0                       # 只有一条路可走：朝目标那一侧

    def _prefer(self, pos: int, target_frame: int, go_fwd: bool) -> bool:
        return go_fwd                        # 线性弧上没有"绕正面"这回事

    def spin(self, facing: "Facing", done=None) -> bool:
        return False                         # 180° 弧转不出一整圈


class Facing:
    def __init__(self, on_change, count: int = 4, director: TurnDirector | None = None):
        self.on_change = on_change
        self.count = count
        self.director = director or InstantTurn()
        self.value = 0
        self._cycle_pos = 0

    def set_posture(self, faces: list, director: TurnDirector) -> None:
        """换姿态：朝向集合与转身调度器一起换。

        姿态素材都从正面起步，所以统一落到 0。旧姿态的朝向数（站立 4 / 趴着 3 /
        坐着 1）和新姿态不一样，沿用旧索引会直接指到不存在的姿势图上。
        """
        self.count = max(1, len(faces))
        self.director = director
        self.value = 0
        self._cycle_pos = 0
        self.on_change(self.value)

    def abort(self) -> None:
        self.director.abort()

    def _seq(self) -> list:
        """当前姿态下单击能轮到的朝向序列。

        `CLICK_CYCLE` 是按站姿四个朝向写的（正→右→背→左）。
        趴着只有三个、坐着一个，直接套会出"死点击"：
        索引 3 被取模成 0，而她本来就在 0 —— 点一下什么也没发生。
        """
        return [i for i in config.CLICK_CYCLE if i < self.count]

    def apply(self, value: int) -> None:
        self.value = value % self.count
        # 同步循环游标：从菜单直选过朝向之后，下一次单击要接着当前朝向往下走，
        # 而不是从游标残留处跳回起点。
        order = self._seq()
        if self.value in order:
            self._cycle_pos = order.index(self.value)
        self.on_change(self.value)

    def turn_to(self, q: int, done=None, hint: int = 0) -> bool:
        return self.director.start(self, q, done, hint)

    def cycle(self) -> bool:
        """单击循环：正→右→背→左→正，**永远沿视频正向推进**。

        这个顺序本身就是视频的前进方向，所以强制同向。若让它每跳都重新取最短路径，
        连点四次会出现"转过去又转回来"——用户观感就是卡、就是怪。
        """
        order = self._seq()
        if len(order) < 2:
            return False            # 这个姿态没有"换个朝向"可换（坐着只有一个姿势）
        self._cycle_pos = (self._cycle_pos + 1) % len(order)
        return self.turn_to(order[self._cycle_pos], hint=config.CLICK_HINT)

    def spin(self, done=None) -> bool:
        """原地转圈。只有支持整段循环的调度器能做，InstantTurn 下静默忽略。"""
        spin = getattr(self.director, "spin", None)
        return bool(spin and spin(self, done))

    @property
    def turning(self) -> bool:
        return self.director.busy

    def tick(self) -> None:
        self.director.tick(self)
