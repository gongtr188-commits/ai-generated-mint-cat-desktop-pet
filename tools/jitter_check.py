"""抖动核查：**按运行时真正播出来的那一串帧**量，不是按原始逐帧序列。

用户反馈"有时候有的动画会抖动"。这句话有三种可能的病因，分别在三个地方：

  ① 素材自己抖     —— 生成视频里身体逐帧乱跳，管线只是忠实还原
  ② 管线放抖了     —— 每帧单独算摆位锚点/缩放，锚点被手脚尾巴带偏，身体就跟着晃
  ③ 运行时节拍撞车 —— stride 与步态周期不成整除、来回播的折点、或循环接缝

**为什么必须按显示序列量。** 这一版之前本工具量的是"原始逐帧"序列，于是
`微笑拥抱` 报"平均 0.10px，看不见"，而用户仍然说抖 —— 因为运行时是
**每 2 帧取 1 帧**（24fps 素材配 80ms 节拍），抽稀会把相邻帧的跳变放大；
平均数更会骗人：一记 2px 的尖峰摊进 60 个显示帧只剩 0.03px。
所以这里的主指标是 **最大单帧跳**，平均摆幅只当参考。

**判"抖"还要看方向反转。** 一条平滑的大摆动看起来是"走路"，
一段小的高频反转看起来就是"抖"。翻转率：
    运动方向逐帧反转的比例。正弦摆动一个周期只反转 2 次（约 0.1），
    纯噪声几乎每帧都反转（约 0.67）。
只有大跳、方向不反转的，是接缝或来回折返顿了一下 —— 那是 seam_check 的活。

（历史上这条指标错过两次：第一版用 `mean|Δ²| / mean|Δ|` 比值，十条动画全都
 1.0~1.6，一律判成"高频反转为主"，因为质心位移本身只有 0.4px，比值放大的是
 四舍五入噪声；第二版用逐帧平均摆幅，把单帧尖峰摊平了。**指标要能量出用户看见的
 东西，不然它比没量更糟。**）

跑法：`py tools/jitter_check.py`（有动画判成"抖"时退出码 1）
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "assets" / "manifest.json").read_text(encoding="utf-8"))
SK = M["skins"]["default"]
CANVAS_H = M["canvas"]["h"]
# 观感要按**桌面尺寸**读，不是按 896×768 的素材画布。默认显示高 150px。
SCREEN_H = 150.0
SCALE = SCREEN_H / CANVAS_H
FLIP_NOISE = 0.45       # 超过这个翻转率就不是摆动了，是抖
JUMP_VISIBLE = 1.2      # 屏幕上单帧跳超过这个数，就是一次看得见的"蹦一下"
AMP_MEAN = 0.35         # 平均摆幅低于这个数时，再高的翻转率也只是亚像素噪声
STEPS = 90              # 每条动画往后推多少个显示帧（够覆盖两三个步态周期）
ONE_SHOT = {"hug"}      # 播一遍就结束、不循环的那些


def geom(path: pathlib.Path):
    """前景质心、**头部带质心**、包围盒顶边、内容宽 —— 全部用画布像素。

    头部带 = 顶部 35% 那一条的质心。为什么判定要看它而不是包围盒顶边或整幅质心：
    包围盒顶边会在头顶、耳朵、甩起来的尾巴之间来回切换，它跳 3px 不代表她的头
    动了 3px；整幅质心又被伸出去的四肢带偏。这条教训和管线 `bob_dy` 用的是同一条。
    """
    a = np.asarray(Image.open(path).convert("RGBA"))[..., 3] > 40
    ys, xs = np.where(a)
    if not len(xs):
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    band = a[:y0 + max(1, int((y1 - y0 + 1) * 0.35))]
    by, bx = np.where(band)
    return (float(xs.mean()), float(ys.mean()),
            float(bx.mean()) if len(bx) else float(xs.mean()),
            float(by.mean()) if len(by) else float(ys.min()),
            float(y0), float(xs.max() - xs.min() + 1))


def shown_seq(name: str, a: dict) -> list:
    """运行时会依次画到的帧号。三种状态机三种走法，抄错一种就量错了东西：

    * 步态（走/跑/爬）：`Walker.tick()` 从 `gait.phase0` 起步，在 `gait.loop`
      窗口里按 stride 取模循环 —— 窗口可能比素材短（截到整数周期）。
    * 驻留（坐着/趴着/睡）：`PostureHold._show_loop()` 在 `loop` 窗口里按
      `loop_stride` **来回播**（折返点最容易"顿"一下）。
    * 一次性（抱抱）：从第 0 帧按 stride 正放一遍，不循环。
    * 其余（静立呼吸）：`IdlePlayer` 整段按 stride 循环。
    """
    n = int(a["frames"])
    st = max(1, int(a.get("stride", 2)))
    if "gait" in a:
        g = a["gait"]
        lo, hi = g.get("loop", [0, n])
        m = max(1, hi - lo)
        ph = int(g.get("phase0", lo)) - lo
        return [lo + (ph + k * st) % m for k in range(STEPS)]
    if "loop" in a:
        lo, hi = a["loop"]
        m = max(1, hi - lo)
        ls = max(1, int(a.get("loop_stride", 1)))
        if m == 1:
            return [lo] * STEPS
        span = 2 * m - 2
        return [lo + (k if k < m else span - k) for k in ((t * ls) % span for t in range(STEPS))]
    if name in ONE_SHOT:
        return [min(i * st, n - 1) for i in range(STEPS)]
    return [(i * st) % n for i in range(STEPS)]


def flip_rate(v) -> float:
    """相邻两步运动方向相反的比例。只在位移超过 0.3px 的地方计数，
    否则"0.01px → 0.02px"这种四舍五入噪声也算一次反转。"""
    d = np.diff(np.asarray(v, float))
    big = np.abs(d) > 0.3
    if big.sum() < 3:
        return 0.0
    s = np.sign(d[big])
    return float((s[1:] != s[:-1]).mean())


def main() -> int:
    print(f"素材画布 {M['canvas']['w']}x{CANVAS_H}，桌面默认显示高 {SCREEN_H:.0f}px"
          f" → 1 画布 px = {SCALE:.3f} 屏幕 px")
    print("以下全部按**运行时显示序列**算（stride、循环窗口、来回折返都已算进去）")
    print("判定看**头部带**（身体自己动没动）；顶边与轮廓宽度只当诊断列")
    print(f"{'动画':13} {'取样帧':>5} {'头跳':>7} {'头均摆':>7} {'头翻转':>7}  "
          f"{'质心跳':>7} {'顶边跳':>7} {'宽跳':>7}  判定")
    print("-" * 104)
    rows = []
    for name, a in SK["anims"].items():
        d = ROOT / "assets" / a["dir"]
        g = [x for x in (geom(d / f"frame_{i:03d}.png") for i in shown_seq(name, a)) if x]
        if len(g) < 4:
            continue
        r = dict(name=name, n=len(g))
        for col, k in (("hx", 2), ("hy", 3), ("x", 0), ("y", 1), ("t", 4), ("w", 5)):
            v = np.asarray([p[k] for p in g], float) * SCALE
            dd = np.abs(np.diff(v))
            r["jump" + col] = float(dd.max())
            r["amp" + col] = float(dd.mean())
            r["flip" + col] = flip_rate(v)
        # 头 = 顶部 35% 那条带：她的身体真的动了才算抖，四肢甩动不算
        r["jump"] = max(r["jumphx"], r["jumphy"])
        r["amp"] = max(r["amphx"], r["amphy"])
        r["flip"] = max(r["fliphx"], r["fliphy"])
        r["body"] = max(r["jumpx"], r["jumpy"])
        rows.append(r)
    rows.sort(key=lambda r: -r["jump"])
    bad = 0
    for r in rows:
        seen = r["jump"] > JUMP_VISIBLE
        noisy = seen and r["flip"] > FLIP_NOISE and r["amp"] > AMP_MEAN
        if noisy:
            bad += 1
        tag = ("!! 抖（头在高频反转，看得见）" if noisy else
               ("~ 头有大跳但方向不反转（接缝/折返，交给 seam_check）" if seen else
                ("~ 边缘" if r["jump"] > JUMP_VISIBLE * 0.7 else "ok")))
        print(f"{r['name']:13} {r['n']:>5} {r['jump']:7.2f} {r['amp']:7.2f} "
              f"{r['flip']:7.2f}  {r['body']:7.2f} {r['jumpt']:7.2f} "
              f"{r['jumpw']:7.2f}  {tag}")
    print("-" * 104)
    print(f"读法：主指标是**头的最大单帧跳**（屏幕 px，头=顶部 35% 带质心）。"
          f"判抖要三个条件同时成立：\n"
          f"      跳 >{JUMP_VISIBLE}、翻转率 >{FLIP_NOISE}、平均摆幅 >{AMP_MEAN}。"
          f"正弦摆动翻转率约 0.1，纯噪声约 0.67。\n"
          f"      顶边跳/宽跳是轮廓换形状（四肢、尾巴、耳朵），看着大但不一定是抖。")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
