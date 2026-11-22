"""接缝审计：把运行时每一次"画面来源换人"的两端，用管线自己那把尺量出来。

为什么要有这个工具：用户说"不同状态、动作不能很好的衔接"，而"顺不顺眼"没法直接改。
桌宠每一帧只可能来自四个地方（静姿图 / 转身动画 / 步态动画 / 姿态过渡+驻留），
换来源的那一下就是接缝。把每一处接缝量化成"两帧灰度平均差"，就有了可改的东西。

**标尺怎么读**（同一量纲，48×48 灰度、0~255）：
    0.55   相邻两帧的自然差（中位）—— 完全看不出来
    1.94   自然差的 90 分位
    4.04   front.png ↔ idle_front#0 —— 产品里已经存在、用户没抱怨过的接缝，当基线
   11~14   跑步素材与侧视静帧的系统性差异（素材本身画得不一样，见 §13.13）
   19~31   一次真正的姿态变化（站↔坐 27.7、站↔趴 19.1、转 90° 30.8）
所以判"跳不跳"看的是相对这几档的位置，不是绝对值。

用法：py tools/seam_check.py
"""
from __future__ import annotations

import json
import math
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
GRID = 48
BASELINE_OK = 4.04          # front ↔ idle#0：产品里已存在的接缝，拿来当"可接受"刻度
POSE = 19.0                 # 站↔趴：一次真正的姿态变化，肯定看得出来的下限


def thumb(path: pathlib.Path) -> np.ndarray:
    """与 build_assets.thumb 同一把尺：48×48 灰度、铺白底、0~255。"""
    a = np.asarray(Image.open(path).convert("RGBA").resize((GRID, GRID), Image.BOX), np.float32)
    alpha = a[..., 3:4] / 255.0
    rgb = a[..., :3] * alpha + 255.0 * (1.0 - alpha)
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).ravel()


def anim_dir(name: str) -> pathlib.Path:
    return ROOT / "assets" / SK["anims"][name]["dir"]


def pose_path(name: str) -> pathlib.Path:
    return ROOT / "assets" / SK["poses"][name]["file"]


def d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def verdict(v: float) -> str:
    if v <= BASELINE_OK:
        return "ok    "
    if v <= BASELINE_OK * 2:
        return "略跳  "
    if v >= POSE:
        return "!! 素材本身不同"
    return "!! 明显跳"


def main() -> int:
    rows = []
    P = {k: thumb(pose_path(k)) for k in SK["poses"]}

    # ── 步态：起步（静帧 → 起步那一帧）与收步（循环里任意一帧 → 静帧）────────
    # 起步相位由管线标定（gait.phase0）；收步落在哪一帧由"走到目标点"决定，不可控，
    # 所以收步要看**整窗分布**，不能只看最静的那一帧。
    GAIT = {"walk_left": "side_left", "walk_right": "side_right",
            "run_left": "side_left", "run_right": "side_right",
            "walk4_left": "prone_left", "walk4_right": "prone_right"}
    for name, pk in GAIT.items():
        a = SK["anims"].get(name)
        if not a:
            continue
        g = a.get("gait") or {}
        lo, hi = g.get("loop", [0, int(a["frames"])])
        ph0 = int(g.get("phase0", 0))
        ds = np.array([d(thumb(anim_dir(name) / f"frame_{i:03d}.png"), P[pk])
                       for i in range(lo, hi)])
        rows.append((f"{name} 起步  {pk}→f{ph0}", ds[ph0 - lo]))
        rows.append((f"{name} 收步  整窗均值", float(ds.mean())))
        rows.append((f"{name} 收步  最坏 f{lo + int(ds.argmax())}", float(ds.max())))
        rows.append((f"{name} 收步  若停在素材末帧", float(ds[-1])))

    # ── 转身：静帧 ↔ 转身动画两端 ──────────────────────────────────────────
    t = SK["anims"]["turn"]
    n = int(t["frames"])
    keys = t.get("keys") or []
    faces = t.get("faces") or []
    if keys and faces:
        for fk, kf in zip(faces, keys):
            rows.append((f"转身进入  {fk}→turn#{kf}", d(P[fk], thumb(anim_dir("turn") / f"frame_{kf:03d}.png"))))
    rows.append(("转身收尾  turn末帧→faces[0]",
                 d(thumb(anim_dir("turn") / f"frame_{n - 1:03d}.png"), P[faces[0]] if faces else P["front"])))

    # ── 呼吸：静帧 ↔ idle 两端 ─────────────────────────────────────────────
    idle = SK["anims"]["idle_front"]
    ni = int(idle["frames"])
    rows.append(("呼吸进入  front→idle#0", d(P["front"], thumb(anim_dir("idle_front") / "frame_000.png"))))
    rows.append(("呼吸收尾  idle末帧→front",
                 d(thumb(anim_dir("idle_front") / f"frame_{ni - 1:03d}.png"), P["front"])))
    # 呼吸随时会被走路/转身打断，打断点落在哪一帧不可控 —— 所以要看整段最偏的一帧。
    ID = np.array([thumb(anim_dir("idle_front") / f"frame_{i:03d}.png") for i in range(ni)])
    spread = np.array([d(f, P["front"]) for f in ID])
    rows.append((f"呼吸中途  idle#{int(spread.argmax())}→front（被打断的最坏点）", float(spread.max())))

    # ── 坐下：进入起点、过渡→驻留交接、起身收尾 ─────────────────────────────
    sd = SK["anims"]["sitdown"]
    ns = int(sd["frames"])
    ent, settle = int(sd["enter_at"]), int(sd["settle_at"])
    lo, hi = sd["loop"]
    S = np.array([thumb(anim_dir("sitdown") / f"frame_{i:03d}.png") for i in range(ns)])
    step = int(sd.get("stride", 2))
    rows.append(("坐下进入  front→sitdown#16", d(P["front"], S[ent])))
    rows.append(("坐下交接  #76→#77（过渡→驻留）", d(S[settle - 1], S[min(settle, ns - 1)])))
    rows.append(("坐下驻留  窗口内最忙的一帧", float(np.abs(np.diff(S[lo:hi], axis=0)).mean(1).max())))
    rows.append(("起身收尾  sitdown#16→front", d(S[ent], P["front"])))
    rows.append(("（对照）素材第 0 帧→front", d(S[0], P["front"])))
    # 交接处真实跨了几帧：过渡段按 stride 推进，最后一步可能不是 1 帧
    last = ent + ((settle - 1 - ent) // step) * step
    rows.append((f"坐下交接  实际最后一步 #{last}→#{settle}", d(S[last], S[settle])))

    # ── 姿态之间没有过渡素材的直切（已知，量出来才知道有多疼）──────────────
    rows.append(("站→趴  front→prone_front（无过渡素材）", d(P["front"], P["prone_front"])))
    rows.append(("站→坐  front→sit_front（无过渡素材时）", d(P["front"], P["sit_front"])))

    # ── 驻留微动（趴着 / 打瞌睡 / 左右躺睡）：进入、来回折返、被打断 ─────────
    # 这四段都是"一待就是十几到三十秒"的常驻画面，接缝观感被时间放大，所以逐段量：
    # 进入那一跳、来回一轮的最大折返跳、以及中途被拖走时最坏那一帧到静姿图的差。
    # （旧版这里只有一条 prone_sleep：那是接错的那段素材，量的是"站着直接躺下"。）
    HOLDS = (("prone_idle", "prone_front", "正面趴着"),
             ("doze", "doze_front", "打瞌睡"),
             ("sleep_left", "sleep_left", "向左躺睡"),
             ("sleep_right", "sleep_right", "向右躺睡"))
    for anim_name, pose_key, label in HOLDS:
        hd = SK["anims"].get(anim_name)
        if not hd:
            continue
        npl = int(hd["frames"])
        lo, hi = hd.get("loop", [0, npl])
        st = int(hd.get("loop_stride", 1))
        PI = np.array([thumb(anim_dir(anim_name) / f"frame_{i:03d}.png")
                       for i in range(npl)])
        ref = P[pose_key]
        rows.append((f"{label}进入  front→{anim_name}#{lo}", d(P["front"], PI[lo])))
        if pose_key != "prone_front":
            # 趴姿的驻留素材本来就是从 prone_front 那一帧起步的，比它自己没意义
            rows.append((f"{label}↔趴坐  {pose_key}→prone_front（两种身体差多少）",
                         d(ref, P["prone_front"])))
        rows.append((f"{label}驻留  窗口内最忙的一帧",
                     float(np.abs(np.diff(PI[lo:hi], axis=0)).mean(1).max())))
        # 来回一轮真实播到的帧序列（和 PostureHold._show_loop 同一段算法），
        # 端点折返处最容易"顿"一下，所以要量相邻两画之间最大的一跳。
        # n==1 时 _show_loop 直接停在单帧、不来回播，这一档就不存在。
        n = max(1, hi - lo)
        if n > 1:
            span = 2 * n - 2
            period = span // math.gcd(st, span)
            played = [lo + (k if k < n else span - k)
                      for k in ((t * st) % span for t in range(period))]
            rows.append((f"{label}驻留  来回一轮最大一跳（步长{st}）",
                         float(max(d(PI[x], PI[y]) for x, y in zip(played, played[1:])))))
        # 睡着随时会被拖走/打断：起来那一下画的是 front，看整窗最偏的一帧。
        # 另一把尺量"这段微动自己有多大"（离静姿最远的一帧）——它决定改尺寸时
        # 重画那一帧会不会跳，和"被打断"是两件事，别混成一行。
        to_front = np.array([d(f, P["front"]) for f in PI[lo:hi]])
        rows.append((f"{label}退出  最坏帧#{lo + int(to_front.argmax())}→front（被打断）",
                     float(to_front.max())))
        to_pose = np.array([d(f, ref) for f in PI[lo:hi]])
        rows.append((f"{label}幅度  离静姿最远的一帧#{lo + int(to_pose.argmax())}",
                     float(to_pose.max())))

    # ── 一次性动作（hug）：首尾接缝 + 进出两端 ────────────────────────────
    # 它**不进 loop** 的前提就是首尾接不上，所以这个比值必须量出来，不能写感觉。
    hg = SK["anims"].get("hug")
    if hg:
        nh = int(hg["frames"])
        H = np.array([thumb(anim_dir("hug") / f"frame_{i:03d}.png") for i in range(nh)])
        natur = float(np.abs(np.diff(H, axis=0)).mean(1).mean())
        seam = d(H[0], H[nh - 1])
        rows.append((f"hug 首尾接缝／相邻帧差中位 = {seam / max(natur, 1e-6):.1f} 倍"
                     f"（{seam:.2f} vs {natur:.2f}）", seam))
        rows.append(("拥抱进入  front→hug#0", d(P["front"], H[0])))
        rows.append(("拥抱退出  hug末帧→front", d(H[nh - 1], P["front"])))
        spread = np.array([d(f, P["front"]) for f in H])
        rows.append((f"拥抱中途  hug#{int(spread.argmax())}→front（被打断的最坏点）",
                     float(spread.max())))

    print(f"{'接缝':46} {'差值':>6}   相对基线 4.04")
    print("-" * 78)
    bad = 0
    for label, v in sorted(rows, key=lambda r: -r[1]):
        # "幅度"那几行量的不是接缝，而是这段微动自己离静姿有多远（决定改尺寸重画
        # 会不会跳）。拿接缝的尺子判它，报表上就会多出一堆假的"明显跳"。
        if "幅度" in label:
            print(f"{label:46} {v:6.2f}   信息  {'░' * int(min(40, v))}")
            continue
        mark = verdict(v)
        bad += mark.startswith("!!")
        print(f"{label:46} {v:6.2f}   {mark} {'█' * int(min(40, v))}")
    print("-" * 78)
    print(f"{len(rows)} 处接缝，{bad} 处超过基线两倍")
    return 0


if __name__ == "__main__":
    sys.exit(main())
