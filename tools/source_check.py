"""源素材体检：新素材到货先跑这个，再决定怎么接（只读 素材/，不写盘）。

    py -3 tools/source_check.py                 # 全部视频
    py -3 tools/source_check.py 向左跑 向右跑    # 只查指定的

为什么需要：`向左走_四足` 那条是"角色自己走出画面"的废素材，
接进管线之后才被胶片图看出来，白重建两轮素材。
这些问题在**源视频**上一次就能量出来：
  帧数/帧率  背景亮度  贴边与裁切  水平漂移  步态周期  相邻帧断崖
"""
from __future__ import annotations

import sys
import pathlib
import importlib.util
import numpy as np
import cv2

ROOT = pathlib.Path(__file__).resolve().parents[1]
try:                     # 控制台默认 GBK，诊断里的 "⚠" 会让 print 直接抛异常
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
WANT = sys.argv[1:]              # 先取走：下面加载管线模块时会改写 sys.argv
spec = importlib.util.spec_from_file_location("ba", ROOT / "tools" / "build_assets.py")
ba = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(ba)          # 复用管线的抠图判据，量的是同一套前景定义

VID = ROOT / "素材" / "薄荷_猫猫" / "视频"
BAND = ba.SRC_EDGE_PX
DRIFT_MAX = 20.0        # 水平漂移超过这个值 = 素材自己在平移（要 recenter）
CLIP_DROP = 0.08        # 贴边且面积比中位少 8% 以上 = 真被切掉


def load(path, cap_n=400):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    out = []
    while len(out) < cap_n:
        ok, x = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
    cap.release()
    return out, fps


def report(path: pathlib.Path):
    fr, fps = load(path)
    n = len(fr)
    if not n:
        print(f"{path.stem:<20} 读不到帧")
        return
    first = fr[0]
    corner = np.concatenate([first[:20, :20], first[:20, -20:],
                             first[-20:, :20], first[-20:, -20:]]).astype(int)
    bg_lum = float(np.median(corner.mean(1)))
    bg_span = float(np.median(corner.max(1) - corner.min(1)))

    fgs = [ba.foreground(x) for x in fr]
    areas = np.array([int(f.sum()) for f in fgs], float)
    med = float(np.median(areas)) or 1.0
    cols = [np.where(f.any(0))[0] for f in fgs]
    cxs = np.array([(c.min() + c.max()) / 2 for c in cols])
    rows = [np.where(f.any(1))[0] for f in fgs]
    hs = np.array([int(r.max() - r.min() + 1) for r in rows], float)
    ring = np.array([int(f[:BAND].sum() + f[-BAND:].sum()
                         + f[:, :BAND].sum() + f[:, -BAND:].sum()) for f in fgs])
    clipped = [i for i in range(n) if ring[i] > 0 and areas[i] < med * (1 - CLIP_DROP)]
    near = int((ring > 0).sum())

    # 步态周期：签名自相关在 lag≥8 之后找**局部峰**。
    # 不能直接取全局最大——相邻帧本来就相似，最短 lag 永远是峰，
    # 那样任何素材都会报"周期 4 帧"（第一版就是这么错的）。
    sig = np.array([np.concatenate([f.sum(0), f.sum(1)]).astype(np.float32) for f in fgs])
    sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-6)
    lag = max(2, n // 2)
    ac = np.array([float((sig[:n - k] * sig[k:]).mean()) for k in range(1, lag)])
    peaks = [k + 1 for k in range(7, len(ac) - 1) if ac[k] >= ac[k - 1] and ac[k] > ac[k + 1]]
    period = max(peaks, key=lambda k: ac[k - 1]) if peaks else 0
    strength = float(ac[period - 1]) if period else 0.0

    # 整段能不能首尾相接：末帧→首帧的距离 vs 段内相邻帧距离的中位。
    # 呼吸/微动这类"周期=整段"的素材没有短峰，但接缝很平，一样能循环。
    step = np.array([float(np.abs(sig[i + 1] - sig[i]).mean()) for i in range(n - 1)])
    med_step = float(np.median(step)) or 1e-6
    seam = float(np.abs(sig[0] - sig[-1]).mean()) / med_step

    d = np.abs(np.diff(areas)) / med
    cliff = int(np.argmax(d))
    drift = float(cxs.max() - cxs.min())
    loco = any(w in path.stem for w in ("走", "跑"))     # 只有位移动作才谈"漂移"
    loops = (strength > 0.6 and period >= 8) or seam < 2.5

    print(f"\n=== {path.stem} ===")
    print(f"  {n} 帧 @ {fps:.1f}fps（{n / max(fps, 1):.1f}s）  尺寸 {fr[0].shape[1]}x{fr[0].shape[0]}"
          f"  背景 亮度{bg_lum:.0f}/通道差{bg_span:.0f}")
    print(f"  内容高 {hs.min():.0f}~{hs.max():.0f}（起伏 {100 * (hs.max() - hs.min()) / hs.mean():.1f}%）"
          f"  面积 {100 * areas.min() / med:.0f}~{100 * areas.max() / med:.0f}% 中位")
    print(f"  水平漂移 {drift:.0f}px"
          + (f"  →  {'需要 recenter' if drift > DRIFT_MAX and loco else '可用 sequence/per_frame'}"
             if loco else "（非位移动作，仅供参考）"))
    if loco:
        # 漂移分两种，处理方式相反：
        #   趋势 = 她真的在画面里被推着走 → 必须 recenter 抵消，否则观感是"滑步"
        #   残差 = 迈步时身体左右晃 → 是要**保留**的动作，抵消掉就等于把动画压平
        t = np.arange(n)
        slope, icept = np.polyfit(t, cxs, 1)
        trend = abs(slope) * (n - 1)
        osc = float(np.ptp(np.abs(cxs - (slope * t + icept))) / 2)
        print(f"    其中趋势 {trend:.0f}px（{'要抵消' if trend > 12 else '可忽略'}）"
              f" + 摆动 ±{osc:.0f}px（要保留）")
    print(f"  步态周期 ≈{period}帧（自相关 {strength:+.2f}）  首尾接缝 {seam:.1f}×相邻帧差"
          f"  →  {'可循环' if loops else '⚠ 接成 loop 会硬跳'}")
    print(f"  相邻帧最大面积突变 {d[cliff] * 100:.1f}% @f{cliff}→{cliff + 1}"
          f"  （中位 {np.median(d) * 100:.2f}%）")
    print(f"  贴边 {near} 帧，其中判为**被源画面切掉** {len(clipped)} 帧"
          f"{f' 首帧 #{clipped[0]} 例 {clipped[:6]}' if clipped else ''}")
    if bg_lum < 252:
        print(f"  ⚠ 背景亮度 {bg_lum:.0f} < 252：与她白毛(≈248)差 {bg_lum - 248:+.0f} 级，"
              f"压缩振铃一压暗就可能留在前景里（见开发文档 §11.12）")
    bad = []
    if clipped:
        bad.append("角色出画")
    if not loops:
        bad.append("不可循环")
    if loco and drift > DRIFT_MAX:
        bad.append("画面内平移")
    print(f"  结论： {'；'.join(bad) if bad else '可以接'}")


def main():
    files = sorted(VID.glob("*.mp4"))
    if WANT:
        files = [f for f in files if any(w in f.stem for w in WANT)]
    if not files:
        print(f"没找到匹配的素材：{WANT}（现有：{[f.stem for f in sorted(VID.glob('*.mp4'))]}）")
        return
    for f in files:
        report(f)


main()
