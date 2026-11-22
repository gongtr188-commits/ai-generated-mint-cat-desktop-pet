"""随机系统核查：把"各项随机概率"从配置数字换算成**她实际的时间占比**。

为什么不能只看 `BEHAVIOR_WEIGHTS`：权重是"挑中谁"的概率，不是"她在干什么"的占比。
`roam:3, sit:2` 看起来是 60/40，但一次 roam 平均 4 秒、一次 sit 平均 20 秒，
再叠上各自的冷却、`pick()` 的"不许连续两次同一个"降级、以及"没做成 1.5 秒后重试"——
真正的观感是**她一天里有多少时间在走路、多少时间坐着发呆**，只有跑一遍才知道。

跑法：`py tools/behavior_check.py`（默认 12 条种子 × 每条 2 小时仿真）。
它复用 `smoke_test.FakePet`，但把转身时长换成**真实**的 9 tick，
否则"在忙"的时间被系统性低估，报表整份都偏（见 `_StubFacing` 的注释）。

报表怎么读：
  · 时间占比  —— 观感的直接来源。"站着没事干"太高像死机，"坐着"太高像摆件。
  · 动作/小时 —— 太密像抽搐，太稀像不动。
  · 连续重复率 —— `pick()` 会把上一轮那个踢出候选，所以**候选只剩一个**时它必然 100%，
    两个候选时则是**永远不重复**（严格交替）—— 那本身就是"看得出脚本"的机器味。
"""
from __future__ import annotations

import os
import random
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from mint_pet import config                      # noqa: E402
from smoke_test import FakePet                   # noqa: E402

HOURS = 2.0
SEEDS = 12


def ticks_for(hours: float) -> int:
    """仿真长度按秒算，别在模块导入时就把 tick 数钉死。

    （踩过：调小 HOURS 只想跑个快检，结果 `TICKS` 是 import 时按 2 小时算好的，
      仿真照跑 2 小时，报表却按 0.25 小时归一，"次动作/小时"直接虚高 8 倍。）
    """
    return max(1, int(hours * 3600 * 1000 / config.FPS_MS))


def state_of(pet) -> str:
    """这一 tick 她在干什么。判定顺序 = 优先级：越具体的越先判。"""
    h_sit, h_prone = pet.holders["sitting"], pet.holders["prone"]
    h_doze = pet.holders.get("dozing")
    h_sleep = pet.holders.get("sleeping")
    if pet.oneshot.busy:
        # 一次性动作与其余执行者互斥（两边起步时都互相查过），所以放最前面判。
        # 漏了它的话这 3 秒会掉进"站着没事"，idle 占比被系统性高估。
        return "拥抱"
    if h_sleep is not None and h_sleep.busy:
        # 趴睡和趴坐是两个姿态、两种身体，报表里必须分开，不然"趴着"这一栏
        # 会把"歇一会儿"和"睡一觉"混成一个数。
        return "趴着睡"
    if h_doze is not None and h_doze.busy:
        # 打瞌睡是坐着打盹（另一套素材），和"坐着微动"必须分开：
        # 合并了就会把坐这一类的时间占比虚高，看不出打瞌睡到底占多少。
        return "打瞌睡"
    if pet.crawler.busy:
        return "四足爬走"
    if pet.walker.busy:
        if pet.walker.phase != "walking":
            return "走路中转身"
        return "跑步" if pet.walker.gait == "run" else "走路"
    if h_sit.phase in ("entering", "leaving"):
        return "坐下过渡"
    if h_sit.phase == "holding":
        return "坐着微动"
    if h_prone.busy:
        return "趴着驻留"
    if pet.facing.turning:
        return "转身"
    return "站着没事"


def run_session(seed: int, side_prone: bool) -> dict:
    pet = FakePet(x=700, y=300, turn_ticks=config.TURN_FRAMES_90)
    pet.side_prone = side_prone
    # 这一行不是冗余：上一版这里写的是 `pet.allow_prone`（一个已经改名的属性）。
    # Python 只是给它新增了个字段，谁也没报错，于是"开关打开"那一档里爬走从来没被
    # 挑中过，两档报表几乎一模一样 —— 而它正是这个工具要抓的"配了没生效"。
    if pet.behaviors.available("crawl") != side_prone:
        raise SystemExit("behavior_check 与行为池对不上：side_prone 没有传到 available()")
    pet.behaviors.enabled = True
    pet.behaviors.rng = random.Random(seed)
    pet.walker.rng = random.Random(seed * 7919 + 1)
    for h in pet.holders.values():
        h.rng = random.Random(seed * 104729 + 2)
    pet.crawler.rng = random.Random(seed * 6151 + 3)

    share = Counter()
    picks = Counter()
    seq = []
    gaps = []                      # 两次"开始做一件事"之间歇了多久
    since = 0
    fired = 0
    for _ in range(ticks_for(HOURS)):
        pet.tick()
        share[state_of(pet)] += 1
        # 一次挑中 = `fired` 涨了一格。**不能**靠"last 和上次不同"来数：
        # 那样连着做两次的同一动作会被折叠成一次，"连续重复率"恒为 0 ——
        # 那是在量自己构造出来的序列，不是在量她。（第一版就栽在这儿。）
        if pet.behaviors.fired > fired:
            picks[pet.behaviors.last] += 1
            seq.append(pet.behaviors.last)
            gaps.append(since)
            since = 0
            fired = pet.behaviors.fired
        else:
            since += 1
    rep = sum(1 for a, b in zip(seq, seq[1:]) if a == b)
    return dict(share=share, picks=picks, seq=seq, repeats=rep, gaps=gaps,
                prone_on=side_prone)


def report(title: str, runs: list) -> None:
    total = sum(sum(r["share"].values()) for r in runs)
    n_pick = sum(sum(r["picks"].values()) for r in runs)
    rep = sum(r["repeats"] for r in runs)
    pairs = sum(len(r["seq"]) - 1 for r in runs if len(r["seq"]) > 1)
    gaps = [g for r in runs for g in r["gaps"]]
    print(f"\n{'=' * 66}\n{title}（{len(runs)} × {HOURS:g} 小时仿真）\n{'=' * 66}")
    agg = Counter()
    for r in runs:
        agg.update(r["share"])
    print("  时间占比")
    for name, t in agg.most_common():
        print(f"    {name:10} {t / total * 100:5.1f}%   {'█' * int(t / total * 100)}")
    print("  动作被挑中")
    merged = Counter()
    for r in runs:
        merged.update(r["picks"])          # 不能用 dict 推导式合并：键会互相覆盖，只剩最后一条
    # 参照系只能是**这一档配置下真的在池子里的那些动作**：开关关掉时 crawl 根本不可选，
    # 拿全量权重当分母会得出"roam 只占配置的 27%"这种废话。
    # 注意只有 crawl 被门控 —— 正面趴下（prone）是常驻动作，不归这个开关管。
    live = [n for n in config.BEHAVIOR_WEIGHTS
            if n != "crawl" or runs[0]["prone_on"]]
    live_total = sum(config.BEHAVIOR_WEIGHTS[n] for n in live)
    for name, c in merged.most_common():
        print(f"    {name:12} {c / n_pick * 100:5.1f}%  (本档权重占比 "
              f"{config.BEHAVIOR_WEIGHTS[name] / live_total * 100:.0f}%)")
    print(f"  频率：{n_pick / (len(runs) * HOURS):.1f} 次动作/小时"
          f"    平均歇 {sum(gaps) / max(1, len(gaps)) * config.FPS_MS / 1000:.1f}s 才动一下"
          f"    最长歇 {max(gaps) * config.FPS_MS / 1000:.0f}s")
    print(f"  连续重复率：{rep / max(1, pairs) * 100:.1f}%"
          f"{'  ← 严格交替，等于没有随机' if rep == 0 and pairs else ''}")


def main() -> int:
    print(f"节拍 {config.FPS_MS}ms  权重 {config.BEHAVIOR_WEIGHTS}"
          f"\n冷却 {config.BEHAVIOR_COOLDOWN_S}  空闲 {config.BEHAVIOR_GAP_S}s"
          f"\n驻留 {config.POSTURE_HOLD_S}  走 {config.WALK_S}s  跑门槛 {config.RUN_FROM_S}s")
    off = [run_session(s, False) for s in range(SEEDS)]
    on = [run_session(s, True) for s in range(SEEDS)]
    report("开关【关掉】（出厂默认：正面趴下/坐下/拥抱照常，只没有侧趴与四足爬走）", off)
    report("开关【打开】（多出 四足爬走 一类）", on)
    # 产出量断言：两档必须真的不一样。上一版这里两档报表几乎逐字相同，
    # 那是因为开关根本没接上 —— 而"看起来正常"的报表不会自己喊出来。
    c_off = sum(r["picks"].get("crawl", 0) for r in off)
    c_on = sum(r["picks"].get("crawl", 0) for r in on)
    print(f"\n{'=' * 66}")
    if c_off:
        print(f"  !! 开关关掉时爬走仍被挑中 {c_off} 次 —— 门控没生效")
    elif not c_on:
        print("  !! 开关打开时爬走一次都没被挑中 —— 它进不了池子（配了没人做）")
    else:
        print(f"  爬走：关掉 0 次 / 打开 {c_on} 次，门控与产出都对")
    return 0


if __name__ == "__main__":
    sys.exit(main())
