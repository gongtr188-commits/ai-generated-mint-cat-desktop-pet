#!/usr/bin/env python3
"""薄荷（猫猫）素材管线（唯一入口）。

素材有任何变更就重跑本脚本并核对 MD5，禁止临时改写脚本重生成。
用法：
    py -3 tools/build_assets.py                 # 全量重建
    py -3 tools/build_assets.py --only poses    # 只重建静态姿势
    py -3 tools/build_assets.py --report        # 只诊断不写盘
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# 诊断表里有中文和符号，Windows 控制台默认 GBK —— 一个 "⚠" 就能让 print 抛
# UnicodeEncodeError，而它发生在"帧已全部写盘、清单还没写"之间，
# 结果是把整个素材库留在半新半旧的状态。统一按 UTF-8 输出，编不出的字符替换掉。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                          # 不是标准流（被重定向成非文件对象）时跳过
    pass

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "素材" / "薄荷_猫猫"
OUT = ROOT / "assets"

# ── 画布规格（改动即全量重建；数值理由见 docs/素材规范.md）────────────────
# 宽 ≠ 高：坐姿/趴姿在同一机位下比站姿宽得多。按 CHAR_H 归一后实测需宽
#   正面 706 / 坐下_正面 830 / 向左趴着 793 / 趴下_正面 757 / 向右趴着 726
# 768 的方画布会把坐下直接切掉 62px。取 896 = 最大需求 830 + 左右各 33px 余量。
CANVAS_W = 896
CANVAS_H = 768
CHAR_H = 660          # 内容归一化后的目标高
CONTENT_MAX_W = CANVAS_W - 2 * 24   # 内容宽上限，两侧各留 24px 防止贴边被窗口裁掉
BASELINE = 24         # 内容底边距画布底
FPS_MS = 80           # 运行时唯一节拍，写进清单给桌宠读；下面的时长标定也用它算
FRAME_STRIDE = 2      # 24fps 素材配 80ms 节拍 = 每 2 帧取 1 帧。运行时抖动一律按
                      # **抽稀之后**的显示序列量，逐帧原始序列会把观感放大好几倍

# ── 背景判据（详见 background_like()）───────────────────────────────────
# 源图背景实测 246~255（视频比静帧暗，坐下那条只有 246），通道极差 ≤3。
# 判据从"绝对亮度阈值"换成了"与本帧背景色的相对容差"，理由和 610 帧的实测数据
# 都写在 background_like() 的 docstring 里；开发文档 §11.12 记着为这事踩过的死路。
FLOOD_TOL = 8         # 与"本帧四边中位色"的最大通道差
BG_SPAN = 12          # 且通道极差要小：高饱和的（青毛/粉发）永远不算背景
# 低饱和浅灰的放宽档。源素材的**渐变暗角与地面投影**不是纯背景色，离四边中位
# 差 10~30 级，卡在 FLOOD_TOL 外就整片留在前景里。实测两处：
#   向右趴着睡觉  背景中段 234~237（四边 246），泛洪走不进去 → 前景面积 44 万（真身 20 万）
#   微笑拥抱      身下投影 225~240，深色桌面上是一摊白雾，逐帧忽大忽小还带抖
# 放宽到 30 级后：前者 20 万、后者只掉 9k（就是投影），其余 15 段素材面积变化 <1.5%。
# 能放宽的前提仍是"只能从四边连通"——白毛(248)被暗描边围住，泛洪走不进去。
BG_GRAY_TOL = 30      # 低饱和像素（投影/暗角）相对本帧背景色的通道差上限
BG_GRAY_SPAN = 16     # 低饱和的判定：通道极差仍要小，青青粉粉的毛发远大于此
FEATHER = 0.7         # alpha 高斯羽化半径
BOB_SMOOTH = 5        # 纵向去抖的低通窗口（源帧数）。必须远小于步态周期（14~26 帧），
                      # 否则会把"每步一次的真弹跳"一起抹平，走路就变成贴地滑行
BOB_MAX_DX = 8        # 单帧纵向补偿上限（画布 px ≈ 1.6 屏幕 px）。见 bob_dy 的第二条
BOB_ANIMS = {"walk_left", "walk_right", "run_left", "run_right",
             "walk4_left", "walk4_right"}
                      # **只有位移类动画**做纵向去抖。坐着/睡着/呼吸这些身体是贴地的，
                      # 底边对齐就是"脚踩在地上"；把头部钉到趋势上等于让脚离地，
                      # 还会破坏与静帧的基线对齐（冒烟那条 ±1px 的锁就是抓这个的）。
                      # 走路跑步本来就有一瞬间双脚离地，抬一点是物理正确的。
CHOKE = 2             # 轮廓内收像素数，专治 h264 在暗描边外圈的振铃亮边（见下方 foreground()）
TONE_STRENGTH = 0.85    # 色彩矩阵与单位阵的混合强度；留 15% 给视频自身的立体明暗
CYAN_REF = (89.0, 209.0, 226.0)   # 正面图青毛实测均值，仅作报告探针

# 源视频右下角"豆包AI生成"水印的位置，用**比例**而不是像素，换分辨率不用重标。
# 实测 121/121 帧都在这一带；静帧没有水印，所以只对视频生效。
WM_BOX = (0.806, 0.925, 0.992, 0.978)
WM_LUM = (135, 244)       # 文字本体的亮度带：比背景(246~248)暗，比描边(<110)亮
WM_SAT = 26               # 水印是低饱和中性灰；青毛/粉都远高于此，天然可分
OUTLINE_LUM = 110         # 暗描边的亮度上限，当作连通性的"墙"
DEBRIS_MAX = 400          # 描边外小于此面积的连通块视为杂质（见 drop_debris）
SRC_EDGE_PX = 8           # 检出源画面裁切要看的边带宽度。**必须大于 CHOKE**：
                          # 轮廓内收会把最外 CHOKE 圈吃掉，被切的那条边在掩膜里
                          # 根本不存在（取 2 时实测全为 0，等于没检测）。8px 下
                          # 干净素材恒为 0，只有 向左走_四足 报 1166~2048px
SRC_CLIP_DROP = 0.08      # 且该帧前景要比整段中位少 8% 才算"被切掉"（只贴边不算）
SCRUB_LEAK_OK = 300       # 水印涂掉后仍属前景的像素容忍量。实测：填色配套错误时
                          # 转身 2417px / 坐下 4633px；而"笔画被毛发包住"的合法残留
                          # 只有 47px（宁留淡痕不打穿主体），门限落在两者之间

# 每个动画的放置策略，差别是有意的：
#   per_frame  逐帧把内容高钉死到 CHAR_H —— 转身时身高不该变，7.3% 起伏纯属生成噪声
#   sequence   整段共用一个缩放（按平均高）+ 水平锚点取全段中位数。
#              静立那 1.7% 的高度起伏就是呼吸、±3.6px 的横向位移就是摇摆，
#              逐帧归一会把呼吸压平，逐帧居中会把摇摆抵消掉。
#   recenter   整段共用一个缩放 + **逐帧水平居中**。给"素材里人真的在画面里平移"的
#              四足走用：向左走_四足实测水平中心漂移 69px（现有走路只有 4~6px），
#              按 sequence 处理会把这 69px 当成摇摆，看起来就是爬着滑来滑去。
#              趴睡/正趴这类"身体该钉在地上"的驻留素材也用它：素材里的身体漂移
#              只是生成噪声，跟着它走就是躺着左右晃。
#   head       整段共用一个缩放，水平锚点只抵消**头部**的位移（抱抱用）。
#              拥抱时手臂左右开合，包围盒中心来回 7px；按 sequence 会把整只猫
#              平移着抖。头部实测只动 1.1px，拿它当锚，身体不动、手照样抱。
ANIM_MODE = {"turn": "per_frame", "idle_front": "sequence",
             "walk_left": "sequence", "walk_right": "sequence",
             "run_left": "sequence", "run_right": "sequence",
             "sitdown": "sequence", "prone_turn": "sequence",
             "walk4_left": "recenter", "walk4_right": "recenter",
             "prone_idle": "recenter", "doze": "sequence",
             "sleep_left": "recenter", "sleep_right": "recenter",
             "hug": "head"}

# 坐下是一条**一次性过渡 + 一段常驻微动**，素材本身没分三段。
# 由 sit_segments 以静姿图为锚点实测 进入起点 / 交接点 / 待机窗口 写进清单，运行时
# 播 [enter_at, settle_at) 为过渡、loop 内来回播。分界全部量出来，不手填。
SETTLE_SEARCH = ("sitdown",)
# 待机窗口长度与静度门限：18 帧来回一轮 (2*18-2)×80ms = 2.7s，接近呼吸节奏。
# 门限 1.35 是实测定的 —— 坐下素材坐定之后最静的 18 帧窗口平均 0.26 灰阶，
# 起点 f77 起同长窗口 0.34 达标，再往前（f53~f76 那段收腿）就全超过门限被排除。
HOLD_LEN, HOLD_TOL = 18, 1.35

# 向右走有生成瑕疵：每个步态周期固定约 4 帧，肚皮右缘多伸出一小撮带描边的青色"左手"。
# 判据是与全段的**时位中位轮廓**比：凸起只在少数帧出现，中位形状天然不含它。
# 相位和容差都是实测出来的 —— 按 i%24 分组的凸出面积：
#   相位 23=1250 / 0=1320 / 22=946 / 1=656，其余 20 个相位全部 <550
# 而向左走同一指标是 85~274 的平坦分布，没有结构，所以不启用。
# 只在瑕疵相位上裁，其余帧一个像素都不碰 —— 中位轮廓里腿爪的正常摆动会被当成"凸出"，
# 无条件全裁会削掉合法动作（实测非瑕疵帧也有 40~550px 凸出）。
ANIM_ERASE = {"walk_right": dict(
    period=24,
    # 凸起（多出来的一只"左手"）实测跨 ph19→ph2，峰值在 ph21。
    # 只擦 22/23/0/1 会留下两头：ph20/21 那一坨最大，ph2 还剩一个缺口。
    phases=(19, 20, 21, 22, 23, 0, 1, 2),
    dilate=3,
    roi=(640, 495, 720, 615))}      # 画布坐标。凸起活动范围 x663-687 y527-588 的外扩

# 键是**完整文件名（不含扩展名）**，不是后缀。
# 曾经写成后缀通配（`*正面.png`），而新增的 坐下_正面 / 趴下_正面 同样以 `_正面` 结尾，
# 于是 front 静默取到了坐姿图 —— 内容宽从 706 变 830、站立转身 keys 从
# [1,71,47,28] 漂到 [88,71,47,28]，全程不报错。后缀匹配在这里根本无法消歧。
POSE_MAP = {"薄荷猫猫_正面": "front", "薄荷猫猫_左面": "side_left",
            "薄荷猫猫_右面": "side_right", "薄荷猫猫_背面": "back",
            "薄荷猫猫_坐下_正面": "sit_front",
            "薄荷猫猫_趴下_正面": "prone_front",
            "薄荷猫猫_向左趴着": "prone_left", "薄荷猫猫_向右趴着": "prone_right"}
VIDEO_MAP = {"薄荷猫猫_转身360度": "turn", "薄荷猫猫_静立_正面": "idle_front",
             "薄荷猫猫_向左走": "walk_left", "薄荷猫猫_向右走": "walk_right",
             "薄荷猫猫_向左跑": "run_left", "薄荷猫猫_向右跑": "run_right",
             "薄荷猫猫_坐下": "sitdown", "薄荷猫猫_趴着转180度": "prone_turn",
             "薄荷猫猫_向左走_四足": "walk4_left", "薄荷猫猫_向右走_四足": "walk4_right",
             "薄荷猫猫_正面趴着": "prone_idle", "薄荷猫猫_打瞌睡": "doze",
             "薄荷猫猫_向左趴着睡觉": "sleep_left",
             "薄荷猫猫_向右趴着睡觉": "sleep_right",
             "薄荷猫猫_微笑拥抱": "hug"}

# 有的姿态**只有视频、没有静姿图**（打瞌睡/左右趴睡就是）。取该动画的第 N 帧原样当静姿图：
# 它已经是画布尺寸、已校正色调、已按同一套基准归一，和任何一帧同源，
# 比另外找一张图来凑更不可能跑偏。键是姿势键，值是 (动画键, 帧号)。
# 正趴有专门的静姿图（趴下_正面.png），所以 正面趴着.mp4 只当驻留微动，不派生姿势。
POSE_FROM_VIDEO = {"doze_front": ("doze", 0),
                   "sleep_left": ("sleep_left", 0),
                   "sleep_right": ("sleep_right", 0)}

# ── 步态循环的相位对齐 ───────────────────────────────────────────────────
# 走路/跑步这类动画是"整段循环播放"的，但素材长度不一定是整数个步态周期。
# 不是整数个的话，每循环一次相位就错开一次，观感是"走两步绊一下"。
# 实测：向右走_四足 97 帧、周期 23 帧 = 4.2 个周期，首尾接缝是相邻帧差的 7.4 倍；
# 而 向左走/向右走/向左跑/向右跑 的接缝只有 0.3~1.6 倍，本来就能接上。
# 所以只有**接缝超标**的才截到整数个周期，并在所有可能的起点里挑首尾最接得上的那个。
GAIT_ANIMS = ("walk_left", "walk_right", "run_left", "run_right",
              "walk4_left", "walk4_right")
GAIT_SEAM_OK = 2.5      # 接缝 / 相邻帧差 中位数，超过这个倍数才值得为它截窗口
# 姿态驻留用的"整段都是微动"的素材：管线为它们标定来回播的窗口与步长（见 hold_window）。
HOLD_LOOPS = ("prone_idle", "doze", "sleep_left", "sleep_right")
GAIT_MIN_PERIOD = 8     # 短于此的"周期"是相邻帧相似度，不是步态周期


def pick_asset(folder: Path, stem: str, ext: str, what: str) -> Path:
    """精确取 `素材/<folder>/<stem>.<ext>`。缺文件直接退出，不做任何模糊匹配。"""
    p = folder / f"{stem}{ext}"
    if not p.exists():
        near = sorted(q.name for q in folder.glob(f"*{stem[-3:]}*{ext}"))
        sys.exit(f"缺少{what}素材：{p.name}\n  该目录下相近的文件有：{near or '（空）'}\n"
                 f"  管线按完整文件名匹配，请把 {what} 名补进 build_assets 的映射表。")
    return p

# 每条转身视频覆盖哪些姿势。转身图按**姿态**分组：站立用 转身360度，
# 趴着用 趴着转180度（实测它确实是从左趴经过正面趴转到右趴）。
# 列表顺序就是运行时该姿态内 facing 索引的顺序。
# **顺序必须与所属姿态的 faces 完全一致。** 运行时候转身是按 `facing.value` 去索引 keys 的，
# 而 facing.value 就是 faces 的下标 —— 顺序不一致不会报错，只会转错方向：
# 趴着时"向左看"会落到 keys[1]，而按视频时间顺序（左→正→右）排的那份 keys[1] 是正面趴。
# （实测就是这个 bug：趴着往左爬，她转成面朝镜头继续爬。）
# 视频本身的播放顺序是 左→正→右，所以这里的 keys 会是 [40, 8, 89] 这种非单调序列 ——
# 没关系，LinearTurn 算距离用的是真实帧号，不看 keys 的排列。
TURN_POSES = {"turn": ["front", "side_left", "back", "side_right"],
              "prone_turn": ["prone_front", "prone_left", "prone_right"]}
# 线性弧：素材只覆盖一条单向路径（左趴→正面趴→右趴），不是能循环的 360°。
# 运行时据此选 LinearTurn，否则"从右趴回左趴"会被取模算成绕一整圈。
LINEAR_TURNS = ("prone_turn",)

# 姿态 × 朝向 的注册表。facing 只在**同一姿态内**有意义，所以转身图也挂在姿态上。
# enter 是一次性过渡动画（坐下），idle 是该姿态下的常驻微动动画。
# walk / run 是两套步态：跑只给了站姿（素材就是四条腿着地爬，没有"站着跑"）。
POSTURES = {
    "standing": dict(label="站立", faces=["front", "side_left", "back", "side_right"],
                     turn="turn", enter=None, idle="idle_front",
                     walk={"left": "walk_left", "right": "walk_right"},
                     run={"left": "run_left", "right": "run_right"}),
    "sitting": dict(label="坐下", faces=["sit_front"],
                    turn=None, enter="sitdown", idle="sitdown", walk=None, run=None),
    # 正面趴着.mp4 是**睁眼**的趴坐：星星眼、嘴一开一合，没有睡觉的意思。
    # 它只属于正趴的正面，所以 idle 按朝向给成 [有, 无, 无]：
    # 侧趴时停在侧趴静姿上，否则会把正面那只身体画到侧脸下面。
    "prone": dict(label="趴着", faces=["prone_front", "prone_left", "prone_right"],
                  turn="prone_turn", enter=None,
                  idle=["prone_idle", None, None],
                  walk={"left": "walk4_left", "right": "walk4_right"}, run=None),
    # 打瞌睡是**坐着打盹**（闭眼、头一点一点），和坐下共用"只正面"的形态，
    # 但身体素材不同，所以单独一个姿态；它也**不受**左右趴着开关管。
    "dozing": dict(label="打瞌睡", faces=["doze_front"],
                   turn=None, enter=None, idle="doze", walk=None, run=None),
    # 趴着睡是**另一个身体**：侧躺、闭眼、身体横过来。左右两段素材各是一张朝向着，
    # 所以 faces 有两项、idle 一一对应；没有过渡素材，进入/换边都是硬切。
    # 它整族归"左右趴着"开关管（运行时 + 菜单 + 随机池三处都问同一个开关）。
    "sleeping": dict(label="趴着睡", faces=["sleep_left", "sleep_right"],
                     turn=None, enter=None, idle=["sleep_left", "sleep_right"],
                     walk=None, run=None),
}
# 总开关关掉时不可用的姿势/动画。按用户要求：正面趴下、坐下、打瞌睡是常驻的，
# 而"左右趴着（四足着地）+ 四足走 + 左右趴着睡觉"要打开开关才进随机池/菜单。
GATED = ["prone_left", "prone_right", "walk4_left", "walk4_right",
         "sleep_left", "sleep_right"]

# 步态动画 → 起步前她正对着的那张静态朝向图，用来标定"从第几帧开始播"（见 gait_phase）。
# 从姿态注册表推导，不另列一份表，免得加了新步态却忘了更新。
GAIT_POSE = {}
for _pk, _pv in POSTURES.items():
    for _g in ("walk", "run"):
        for _side, _anim in (_pv.get(_g) or {}).items():
            _pose = next((f for f in _pv["faces"] if f.endswith("_" + _side)), None)
            if _pose:
                GAIT_POSE[_anim] = _pose


def bg_color(rgb: np.ndarray) -> np.ndarray:
    """本帧的背景色 = 四边 6px 环的中位。素材规范保证角色不贴边，所以这一圈一定是背景。"""
    ring = np.concatenate([rgb[:6].reshape(-1, 3), rgb[-6:].reshape(-1, 3),
                           rgb[:, :6].reshape(-1, 3), rgb[:, -6:].reshape(-1, 3)])
    return np.median(ring, 0)


def scrub_watermark(rgb: np.ndarray) -> tuple:
    """把水印涂回**本帧背景色**，交给泛洪当背景吃掉。返回 (处理后图像, 清除像素数)。

    只清除**能从框边连通到外部**的文字像素。被毛发包住的文字如果也涂白，
    泛洪从边界走不到它（它四周是前景），结果就是在身体上留下一个不透明的白洞
    ——比原来的白边更难看。宁可留一点淡痕，不打穿主体。

    填色必须是 `bg_color(rgb)` 而不是写死的 255。泛洪现在判的是"离本帧背景色多近"，
    涂成纯白就等于在背景里挖了一块色差 9 级的异物 —— 实测坐下素材右下角因此
    留下一个边缘整齐的纯白矩形（深色桌面上非常刺眼），而旧判据"亮度>235"恰好能吃掉它。
    """
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = (int(w * WM_BOX[0]), int(h * WM_BOX[1]),
                      int(w * WM_BOX[2]), int(h * WM_BOX[3]))
    sub = rgb[y0:y1, x0:x1]
    if sub.size == 0:
        return rgb, 0, np.zeros(rgb.shape[:2], bool)
    lum, sat = sub.mean(2), sub.max(2) - sub.min(2)
    text = (lum >= WM_LUM[0]) & (lum <= WM_LUM[1]) & (sat <= WM_SAT)
    near_bg = (sub.min(2) >= 200) & (sat <= 30)
    labels, _ = ndimage.label(text | near_bg)
    edge = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    edge.discard(0)
    keep = text & np.isin(labels, list(edge))
    if not keep.any():
        return rgb, 0, np.zeros(rgb.shape[:2], bool)
    out = rgb.copy()
    region = out[y0:y1, x0:x1]
    fill = bg_color(rgb).round().astype(np.uint8)
    region[keep] = fill
    painted = np.zeros(rgb.shape[:2], bool)
    painted[y0:y1, x0:x1] = keep
    return out, int(keep.sum()), painted


def background_like(rgb: np.ndarray) -> np.ndarray:
    """哪些像素算背景。

    判据是**离本帧自己的背景色足够近**（`FLOOD_TOL`），不是"亮度高于某个绝对值"。
    这是照菲比 §3.3 的 `FLOOD_TOL` 思路改的，改的直接原因是实测：
    她耳侧蝴蝶结外那圈锯齿光晕就是背景，只是被 h264 压暗到亮度 237（min 通道 234），
    在写死的 `min > BG_MIN(235)` 下**差一级没被抓到**，于是留在前景里，
    深色桌面上就是一圈白毛刺（开发文档 §11.12 记了为它试过的另一条死路）。

    换成相对判据后逐帧实测（610 帧）：
      sitdown      孤岛 8 块 → **0 块**，面积 -7.4%（掉的就是光晕，目视身体完整）
      其余 7 段     面积差 ≥ -0.33%，孤岛 0 → 0        ← 没有任何附带损伤
    她的白毛(248)离背景(246)只有 2 级、同样落在容差内，之所以没被吃掉，
    靠的是泛洪只能沿"与四边连通"的区域走，而白毛被暗描边围住了 ——
    这一点和改之前完全一样，改的只是"哪些像素算可通行的背景"。

    低饱和浅灰单独走 `BG_GRAY_TOL`：渐变暗角和身下投影不是"背景色 ± 8"，
    但它们和背景一样是中性浅色。放宽的只是色差上限，
    "必须从四边连通"和"被暗描边围住的白毛不算"这两条保护一条没动。
    """
    bg = bg_color(rgb)
    diff = np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).max(2)
    span = rgb.max(2).astype(int) - rgb.min(2).astype(int)
    return ((diff < BG_GRAY_TOL) & (span < BG_GRAY_SPAN)) \
        | ((diff < FLOOD_TOL) & (span < BG_SPAN))


def foreground(rgb: np.ndarray) -> np.ndarray:
    """前景掩码。

    背景判定用「边缘种子泛洪」而非色键：角色有白色绒毛，纯色键必然抠穿；
    只有从四边连通的浅色区才算背景，被毛边围住的白色保留。
    再只保留最大连通块（源视频右下角的"豆包AI生成"水印与主体不连通，在此清除）。

    已知残留：她耳侧锯齿蝴蝶结**外侧**的背景凹槽被锯齿自己的描边围住，逃过泛洪，
    深色桌面上会看到一圈白色毛刺（实测 sitdown 每帧多算 2~4%，亮度 237/通道差 6，
    就是被 h264 压暗了 3 级的背景）。试过"拿暗描边当墙再泛洪一次"来收它，
    **代价是劈开 walk4_left 的胸口与前腿**（约 1/3 帧各缺 4~5 万像素），
    收窄成"只砍小而像背景的块"又不成立——她的白毛亮度 248 比背景 240 还高，
    颜色上分不开。整套判据与实测数据见开发文档 §11.12，不要再试第二遍。
    """
    light = background_like(rgb)
    labels, _ = ndimage.label(light)
    seeds = set(np.unique(np.concatenate(
        [labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    seeds.discard(0)
    fg = ~np.isin(labels, list(seeds)) if seeds else np.ones(light.shape, bool)

    labels, n = ndimage.label(fg)
    if n > 1:
        sizes = np.bincount(labels.ravel())
        fg = labels == int(np.argmax(sizes[1:]) + 1)

    # 轮廓内收：h264 在高对比的暗描边外圈会甩出一圈振铃亮斑（mosquito noise）。
    # 泛洪抓不到它们——亮度够但通道极差超标，而且紧贴描边所以属于最大连通块，
    # 结果就是合成到深色桌面上时轮廓外一层锯齿白边。
    # 实测亮边占前景面积：视频 0.91~0.95% vs 静帧 0.09%，内收 2px 后视频降到 0.09~0.12%，
    # 正好回到静帧的干净水平。放在这里而不是 to_canvas 里，是为了让高度/面积/色调
    # 所有测量和渲染都用同一张掩膜，不然归一化会差出这 2px。
    if CHOKE:
        fg = ndimage.binary_erosion(fg, iterations=CHOKE)
    return drop_debris(fg, rgb)


def drop_debris(fg: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """删掉贴在描边【外侧】且面积很小的连通块，那些就是残留的白点点。

    这套画风全身都有粗暗描边，所以拿描边当墙：对 fg & ~墙 做连通标记，
    凡是与背景相邻的块都在描边之外。杂质只有几十像素，而描边一旦有缺口、
    整片身体会连成十几万像素的巨块 —— 所以必须带尺寸上限，
    否则 f080 那种缺口会让身体整体漏掉（实测无上限时面积保留率塌到 15.9%）。
    上限取 DEBRIS_MAX=400：实测杂质最大 83px、漏块 194375px，200~4000 结果一致，不敏感。
    """
    wall = fg & (rgb.mean(2) < OUTLINE_LUM)
    space = fg & ~wall
    labels, n = ndimage.label(space)
    if n == 0:
        return fg
    touches_bg = ndimage.binary_dilation(~fg, iterations=1) & space
    outside = set(np.unique(labels[touches_bg]))
    outside.discard(0)
    if not outside:
        return fg
    sizes = np.bincount(labels.ravel())
    drop = [i for i in outside if sizes[i] <= DEBRIS_MAX]
    if not drop:
        return fg
    kill = np.zeros(n + 1, bool)
    kill[drop] = True
    return fg & ~kill[labels]


def span(mask: np.ndarray, axis: int) -> int:
    """axis=1 → 纵向(高)跨度；axis=0 → 横向(宽)跨度。"""
    counts = mask.astype(int).sum(axis=axis).nonzero()[0]
    return int(counts.max() - counts.min() + 1)


def to_canvas(rgb: np.ndarray, scale: float, fg: np.ndarray | None = None,
              anchor_dx: float | None = None,
              anchor_dy: float | None = None) -> tuple[Image.Image, tuple]:
    """抠图 → 按 scale 缩放 → 底部对齐 BASELINE 贴到统一画布。

    anchor_dx 是**相对默认居中位置的横向偏移量**（画布像素），不是绝对坐标。
    动画序列用它保留左右摇摆：给"该帧 bbox 中心 − 全段中位中心"的缩放值，
    于是平均位置仍然居中、每帧又各自摆动。

    别传绝对中心坐标：`px = anchor − sw/2` 里的 sw 是该帧自己的宽度，
    那样等于把每帧重新居中，摇摆会被完全抵消掉。

    anchor_dy 同理，是**纵向**偏移量（正数往下）。用它把每帧的顶边拉回低频趋势，
    也就是只削掉"逐帧乱跳"那部分，保留步态本身的起伏（见 `bob_dy()`）。
    """
    fg = foreground(rgb) if fg is None else fg
    alpha = ndimage.gaussian_filter(fg.astype(np.float32), FEATHER)
    alpha = np.clip(alpha / max(alpha.max(), 1e-6), 0, 1)

    # 半透明环带的 RGB 一律回填"最近的前景像素颜色"。
    # 源图边缘本来就是深色描边与白底的混合色，视频还叠了 h264 振铃——
    # 直接沿用原色就是一圈近白毛边（实测视频帧环带 15~21% 的像素亮度 >200）。
    # 回填之后环带取的是描边本色，白边从根上消失，且不影响不透明内部。
    nn = ndimage.distance_transform_edt(~fg, return_distances=False, return_indices=True)
    rgb = rgb[tuple(nn)]

    ys, xs = np.where(fg)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    arr = np.dstack([rgb, (alpha * 255).astype(np.uint8)])
    crop = Image.fromarray(arr).crop((x0, y0, x1 + 1, y1 + 1))
    sw, sh = max(1, round((x1 - x0 + 1) * scale)), max(1, round((y1 - y0 + 1) * scale))
    crop = crop.resize((sw, sh), Image.LANCZOS)
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    if sw > CANVAS_W or sh > CANVAS_H - BASELINE:
        print(f"警告：内容 {sw}x{sh} 超出画布 {CANVAS_W}x{CANVAS_H - BASELINE}，需上调画布规格",
              file=sys.stderr)
    px = (CANVAS_W - sw) // 2 + (0 if anchor_dx is None else int(round(anchor_dx)))
    py = CANVAS_H - BASELINE - sh + (0 if anchor_dy is None else int(round(anchor_dy)))
    canvas.paste(crop, (px, py), crop)
    return canvas, (x0, y0, x1, y1)


def bob_dy(tops: list, heads: list, heights: list, scale: float,
           window: int = BOB_SMOOTH, lim: float = BOB_MAX_DX) -> list:
    """纵向去抖：让**头部**沿低频趋势走，削掉逐帧乱跳，保留每步一次的真弹跳。

    三个坑，都是先错过才知道的：

    1. **不能用包围盒顶边当参照。** 跑步时最上面那一行会在头顶、耳朵、甩起来的
       尾巴之间来回切换，它跳 3px 不代表她头动了 3px。用"顶部 35% 那条带的质心"
       才是身体自己。
    2. **不能按"顶边落在 top×s"算偏移。** `to_canvas` 是**底边**对齐基线的，
       内容顶边落在哪由"画布高 − 基线 − 本帧内容高"决定 —— 高度起伏会自己把顶边
       推来推去。第一版就是漏了这项，报告里写着"8.9→3.6"，成品量出来纹丝没动。
    3. **只对位移类动画成立**（见 `BOB_ANIMS`）。坐着/睡着/呼吸时底边对齐就等于
       脚踩在地上，把头部钉到趋势上会让脚离地，还会破坏与静帧的基线对齐。
       走路跑步本来就有一瞬双脚离地，抬一点是物理正确的；补偿量仍然夹在 8 画布px
       （≈1.6 屏幕px）以内。

    做法：先算这一帧头部**当前会落在画布哪儿**（相对底边），再算它"顺着低频趋势
    该落在哪儿"，差值就是补偿量；整段补偿的均值归零，免得她整体上下漂。
    """
    t = np.asarray(tops, float)
    hd = np.asarray(heads, float)
    hh = np.asarray(heights, float)
    pad = window // 2
    trend = np.convolve(np.pad(hd, (pad, pad), mode="edge"),
                        np.ones(window) / window, mode="valid")[:len(hd)]
    pos = (hd - t - hh) * scale                         # 头部相对底边的位置（向上为负）
    want = (trend - float(trend.mean())) * scale        # 只保留趋势相对自身均值的起伏
    dy = want - (pos - float(pos.mean()))
    return list(np.clip(dy, -lim, lim))


ANCHORS = ("cyan", "white", "pink", "dark")


def anchor_means(rgb: np.ndarray, fg: np.ndarray) -> dict:
    """四个锚点区域的平均 RGB —— 调色指纹。

    为什么不用整块前景的 mean/std：那种做法只能对齐"亮度与对比"，对**色相旋转**无能为力。
    实测视频帧的暗描边是红棕色、静帧是藏青，毛色阴影发紫灰，就是典型的色相偏移。
    锚点必须跨明暗三段（青毛=中间调、白肚皮=高光、暗描边=阴影、粉=饱和点缀），
    才能把 3x3 色彩矩阵的 9 个自由度定住。
    """
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    lum = rgb.mean(2)
    seg = {
        "cyan": fg & (g > r + 10) & (b > r + 10) & (g > 120),
        "white": fg & (r > 225) & (g > 225) & (b > 225),
        "pink": fg & (r > 180) & (r - g > 45),
        "dark": fg & (lum < 110),
    }
    return {k: np.array([rgb[..., i][m].mean() for i in range(3)], np.float32)
            for k, m in seg.items() if m.sum() >= 400}


def color_matrix(src: dict, ref: dict, strength: float = TONE_STRENGTH,
                 ridge: float = 0.35) -> np.ndarray:
    """最小二乘拟合 3x3 色彩矩阵，把 src 的各锚点映射到 ref 的各锚点。

    带单位矩阵方向的岭回归：只有 4 个锚点、9 个自由度，不加约束的话
    在锚点色域之外会外推出离谱的值。再按 strength 与单位阵混合，留一点原始风格。
    """
    keys = [k for k in ANCHORS if k in src and k in ref]
    if len(keys) < 3:
        return np.eye(3, dtype=np.float32)
    V = np.stack([src[k] for k in keys])            # (n,3)
    P = np.stack([ref[k] for k in keys])            # (n,3)
    w = np.array([np.linalg.norm(v) + 1e-3 for v in V], np.float32)   # 亮锚点权重高
    A = V * w[:, None]
    B = P * w[:, None]
    M = np.zeros((3, 3), np.float32)
    eye = np.eye(3, dtype=np.float32)
    lam = ridge * float(len(keys))
    for c in range(3):
        G = A.T @ A + lam * np.eye(3, dtype=np.float32)
        y = A.T @ B[:, c] + lam * eye[c]
        M[c] = np.linalg.solve(G, y)
    return eye + (M - eye) * strength


def apply_matrix(rgb: np.ndarray, M: np.ndarray) -> np.ndarray:
    if np.allclose(M, np.eye(3), atol=1e-4):
        return rgb
    flat = rgb.astype(np.float32).reshape(-1, 3) @ M.T
    return np.clip(flat, 0, 255).reshape(rgb.shape).astype(np.uint8)


def cyan_stats(rgb: np.ndarray, fg: np.ndarray) -> tuple:
    """青色毛色均值，跨素材色调一致性探针。"""
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    cyan = fg & (g > r + 10) & (b > r + 10)
    if cyan.sum() < 5000:
        return (0.0, 0.0, 0.0)
    return tuple(float(rgb[..., i][cyan].mean()) for i in range(3))


def fmt(v) -> str:
    return "-" if v is None or len(v) == 0 else "/".join(f"{x:.0f}" for x in v)


def build_poses(report: list) -> dict:
    loaded = {}
    for stem, key in POSE_MAP.items():
        src = pick_asset(SRC / "图片", stem, ".png", "静态姿势")
        rgb = np.array(Image.open(src).convert("RGB"))
        fg = foreground(rgb)
        loaded[key] = (src, rgb, fg, span(fg, 1), span(fg, 0), cyan_stats(rgb, fg))

    ref_anchors = anchor_means(loaded["front"][1], loaded["front"][2])   # 基准 = 正面图
    ref_cyan = loaded["front"][5]
    poses = {}
    for key, (src, rgb, fg, ch, cw, stats) in loaded.items():
        scale = CHAR_H / ch
        M = color_matrix(anchor_means(rgb, fg), ref_anchors)
        img, bbox = to_canvas(apply_matrix(rgb, M), scale, fg)
        out = OUT / "skins" / "default" / "poses" / f"{key}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        report.append(dict(kind="pose", key=key, src=src.name, bbox=bbox,
                           scale=round(scale, 4),
                           tone=f"{fmt(stats)}→{fmt(ref_cyan)}", md5=hashlib.md5(out.read_bytes()).hexdigest()[:8]))
        # content 记录归一化到画布后的实际内容尺寸，供窗口算贴边留白
        poses[key] = dict(file=f"skins/default/poses/{key}.png", k=1.0,
                          content=[round(cw * scale), CHAR_H], canvas=[CANVAS_W, CANVAS_H])
    return poses, (ref_anchors, ref_cyan)


def thumb(path: Path, grid: int = 48) -> np.ndarray:
    """48×48 灰度缩略（铺白底），用来量"两帧看起来差多少"。

    刻意保留 0~255 量纲：turn_geometry 的判据阈值都是按这个量纲定的，
    改成 0~1 会让"回到正面"的绝对门限永远成立，周期直接被误判成半圈。
    """
    a = np.asarray(Image.open(path).convert("RGBA").resize((grid, grid), Image.BOX),
                   np.float32)
    alpha = a[..., 3:4] / 255.0
    rgb = a[..., :3] * alpha + 255.0 * (1.0 - alpha)
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).ravel()


def turn_geometry(frame_files: list, pose_files: dict, pose_keys=None) -> dict:
    """从成品帧反算转身所需的三样东西：周期、各朝向关键帧、累计视觉路径。

    - cycle：视频常常在转完 360° 后还多留一段正面静止帧。这些帧必须排除在循环之外，
      否则"转身"会先顿一下才开始动（菲比文档 §4.1 的"多余画面"就是这么来的）。
    - keys：每个朝向在循环内与标准姿势最接近的那一帧。
    - path：累计相邻帧差异，运行时用来识别采样区间是否落在停滞区（§4.4）。

    `pose_keys` 决定这条转身图覆盖哪些姿势 —— 趴着的 180° 只有三个姿势，
    和站立的四个不是一回事，不能共用一份硬编码列表。
    """
    pose_keys = pose_keys or ("front", "side_left", "back", "side_right")
    th = np.array([thumb(f) for f in frame_files])
    adj = np.abs(np.diff(th, axis=0, append=th[:1])).mean(1)
    d0 = np.abs(th - th[0]).mean(1)
    tol = max(2.0, 0.1 * d0.max())
    later = [i for i in range(len(th) // 2, len(th)) if d0[i] <= tol]
    cycle = int(later[0]) if later else len(th)

    keys = []
    for key in pose_keys:
        ref = thumb(pose_files[key])
        keys.append(int(np.argmin(np.abs(th[:cycle] - ref).mean(1))))
    return dict(cycle=cycle, keys=keys, pose_keys=list(pose_keys),
                path=[round(float(x), 2) for x in np.concatenate([[0.0], np.cumsum(adj[:cycle])])],
                dropped=len(th) - cycle, total=round(float(adj[:cycle].sum()), 1))


def gait_period(masks: list) -> tuple:
    """量步态周期与首尾接缝。返回 (周期帧数, 接缝是相邻帧差的几倍)。

    周期用**签名自相关在 lag≥GAIT_MIN_PERIOD 之后的局部峰**来定。
    不能取全局最大值——相邻帧本来就相似，最短 lag 永远是峰，
    那样任何素材都会报"周期 8 帧"（写这版时第一发就是这么错的）。

    接缝 = 末帧与首帧的距离 ÷ 段内相邻帧差的中位。
    它回答的是"整段播完再接回头，会不会绊一下"。
    """
    n = len(masks)
    if n < GAIT_MIN_PERIOD * 2:
        return 0, 0.0
    sig = np.array([np.concatenate([m.sum(0), m.sum(1)]).astype(np.float32) for m in masks])
    step = np.array([float(np.abs(sig[i + 1] - sig[i]).mean()) for i in range(n - 1)])
    med = float(np.median(step)) or 1e-9
    seam = float(np.abs(sig[0] - sig[-1]).mean()) / med
    s = (sig - sig.mean(0)) / (sig.std(0) + 1e-9)
    ac = np.array([float((s[:n - k] * s[k:]).mean()) for k in range(1, n // 2)])
    peaks = [k for k in range(GAIT_MIN_PERIOD, len(ac) - 1)
             if ac[k] >= ac[k - 1] and ac[k] > ac[k + 1]]
    period = max(peaks, key=lambda k: ac[k]) if peaks else 0
    return period, seam


def gait_window(masks: list, period: int):
    """找一个"整数个步态周期、且首尾接得上"的循环窗口。

    两层搜索：先少丢素材（周期数从多到少），第一个达标的就收手。
    只按最长的那个整数周期截是不够的 —— 四足走截到 4 个周期(88 帧)接缝仍有
    9.5× 相邻帧差，观感就是爬着爬着"咯噔"一下；退到 2 个周期(44 帧)才降到 2.05×。
    反过来也不该一味求短：向左走整段接缝本来只有 1.5×，为凑整数周期丢掉 21 帧不值。

    返回 (lo, hi, 接缝倍数)；量不出周期、或怎么截都达标不了时返回 None（保持整段）。
    """
    n = len(masks)
    if period < GAIT_MIN_PERIOD or n < period * 2:
        return None
    sig = np.array([np.concatenate([m.sum(0), m.sum(1)]).astype(np.float32) for m in masks])
    step = np.array([float(np.abs(sig[i + 1] - sig[i]).mean()) for i in range(n - 1)])
    med = float(np.median(step)) or 1e-9
    for k in range(n // period, 1, -1):
        length = k * period
        if length < n * 0.4:            # 丢掉的比留下的还多，宁可不循环
            break
        best = min(((float(np.abs(sig[s + length - 1] - sig[s]).mean()) / med, s)
                    for s in range(n - length + 1)), key=lambda t: (round(t[0], 3), t[1]))
        if best[0] <= GAIT_SEAM_OK:
            return best[1], best[1] + length, round(best[0], 2)
    return None


def hold_window(n: int, period: int):
    """姿态驻留素材的来回播窗口：返回 (lo, hi, stride)。

    这类素材整段都是微动，所以窗口取**一个自相关周期**而不是整段 —— 来回播是
    "正放一遍再倒放一遍"，跨两个周期去镜像会把她"呼到一半突然吸回去"的节奏拉长一倍。
    步长挑到让"来回一轮"落在 2.5~4.5 秒：坐下那段实测 2.72s 就是这个量级，
    再快是发抖，再慢就看不出她在动。

    量不出周期（或素材短到不够一个周期）就退回整段，步长仍按时长挑。
    """
    hi = period if period and 8 <= period <= n else n
    m = max(2, hi)
    for stride in (1, 2, 3, 4):
        if (2 * m - 2) / stride * FPS_MS / 1000 <= 4.5:
            break
    return 0, hi, stride


def gait_phase(frame_files: list, pose_file: Path) -> int:
    """步态动画该从哪一帧开始播 = 整段里最像"起步前那个静态朝向"的一帧。

    素材的第 0 帧不保证是"标准步态相位"。实测 向右跑 第 0 帧与 右面 姿势图差 1.192，
    而第 22 帧只差 0.882 —— 从第 0 帧起步，她会在起步那一拍"猛地一歪"。
    走路那两段差得少（0.093 / 0.219），但同样按最贴合帧起更稳。
    """
    th = np.array([thumb(f) for f in frame_files])
    return int(np.argmin(np.abs(th - thumb(pose_file)).mean(1)))


def sit_segments(frame_files: list, stand_pose: Path, sit_pose: Path) -> dict:
    """把"坐下"这类一次性过渡视频切成 进入起点 / 到达点 / 待机段，全部以素材自带的
    静姿图为**语义锚点**：`front.png` = 她站着长什么样，`sit_front.png` = 她坐着长什么样。

    为什么不用内容高判（上一版就是）：她"把两条腿往外撇"的时候身高几乎不变 ——
    高度对腿部动作完全瞎。实测驻留段因此选中了撇腿那 21 帧，窗口内平均每帧变化
    1.19 灰阶，而坐定之后的安静段只有 0.21~0.34，差 3.5~5.7 倍；观感就是
    "坐着坐着自己疯狂蹬腿"，真正该循环的结尾待机段一帧都没用上。用户原话：
    "腿动的太厉害了……前面多了一点，后面好像少了一点"。
    素材里明摆着给了"坐着待机长什么样"的静姿图，就该拿它当锚点。

    为什么是"定长窗口取最早达标起点"而不是"最长的连续平静段"（第一版这里是这么找的，
    量出来只有 10 帧、来回一轮 1.44s，比呼吸快一倍，还是抖）：素材里根本没有 20 帧
    不断流的绝对平静段，"最长连续段"优化的是**长度**，找到一段短的就已经停手；
    我们要的是**一段够长的、静到接近最静窗口的**画面，所以长度由 HOLD_LEN 定，
    起点在满足静度门限的前提下尽量早 —— 早一点进待机，她就少等一会儿。

    返回：
      enter_at  进入段从这一帧开始播（最像站立的那一帧；它之前是她站着没事干，
                而且比标准站姿图大 3.3%，一开场就先"长高"一下 —— 就是"前面多了"）
      arrive_at 到达坐姿的那一帧（此后一直在坐姿带内）—— 只是测量，运行时不看它
      settle_at 过渡段交给驻留段的那一帧，等于 loop 起点
      loop      运行时要循环播的待机窗口
    """
    th = np.array([thumb(f) for f in frame_files])
    n = len(th)
    d_sit = np.abs(th - thumb(sit_pose)).mean(1)
    d_stand = np.abs(th - thumb(stand_pose)).mean(1)
    move = np.concatenate([[0.0], np.abs(np.diff(th, axis=0)).mean(1)])

    band = float(d_sit.min()) * 1.02
    arrive = next((i for i in range(n) if bool((d_sit[i:] <= band).all())), n - 1)
    arrive = max(1, arrive)
    enter_at = int(np.argmin(d_stand[:arrive]))

    L = min(HOLD_LEN, n - arrive)
    if L < 4:
        # 坐定之后没剩几帧可循环：给单帧窗口，运行时就地停住，而不是把抖腿圈成循环
        lo, hi = arrive, arrive + 1
    else:
        starts = list(range(arrive, n - L + 1))
        means = np.array([move[s:s + L].mean() for s in starts])
        limit = float(means.min()) * HOLD_TOL
        lo = next((s for s, m in zip(starts, means) if m <= limit), starts[int(np.argmin(means))])
        hi = lo + L
    return dict(enter_at=enter_at, arrive_at=arrive, settle_at=lo, loop=[lo, hi],
                d_stand=round(float(d_stand[enter_at]), 3),
                d_sit=round(float(d_sit[lo]), 3),
                move_hold=round(float(move[lo:hi].mean()), 3),
                move_old=round(float(move[enter_at:lo].mean()), 3))


def erase_protrusions(key: str, out_dir: Path, masks: list) -> int:
    """把瑕疵帧的轮廓裁回时位中位形状。返回被裁的帧数；未配置的动画直接返回 0。

    masks 必须是**本次刚生成、尚未擦除**的轮廓：若从磁盘重读，第二次跑管线时
    已裁过的帧会拉低中位数，把凸起"洗白"成正常形状，判据随之失效。

    两条来自实测的硬约束，动之前先读：
    1. **只在 roi 内动手**。中位形状是"整段共用"的，走路时手脚本来就到处伸，
       全屏裁会把合法摆臂一起削掉；roi 之外一律不碰。
    2. **裁完必须硬切，不能羽化**。上一版用 `min(alpha, gaussian(keep))`，
       在 keep 外留了一圈 alpha 20~128 的半透明青边（每帧约 70 px），
       观感就是"还是没扣干净"。keep 线在真实身体轮廓之外 3 px，硬切不会啃到身子。
    """
    rule = ANIM_ERASE.get(key)
    if not rule:
        return 0
    n = len(masks)
    med = np.stack(masks).mean(0) > 0.5
    keep = ndimage.binary_dilation(med, iterations=rule["dilate"])
    x0, y0, x1, y1 = rule["roi"]
    zone = np.zeros(keep.shape, dtype=bool)
    zone[y0:y1, x0:x1] = True
    drop = zone & ~keep                       # 允许动手的那一圈像素
    hit = 0
    for i in range(n):
        if (i % rule["period"]) not in rule["phases"]:
            continue
        path = out_dir / f"frame_{i:03d}.png"
        a = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32)
        cut = (a[..., 3] > 8) & drop          # 半透明裙边也要数，否则会被下面跳过
        if cut.sum() < 20:                    # 这一帧其实没有凸起，别白动它
            continue
        a[..., 3][cut] = 0
        Image.fromarray(a.astype(np.uint8)).save(path)
        hit += 1
    return hit


def build_anims(report: list, ref: tuple, poses: dict) -> dict:
    anims = {}
    for stem, key in VIDEO_MAP.items():
        src = pick_asset(SRC / "视频", stem, ".mp4", "动画")
        if src is None:
            continue
        tmp = Path(tempfile.mkdtemp(prefix="mint_frames_"))
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                            str(tmp / "%04d.png")], check=True)
            frames = sorted(tmp.glob("*.png"))
            raw_alpha = []          # 未擦除的原始轮廓，供 erase_protrusions 求中位
            scrubbed = [scrub_watermark(np.array(Image.open(f).convert("RGB"))) for f in frames]
            rgbs = [arr for arr, _, _ in scrubbed]
            painted = [p for _, _, p in scrubbed]
            counts = [c for _, c, _ in scrubbed]
            report.append(dict(kind="anim", key=f"{key} 水印清除", src=src.name,
                               bbox=(min(counts), int(np.median(counts)), max(counts)),
                               scale=f"{sum(1 for c in counts if c)}/{len(counts)} 帧",
                               tone="", md5=""))
            fgs = [foreground(rgb) for rgb in rgbs]
            # 自检：被涂掉的水印像素，最后必须**真的被泛洪当背景吃掉**。
            # 残留（涂了却仍属前景）意味着在她身上打穿了一个不透明的洞 ——
            # 宁可留一点淡痕也不要洞，所以这里量的是"涂完仍然是前景"的那部分。
            # 这条锁抓的是"两个各自正确的模块配套错了"这类错误：
            # 泛洪判据换成相对容差后，涂死的纯白离背景 9 级、落在容差之外，
            # 坐下素材右下角就留下一个边缘整齐的纯白矩形。
            scrub_leak = max((int((p & fg).sum()) for p, fg in zip(painted, fgs)), default=0)
            report.append(dict(kind="anim", key=f"{key} 涂白回吃", src="",
                               bbox=f"最大残留 {scrub_leak}px",
                               scale="" if scrub_leak <= SCRUB_LEAK_OK else "!! 涂的色没被当背景",
                               tone="", md5=""))
            # 源视频把角色**切到画面外**：这类缺陷管线修不了（像素根本不存在），
            # 只能回头要素材，所以在这里量出来写进清单，让冒烟能锁住它。
            # 两个条件都要满足，缺一不可：
            #   贴边   —— 前景碰到源画面外沿 8px 带（必须宽过 CHOKE，否则被内收吃掉）
            #   掉面积 —— 该帧前景比整段中位少 8% 以上
            # 只贴边不掉面积的是"尾巴快扫到画面边"（趴着转180、向右走_四足 实测如此，
            # 逐帧目视完全干净），把它一起报出来就是狼来了。
            ring = [int(f[:SRC_EDGE_PX].sum() + f[-SRC_EDGE_PX:].sum()
                        + f[:, :SRC_EDGE_PX].sum() + f[:, -SRC_EDGE_PX:].sum()) for f in fgs]
            areas = [int(f.sum()) for f in fgs]
            med_a = float(np.median(areas)) or 1.0
            clipped = [i for i in range(len(fgs))
                       if ring[i] > 0 and areas[i] < med_a * (1 - SRC_CLIP_DROP)]
            report.append(dict(
                kind="anim", key=f"{key} 源画面裁切", src=src.name,
                bbox=f"贴边 {sum(1 for v in ring if v)} 帧 / 其中真被切掉 {len(clipped)} 帧",
                scale=f"最轻 {min(areas) / med_a * 100:.0f}%中位面积" if clipped else "",
                tone=f"首帧 #{clipped[0]}" if clipped else "", md5=""))
            heights = [span(fg, 1) for fg in fgs]
            rows = [np.where(fg.any(1))[0] for fg in fgs]
            bottoms = [int(r.max()) for r in rows]
            cols = [np.where(fg.any(0))[0] for fg in fgs]
            cxs = [(c.min() + c.max()) / 2 for c in cols]
            widths = [int(c.max() - c.min() + 1) for c in cols]

            def fit(unit_anchors: list) -> float:
                """高度与宽度双约束下的共用缩放。

                只按身高归一的话，横躺的睡姿会被放大到 995~1094px 宽，直接超出
                896 的画布 —— 两端的头和尾巴被切掉，而且 `to_canvas` 只在 stderr
                喊一行警告，清单照写。宽度约束按"每帧半宽 + 该帧最大横移量"算，
                这样 sequence 的摇摆和 head 的头部锚定也不会把内容推出画布。
                """
                need = max(w / 2 + abs(a) for w, a in zip(widths, unit_anchors))
                return min(CHAR_H / float(np.mean(heights)),
                           (CONTENT_MAX_W / 2) / need)

            def report_cap(key: str, s: float) -> None:
                s_h = CHAR_H / float(np.mean(heights))
                if s < s_h - 1e-6:
                    report.append(dict(
                        kind="anim", key=f"{key} 宽度封顶", src="",
                        bbox=f"最宽 {max(widths)}px 按身高归一需 {max(widths) * s_h:.0f}px",
                        scale=f"{s:.3f}（身高归一 {s_h:.3f}）", tone="", md5=""))

            mode = ANIM_MODE.get(key, "per_frame")

            def head_band():
                """顶部 35% 那一条带的质心（源 px）。包围盒顶边不能当参照：
                跑步时最上面那行会在头顶、耳朵、甩起来的尾巴之间来回切换。"""
                hx, hy = [], []
                for fg, r in zip(fgs, rows):
                    y0, y1 = int(r.min()), int(r.max())
                    cut = y0 + max(1, int((y1 - y0 + 1) * 0.35))
                    band = fg.copy()
                    band[cut:] = False
                    yy, xx = np.where(band)
                    hx.append(float(xx.mean()) if len(xx) else 0.0)
                    hy.append(float(yy.mean()) if len(yy) else float(y0))
                return hx, hy

            hc = head_band() if (mode == "head" or key in BOB_ANIMS) else ([], [])
            if mode == "per_frame":
                # 逐帧把内容高钉死到 CHAR_H，不共用整段缩放。
                # 原地转身的真实身高是不变的，素材里 7.3% 的高低起伏纯属生成噪声；
                # 不压平就是"颠簸"，压平后每一帧（含四个关键帧）才和静态姿势同尺寸，
                # 否则转身收尾会从 ~620 弹到 660。底边与水平中心实测本来就稳（std 0.4）。
                scales = [min(CHAR_H / h, CONTENT_MAX_W / w)
                          for h, w in zip(heights, widths)]
                anchors = [None] * len(frames)
                if any(CHAR_H / h > CONTENT_MAX_W / w for h, w in zip(heights, widths)):
                    report.append(dict(kind="anim", key=f"{key} 宽度封顶", src="",
                                       bbox=f"最宽 {max(widths)}px", scale="逐帧",
                                       tone="", md5=""))
            elif mode == "recenter":
                # 两处同样的毛病：素材里的身体在画面里自己平移，跟着它走就是"滑/晃"。
                #   四足走  向左走_四足漂移 69px（现有走路只有 4~6px），不抵消就是爬着滑
                #   趴睡    身体漂移 8px，躺在地上还左右挪，一眼假
                # 所以逐帧把水平中心钉回中位数，但**共用整段缩放** ——
                # 爬行时的高度起伏是四条腿交替的信号，不能压平。
                s = fit([0.0] * len(frames))
                report_cap(key, s)
                med = float(np.median(cxs))
                scales = [s] * len(frames)
                anchors = [0.0] * len(frames)
                report.append(dict(kind="anim", key=f"{key} 平移抵消", src="",
                                   bbox=f"水平漂移 {max(cxs)-min(cxs):.0f}px 已逐帧居中抵消",
                                   scale="", tone="", md5=""))
            elif mode == "head":
                # 拥抱那 7px 的包围盒横移全来自手臂开合，身体其实没动（头部只动 1.1px）。
                # 先按 sequence 算锚点，再减掉头部自身的位移：头部钉住，手照样抱。
                # 这样既不压平拥抱动作，也不会让整只猫跟着手一起左右抖。
                med = float(np.median(cxs))
                heads = hc[0]
                med_h = float(np.median(heads))
                unit = [(cx - med) - (hx - med_h) for cx, hx in zip(cxs, heads)]
                s = fit(unit)
                report_cap(key, s)
                scales = [s] * len(frames)
                anchors = [a * s for a in unit]
                report.append(dict(kind="anim", key=f"{key} 头部锚定", src="",
                                   bbox=f"包围盒中心漂移 {max(cxs)-min(cxs):.0f}px / "
                                        f"头部 {max(heads)-min(heads):.1f}px",
                                   scale="", tone="", md5=""))
            else:
                # sequence：整段共用一个缩放（按平均高）+ 每帧一个相对偏移量。
                # 静立的 1.7% 高度起伏是呼吸、±3.6px 横向位移是摇摆，
                # 逐帧归一会把呼吸压平、逐帧居中会把摇摆抵消，动画就白做了。
                # 偏移量以全段中位为中心，所以平均位置仍与静帧/转身帧重合，
                # 不会整体左右偏出去（偏出去会顶到画布右缘，把尾巴切掉）。
                med = float(np.median(cxs))
                s = fit([cx - med for cx in cxs])
                report_cap(key, s)
                scales = [s] * len(frames)
                anchors = [(cx - med) * s for cx in cxs]
            ref_anchors, ref_cyan = ref
            # 纵向去抖：只给位移类动画做（理由见 BOB_ANIMS 与 bob_dy）。
            tops = [int(r.min()) for r in rows]
            if key not in BOB_ANIMS:
                dyss = [None] * len(frames)
            else:
                dyss = bob_dy(tops, hc[1], heights, scales[0])
                s = scales[0]
                # 与 bob_dy 内部同一个式子：底边钉死时，头落在"−内容高 + 头在盒内的相对高"
                pos = (np.asarray(hc[1]) - np.asarray(tops) - np.asarray(heights, float)) * s
                after = pos + np.asarray(dyss)
                st = FRAME_STRIDE
                before = float(np.abs(np.diff(pos))[::st].max())
                gain = float(np.abs(np.diff(after))[::st].max())
                report.append(dict(kind="anim", key=f"{key} 纵向去抖", src="",
                                   bbox=f"头部相邻显示帧跳 {before:.1f}→{gain:.1f} 画布px",
                                   scale=f"窗口{BOB_SMOOTH}帧",
                                   tone=f"最大补偿 {np.abs(dyss).max():.1f}px",
                                   md5=""))
            out_dir = OUT / "skins" / "default" / "anim" / key
            shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            drift = []
            for i, (rgb, fg, f) in enumerate(zip(rgbs, fgs, frames)):
                fixed = apply_matrix(rgb, color_matrix(anchor_means(rgb, fg), ref_anchors))
                drift.append([float(v) for v in anchor_means(fixed, fg).get("cyan", [0, 0, 0])])
                img, bbox = to_canvas(fixed, scales[i], fg, anchors[i], dyss[i])
                raw_alpha.append(np.asarray(img)[..., 3] > 128)
                img.save(out_dir / f"frame_{i:03d}.png")
                if i % 30 == 0:
                    report.append(dict(kind="anim", key=f"{key}[{i:03d}]", src=f.name,
                                       bbox=bbox, scale=round(scales[i], 4),
                                       tone=f"{fmt(cyan_stats(rgb, fg))}→{fmt(ref_cyan)}",
                                       md5=""))
            d = np.array(drift)
            report.append(dict(kind="anim", key=f"{key} 校正后前景均值漂移", src="",
                               bbox=[f"{d[:,j].mean():.1f}±{d[:,j].std():.1f}" for j in range(3)],
                               scale="", tone="", md5=""))
            swing = 100 * (max(heights) - min(heights)) / float(np.mean(heights))
            sway = (max(cxs) - min(cxs)) / 2 * scales[0]
            report.append(dict(kind="anim", key=f"{key} 共{len(frames)}帧", src=src.name,
                               bbox=(min(heights), max(heights)),
                               scale=f"{mode} 高起伏{swing:.1f}% 摆幅±{sway:.1f}px",
                               tone="", md5=""))
            erased = erase_protrusions(key, out_dir, raw_alpha)
            anims[key] = dict(dir=f"skins/default/anim/{key}", frames=len(frames),
                              fps=24, stride=FRAME_STRIDE, mode=mode,
                              **({"erased_frames": erased} if erased else {}),
                              **({"scrub_leak": scrub_leak} if scrub_leak else {}),
                              **({"src_clipped": len(clipped), "clip_at": clipped[0]}
                                 if clipped else {}))
            if key in GAIT_ANIMS:
                p, seam = gait_period(fgs)
                # **接缝已经够小就别动。** "截到整数个周期"只是"让首尾接得上"的代理手段，
                # 真正要看的是首尾那一接的画面差。实测：向右走整段接缝只有 0.65×，
                # 硬截成 5 个周期反而升到 1.85× —— 为了一个代理指标把更好的结果换掉了。
                win = gait_window(fgs, p) if seam > GAIT_SEAM_OK else None
                lo, hi = (win[0], win[1]) if win else (0, len(frames))
                files = [out_dir / f"frame_{i:03d}.png" for i in range(lo, hi)]
                ph0 = lo
                if key in GAIT_POSE and GAIT_POSE[key] in poses:
                    # 起步相位只能在**循环窗口内**挑：窗口外的帧运行时根本到不了，
                    # 拿整段的最优帧当起点会让她第一帧就跳到窗口另一头。
                    ph0 = lo + gait_phase(files, OUT / poses[GAIT_POSE[key]]["file"])
                anims[key]["gait"] = dict(period=p, seam=round(seam, 2), loop=[lo, hi],
                                          loop_seam=(win[2] if win else round(seam, 2)),
                                          phase0=ph0)
                # "没截"和"截不动"是两件事，以前都印成"整段可循环"：
                # 向左走_四足 量不出周期（p=0）→ gait_window 返回 None → 窗口=整段，
                # 于是这张**接缝 29.3×、24 帧被源画面切掉**的废素材在报告里写着
                # "整段可循环"。报告说假话比没报告更糟，这里分成三档。
                if win:
                    verdict = f"循环 [{lo},{hi})"
                    why = (f"截到整数周期并挑接缝最小的起点，丢掉 {len(frames) - (hi - lo)} 帧")
                elif seam <= GAIT_SEAM_OK:
                    verdict, why = "整段可循环", ""
                else:
                    verdict = "!! 量不出周期，整段照播"
                    why = (f"每圈脚位不重合（p={p}），截任何前缀都接不上；"
                           f"运行时表现为循环处硬跳，只能重做素材")
                report.append(dict(
                    kind="anim", key=f"{key} 步态相位", src="",
                    bbox=f"周期{p}帧 整段接缝{seam:.1f}× 窗口接缝{anims[key]['gait']['loop_seam']}×"
                         f" 起步f{ph0}",
                    scale=verdict, tone=why, md5=""))
            if key in SETTLE_SEARCH:
                owner = next((pk for pk, pv in POSTURES.items() if pv.get("enter") == key), None)
                if owner:
                    # 两端锚点就用素材自己给的静姿图：从"基础姿态长什么样"坐到"该姿态长什么样"
                    from_pose = POSTURES["standing"]["faces"][0]
                    to_pose = POSTURES[owner]["faces"][0]
                    seg = sit_segments(
                        [out_dir / f"frame_{i:03d}.png" for i in range(len(frames))],
                        OUT / poses[from_pose]["file"], OUT / poses[to_pose]["file"])
                    anims[key].update(settle_at=seg["settle_at"], loop=seg["loop"],
                                      enter_at=seg["enter_at"],
                                      hold_move=seg["move_hold"], enter_move=seg["move_old"])
                    report.append(dict(kind="anim", key=f"{key} 三段切分", src="",
                                       bbox=f"进入 f{seg['enter_at']}→{seg['settle_at']}，"
                                            f"待机循环 {seg['loop']}",
                                       scale=f"坐到位f{seg['arrive_at']} 锚点距 站{seg['d_stand']} 坐{seg['d_sit']}",
                                       tone=f"待机段逐帧变化 {seg['move_hold']} "
                                            f"（进入段 {seg['move_old']}）", md5=""))
            if key in HOLD_LOOPS:
                per, _ = gait_period(fgs)
                lo, hi, lst = hold_window(len(frames), per)
                anims[key]["loop"] = [lo, hi]
                anims[key]["loop_stride"] = lst
                report.append(dict(kind="anim", key=f"{key} 驻留窗口", src="",
                                   bbox=f"[{lo},{hi}) 步长{lst}",
                                   scale=f"周期{per}帧",
                                   tone=f"来回一轮 {(2 * (hi - lo) - 2) / lst * FPS_MS / 1000:.2f}s",
                                   md5=""))
            if key in TURN_POSES and poses:
                geo = turn_geometry([out_dir / f"frame_{i:03d}.png" for i in range(len(frames))],
                                    {k: OUT / v["file"] for k, v in poses.items()},
                                    TURN_POSES[key])
                anims[key].update(cycle=geo["cycle"], keys=geo["keys"], path=geo["path"],
                                  faces=geo["pose_keys"],
                                  **({"linear": True} if key in LINEAR_TURNS else {}))
                report.append(dict(kind="turn", key=f"geometry {key}",
                                   src=f"丢弃尾部 {geo['dropped']} 帧",
                                   bbox=geo["keys"], scale=geo["total"],
                                   tone=f"cycle={geo['cycle']} "
                                        f"{dict(zip(geo['pose_keys'], geo['keys']))}", md5=""))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return anims


def export_poses_from_anims(poses: dict, anims: dict, report: list) -> None:
    """把 `POSE_FROM_VIDEO` 指定的那一帧原样复制成静姿图并登记进 poses。

    放在清单写盘之前、`--only` 合并之后：这样 `--only poses` 与 `--only anims`
    两种半重建都能拿到图（帧文件在磁盘上，两种模式下都存在）。
    """
    for key, (anim_name, i) in POSE_FROM_VIDEO.items():
        a = anims.get(anim_name)
        if not a:
            continue                      # 这一半没重建、旧清单里也没有：交给继承那条路
        src = OUT / a["dir"] / f"frame_{min(i, int(a['frames']) - 1):03d}.png"
        if not src.exists():
            sys.exit(f"配置错误：POSE_FROM_VIDEO['{key}'] 要取 {src.name}，但它不存在")
        alpha = np.asarray(Image.open(src).convert("RGBA"))[..., 3]
        ys, xs = np.where(alpha > 8)
        dst = OUT / "skins" / "default" / "poses" / f"{key}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        # content 用**实测** alpha>8 的包围盒，不照抄别的姿势那样写死 CHAR_H：
        # 这张图的全部意义就是"和驻留素材那一帧一模一样"，量出来是多少就是多少。
        poses[key] = dict(file=f"skins/default/poses/{key}.png", k=1.0,
                          content=[int(xs.max() - xs.min() + 1),
                                   int(ys.max() - ys.min() + 1)],
                          canvas=[CANVAS_W, CANVAS_H],
                          derived_from=f"{anim_name}#{i}")
        report.append(dict(kind="pose", key=f"{key}(取自{anim_name}#{i})", src=src.name,
                           bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                           scale="copy", tone="", md5=hashlib.md5(dst.read_bytes()).hexdigest()[:8]))


def check_registry() -> None:
    """开跑前核对两张表互相咬合。错了就直接退出，而不是产出一份"能跑但转错方向"的清单。"""
    for turn, poses in TURN_POSES.items():
        owners = [pk for pk, pv in POSTURES.items() if pv.get("turn") == turn]
        if not owners:
            sys.exit(f"配置错误：TURN_POSES['{turn}'] 没有被任何姿态引用")
        for pk in owners:
            if POSTURES[pk]["faces"] != poses:
                sys.exit(f"配置错误：TURN_POSES['{turn}']={poses} 与姿态 '{pk}' 的 "
                         f"faces={POSTURES[pk]['faces']} 顺序不一致。"
                         f"运行时按 facing 索引 keys，这会让她转错方向且不报错。")
    for g in GATED:
        if g not in POSE_MAP.values() and g not in VIDEO_MAP.values():
            sys.exit(f"配置错误：GATED 里的 '{g}' 既不是姿势键也不是动画键")
    have = set(POSE_MAP.values()) | set(POSE_FROM_VIDEO)
    for pk, pv in POSTURES.items():
        for f in pv["faces"]:
            if f not in have:
                sys.exit(f"配置错误：姿态 '{pk}' 的朝向图 '{f}' 没有任何来源"
                         f"（既不在 POSE_MAP 也不在 POSE_FROM_VIDEO）—— "
                         f"运行时 current_pose() 会查不到图")
    for key, (anim_name, _i) in POSE_FROM_VIDEO.items():
        if anim_name not in VIDEO_MAP.values():
            sys.exit(f"配置错误：POSE_FROM_VIDEO['{key}'] 指向不存在的动画 '{anim_name}'")
    # idle 可以是串（所有朝向共用）或与 faces 等长的列表（每朝向一段，None=停静帧）。
    # 列表长度写错不会当场报错，运行时会按 facing 索引到错位的那一段。
    for pk, pv in POSTURES.items():
        idle = pv.get("idle")
        if isinstance(idle, (list, tuple)):
            if len(idle) != len(pv["faces"]):
                sys.exit(f"配置错误：姿态 '{pk}' 的 idle 有 {len(idle)} 项，"
                         f"faces 有 {len(pv['faces'])} 项 —— 必须一一对应")
            names = [x for x in idle if x]
        else:
            names = [idle] if idle else []
        for n in names:
            if n not in VIDEO_MAP.values():
                sys.exit(f"配置错误：姿态 '{pk}' 的 idle '{n}' 不是已登记的视频动画")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["poses", "anims", "all"], default="all")
    ap.add_argument("--report", action="store_true", help="只诊断不写盘")
    args = ap.parse_args()
    check_registry()

    report: list = []
    poses, ref = {}, (None, CYAN_REF)
    if args.only in ("poses", "all"):
        poses, ref = build_poses(report)
    elif (OUT / "manifest.json").exists():      # --only anims 时复用已有清单，别重新解一遍色彩基准
        raw = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        poses = raw["skins"]["default"]["poses"]
        if raw.get("tone_ref"):
            ref = ({k: np.array(v, np.float32) for k, v in raw["tone_ref"].items()},
                   tuple(raw.get("cyan_ref", CYAN_REF)))
    anims = build_anims(report, ref, poses) if args.only in ("anims", "all") else {}

    def dump_report():
        print(f"{'kind':5} {'key':16} {'bbox':28} {'scale':7} {'tone(RGB)':30} md5")
        for r in report:
            print(f"{r['kind']:5} {r['key']:16} {str(r['bbox']):28} {r['scale']:<7} "
                  f"{r['tone']:30} {r['md5']}")

    if args.report:
        dump_report()
        return
    # 只重建一半时必须把另一半从旧清单里搬过来，否则 --only poses 会把
    # turn / idle_front 的登记整个抹掉，桌宠直接退回静态姿势且没有任何报错。
    mpath = OUT / "manifest.json"
    prev = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else None
    if prev:
        prev_skin = prev["skins"]["default"]
        if args.only == "poses":
            anims = prev_skin.get("anims", {})
        elif args.only == "anims":
            poses = prev_skin.get("poses", {})
        if not ref[0]:
            ref = ({k: np.array(v, np.float32) for k, v in prev["tone_ref"].items()},
                   tuple(prev.get("cyan_ref", CYAN_REF)))
    export_poses_from_anims(poses, anims, report)
    # 改掉 VIDEO_MAP 的键（这次 prone_sleep → prone_idle，另加 doze/sleep_left/sleep_right）
    # 之后，旧名字的帧目录会永远留在 assets 里当孤儿：清单不引用它，
    # 但 `--only anims` 也不会删它，看目录的人会以为还有这条动画。
    # 清单是唯一权威，所以按清单裁剪。
    keep = {k for k, a in anims.items() if (OUT / a["dir"]).is_dir()}
    anim_root = OUT / "skins" / "default" / "anim"
    if anim_root.is_dir():
        for stale in anim_root.iterdir():
            if stale.is_dir() and stale.name not in keep:
                shutil.rmtree(stale)
                report.append(dict(kind="anim", key=f"清理孤儿目录 {stale.name}", src="",
                                   bbox="", scale="", tone="", md5=""))
    manifest = dict(
        canvas=dict(w=CANVAS_W, h=CANVAS_H, char_h=CHAR_H, baseline=BASELINE),
        fps_ms=FPS_MS,
        tone_ref={k: [round(float(x), 1) for x in v] for k, v in ref[0].items()},
        cyan_ref=[round(float(x), 1) for x in ref[1]],
        faces=POSTURES["standing"]["faces"],        # 旧字段保留：站立姿态的朝向顺序
        postures=POSTURES,
        gated=GATED,
        skins={"default": dict(label="薄荷（猫猫）", poses=poses, anims=anims, variants={})},
    )
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {OUT / 'manifest.json'}")
    # 诊断表放在清单之后：它只是给人看的，不该有能力把产品留在半成品状态
    dump_report()


if __name__ == "__main__":
    main()
