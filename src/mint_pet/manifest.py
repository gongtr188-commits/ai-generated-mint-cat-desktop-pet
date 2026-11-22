"""素材清单：皮肤 / 变体 / 姿势解析。

程序不认识文件名，只读 assets/manifest.json。加素材 = 跑一次 build_assets.py，
清单自动更新，代码零改动。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[2] / "assets"


@dataclass(frozen=True)
class Pose:
    key: str
    path: Path
    k: float = 1.0
    content: tuple = ()      # 归一化后的内容宽高（画布像素）


@dataclass(frozen=True)
class Canvas:
    w: int
    h: int
    char_h: int
    baseline: int


@dataclass(frozen=True)
class Skin:
    id: str
    label: str
    poses: dict
    anims: dict
    variants: dict


class Manifest:
    def __init__(self, path: Path = ASSETS / "manifest.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.root = path.parent
        self.canvas = Canvas(**raw["canvas"])
        self.fps_ms = raw.get("fps_ms", 80)
        self.faces = raw["faces"]
        # 姿态 × 朝向 的注册表。facing 只在同一姿态内有意义，所以转身图也挂在姿态上。
        self.postures = raw.get("postures", {})
        # 清单里的 `gated` 是管线留给报告/审计的溯源，运行时**不读**：
        # 趴着整族现在由 `stage.posture_allowed()` 一个入口守卫，
        # 在这里再存一份只会变成第二个说法。
        self.skins = {
            sid: Skin(sid, s["label"],
                      {k: Pose(k, self.root / v["file"], v.get("k", 1.0),
                               tuple(v.get("content", ())))
                       for k, v in s["poses"].items()},
                      {k: dict(a, name=k) for k, a in s.get("anims", {}).items()},
                      s.get("variants", {}))
            for sid, s in raw["skins"].items()
        }
        self.skin_id = "default" if "default" in self.skins else next(iter(self.skins))

    @property
    def skin(self) -> Skin:
        return self.skins[self.skin_id]

    def pose(self, key: str, variant: str | None = None) -> Pose:
        """解析姿势。

        变体缺素材时一律回退基准同名姿势，再回退 front——菲比项目里"无帽状态冒出帽子"
        反复返工后才加的兜底规则，这里从一开始就写死。
        """
        skin = self.skin
        if variant:
            override = skin.variants.get(variant, {}).get("poses", {}).get(key)
            if override:
                return Pose(key, self.root / override, 1.0)
        if key in skin.poses:
            return skin.poses[key]
        return skin.poses["front"]

    def pose_for_face(self, facing: int, variant: str | None = None) -> Pose:
        return self.pose(self.faces[facing % len(self.faces)], variant)

    # ── 姿态 ────────────────────────────────────────────────────────────
    def posture_faces(self, key: str) -> list:
        """某个姿态下的朝向姿势名列表。坐姿只有一个，趴姿有三个。"""
        p = self.postures.get(key)
        return list(p["faces"]) if p else list(self.faces)

    def pose_path(self, pose_key: str) -> Path:
        return self.pose(pose_key).path

    def posture_turn(self, key: str) -> str | None:
        p = self.postures.get(key)
        return p.get("turn") if p else None

    def posture_enter(self, key: str) -> str | None:
        """进入该姿态的一次性过渡动画名。没有素材就是 None（正面趴下就是硬切）。"""
        p = self.postures.get(key)
        return p.get("enter") if p else None

    def posture_idle(self, key: str) -> str | None:
        """该姿态驻留期间来回播的微动素材。"""
        p = self.postures.get(key)
        return p.get("idle") if p else None

    def _side_index(self, posture: str, side: str) -> int:
        """该姿态下"脸朝左/朝右"是第几个朝向。靠姿势名的后缀认，不再加一层配置。"""
        for i, k in enumerate(self.posture_faces(posture)):
            if k.endswith("_" + side):
                return i
        return 0

    def walk_facing(self, posture: str, side: str) -> int:
        return self._side_index(posture, side)

    def gait_anim(self, posture: str, side: str, gait: str = "walk") -> dict | None:
        """该姿态下往左/往右移动用哪段步态。四足走只在趴姿注册表里，跑只在站姿里。

        gait 是"走/跑"这一层，与 posture（站/坐/趴）正交：
        同一姿态可以有两种步态，同一种步态在不同姿态下用的是不同素材。
        """
        p = self.postures.get(posture) or {}
        name = (p.get(gait) or {}).get(side)
        return self.skin.anims.get(name) if name else None

    def has_gait(self, posture: str, gait: str) -> bool:
        """两侧都齐才算有这套步态 —— 缺一侧的话走到头就会有一段没动画可播。"""
        return bool(self.gait_anim(posture, "left", gait)
                    and self.gait_anim(posture, "right", gait))

    def anim(self, name: str) -> dict | None:
        return self.skin.anims.get(name)

    def face_of(self, pose_key: str) -> int:
        return self.faces.index(pose_key) if pose_key in self.faces else 0
