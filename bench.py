"""Shared pieces for levels 3 and up: scene building, random layouts, tools with settings, and scoring helpers.

Levels 1 and 2 are in sort_task.py and stack_task.py.
"""
from __future__ import annotations

import base64
import hashlib
import io
import random
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from inspect_ai.model import ContentImage
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import tool
from inspect_ai.util import store
from PIL import Image

from cell import GRIPPER_CLOSED, Cell

HERE = Path(__file__).parent
COLOURS = {"red": "0.9 0.15 0.15 1", "blue": "0.15 0.3 0.9 1", "green": "0.2 0.7 0.3 1",
           "orange": "0.98 0.55 0.1 1", "yellow": "0.95 0.85 0.15 1"}


# ---------------------------------------------------------------- scene
@dataclass
class Block:
    name: str
    colour: str
    x_cm: float
    y_cm: float
    yaw_deg: float = 0.0
    z_cm: float = 1.5


@dataclass
class Layout:
    blocks: list[Block]
    bins: bool = True
    notes: dict = field(default_factory=dict)


def scene_xml(layout: Layout) -> Path:
    """Write a scene file with exactly these blocks next to the arm model, and return its path."""
    base = (HERE / "so101" / "blocks_scene.xml").read_text()
    head, _, rest = base.partition("    <!-- Blocks: 3 cm cubes -->")
    body_end = rest.index("  </worldbody>")
    blocks = "".join(
        f'    <body name="{b.name}" pos="{b.x_cm / 100} {b.y_cm / 100} {b.z_cm / 100}" euler="0 0 {np.deg2rad(b.yaw_deg)}">'
        f'<freejoint/><geom class="block" rgba="{COLOURS[b.colour]}"/></body>\n'
        for b in layout.blocks)
    xml = head + "    <!-- Blocks -->\n" + blocks + rest[body_end:]
    if not layout.bins:
        start = xml.index('    <!-- Bins:')
        end = xml.index("    <!-- Blocks -->")
        xml = xml[:start] + xml[end:]
    key = hashlib.md5(xml.encode()).hexdigest()[:10]
    path = HERE / "so101" / f"_scene_{key}.xml"
    if not path.exists():
        path.write_text(xml)
    return path


class LevelCell(Cell):
    """The level 1-2 arm with two changes: the jaws open less wide, and the swinging jaw always opens away from the arm's base.

    Opening: levels 1-2 opened the jaws fully (about 9 cm between the tips). With more blocks on the mat, the wide
    swing kept knocking neighbours over, so here they open to about 7 cm, still more than twice a block's width.

    The SO-101 has one fixed jaw and one swinging jaw. Any jaw line can be reached with the wrist turned
    two ways, half a turn apart, which puts the swinging jaw on opposite sides of the block. Levels 1-2
    took whichever was the shorter turn from where the wrist happened to be, so the same command could
    behave differently depending on history, and grips with the jaw swinging back toward the base
    failed more often. Here the choice is fixed, and the prompt tells the model about it.
    """

    OPEN = 0.6                               # jaw angle when open (levels 1-2: 1.0)

    def reset(self) -> None:
        super().reset()
        self.d.qpos[self.grip_qadr] = self.OPEN
        self.d.ctrl[self.grip_act] = self.OPEN
        mujoco.mj_forward(self.m, self.d)

    def grip(self, close: bool) -> dict:
        self.d.ctrl[self.grip_act] = GRIPPER_CLOSED if close else self.OPEN
        self.step(0.6)
        jaw = float(self.d.qpos[self.grip_qadr])
        return {"closed": close, "holding_something": bool(close and jaw > GRIPPER_CLOSED + 0.08)}

    def solve_ik(self, pos, jaw_deg=None):
        q, err, tilt = super().solve_ik(pos, None)
        if jaw_deg is None:
            return q, err, tilt
        pos = np.asarray(pos, dtype=float)
        outward = np.arctan2(pos[1], pos[0] - 0.039)
        want = np.deg2rad(jaw_deg)
        if np.cos(want - outward) < 0:                                  # same jaw line, swinging jaw on the far side
            want += np.pi
        for _ in range(4):
            _, _, R = self._pose_error(pos, q)
            have = np.arctan2(R[1, 2], R[0, 2])                         # from the fixed jaw toward the swinging jaw
            jacr = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, self._scratch, None, jacr, self.tip)
            about_vertical = jacr[2, self.dadr[4]]
            if abs(about_vertical) < 0.2:
                break
            options = []
            for target in (want, want + np.pi):                         # the preferred side first
                turn = (target - have + np.pi) % (2 * np.pi) - np.pi
                options = [q[4] + (turn + k * 2 * np.pi) / about_vertical for k in (-1, 0, 1)]
                options = [r for r in options if self.lo[4] <= r <= self.hi[4]]
                if options:
                    break
            if not options:
                break
            new_roll = min(options, key=lambda r: abs(r - q[4]))
            if abs(new_roll - q[4]) < np.deg2rad(0.5):
                break
            q[4] = new_roll
            q = self._reach(pos, q, iters=80)
        err, tilt, _ = self._pose_error(pos, q)
        return q, err, tilt


def make_cell(layout: Layout) -> Cell:
    c = LevelCell(str(scene_xml(layout)))
    c.step(0.5)
    return c


def random_layout(seed: int, colours: list[str], *, x=(15.5, 26.5), y=(-9.5, 9.5), spacing=5.5,
                  yaw=40.0, keep_clear=None, bins=True) -> Layout:
    """Blocks at random spots in the working zone, at least `spacing` cm apart, turned up to +/- `yaw` degrees."""
    rng = random.Random(seed)
    spots: list[tuple[float, float]] = []
    misses = 0
    while len(spots) < len(colours):
        if misses > 400:                    # boxed in: start the layout again (still decided by the seed)
            spots, misses = [], 0
        p = (round(rng.uniform(*x), 1), round(rng.uniform(*y), 1))
        clear = not keep_clear or abs(p[0] - keep_clear[0]) >= keep_clear[2] or abs(p[1] - keep_clear[1]) >= keep_clear[2]
        if clear and all(np.hypot(p[0] - q[0], p[1] - q[1]) >= spacing for q in spots):
            spots.append(p)
        else:
            misses += 1
    blocks = [Block(f"blk_{i}_{c}", c, px, py, round(rng.uniform(-yaw, yaw))) for i, (c, (px, py)) in enumerate(zip(colours, spots))]
    return Layout(blocks, bins=bins)


# ---------------------------------------------------------------- one simulator per run
_cells: dict[str, Cell] = {}


def cell() -> Cell:
    return _cells[store().get("cell_id")]


@solver
def build_cell(make_layout, budget: int):
    """Before the model starts: build this sample's scene from its seed, and set its action budget."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        layout = make_layout(state.metadata["seed"])
        key = uuid.uuid4().hex
        _cells[key] = make_cell(layout)
        store().set("cell_id", key)
        store().set("blocks", [b.__dict__ for b in layout.blocks])
        store().set("budget", budget)
        store().set("actions_used", 0)
        store().set("trace", [snapshot(_cells[key])])
        return state
    return solve


def snapshot(c: Cell) -> dict:
    """The simulator's exact state, saved after every arm action so replays can follow the real run step by step."""
    return {"qpos": np.round(c.d.qpos, 7).tolist(), "qvel": np.round(c.d.qvel, 7).tolist(),
            "ctrl": np.round(c.d.ctrl, 7).tolist(), "jaw_deg": c.jaw_deg, "time": round(c.d.time, 4)}


def restore(c: Cell, snap: dict) -> None:
    c.d.qpos[:] = snap["qpos"]
    c.d.qvel[:] = snap["qvel"]
    c.d.ctrl[:] = snap["ctrl"]
    c.jaw_deg = snap["jaw_deg"]
    c.d.time = snap["time"]
    mujoco.mj_forward(c.m, c.d)


def _record() -> None:
    store().set("trace", store().get("trace", []) + [snapshot(cell())])


def _spend() -> str | None:
    """Count one action. Once the budget is used up, every tool refuses and says so."""
    used, budget = store().get("actions_used", 0), store().get("budget", 10**9)
    if used >= budget:
        return f"You have used all {budget} actions. Nothing happened. Call submit now."
    store().set("actions_used", used + 1)
    return None


def cm(xyz) -> list[float]:
    return [round(v * 100, 1) for v in xyz]


def png(img: np.ndarray) -> ContentImage:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return ContentImage(image="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())


# ---------------------------------------------------------------- tools with settings
CAMERA_TEXT = {
    "top_cam": '"top_cam" looks straight down with the arm\'s base at the bottom of the picture',
    "front_cam": '"front_cam" looks back at the arm from the far side of the table',
    "wrist_cam": '"wrist_cam" is mounted on the gripper',
}


@tool
def look(cameras: tuple[str, ...] = ("top_cam", "front_cam", "wrist_cam")):
    async def execute(camera: str):
        """Take a picture of the workspace.

        Args:
            camera: The name of one of the cameras listed in your instructions.
        """
        if camera not in cameras:
            return f"There is no camera called {camera!r}. Available: {', '.join(cameras)}."
        if (out := _spend()):
            return out
        return png(cell().image(camera))
    return execute


@tool
def move_gripper(feedback: bool = True):
    async def execute(x_cm: float, y_cm: float, z_cm: float, jaw_deg: float | None = None):
        """Move the gripper tip to a point. The gripper points down where it can.

        Args:
            x_cm: Distance in front of the arm's base, in centimetres.
            y_cm: Distance to the arm's left (negative = right), in centimetres.
            z_cm: Height above the mat, in centimetres.
            jaw_deg: Direction the jaws close along. 0 = along x, 90 = along y. Omit to keep it unchanged.
        """
        if (out := _spend()):
            return out
        r = cell().move_to(x_cm / 100, y_cm / 100, z_cm / 100, jaw_deg)
        _record()
        if not r["ok"]:
            return str({"ok": False, "reason": "out of reach"} | ({"gripper_cm": cm(r["gripper"])} if feedback else {}))
        if feedback:
            return str({"ok": True, "gripper_cm": cm(r["gripper"]), "tilt_deg": r["tilt_deg"]})
        return str({"ok": True})
    return execute


@tool
def grip(feedback: bool = True):
    async def execute(close: bool):
        """Close or open the gripper.

        Args:
            close: True to close the jaws, False to open them.
        """
        if (out := _spend()):
            return out
        r = cell().grip(close)
        _record()
        return str(r) if feedback else str({"closed": close})
    return execute


@tool
def home(feedback: bool = True):
    async def execute():
        """Lift the arm up and back so it does not block the cameras."""
        if (out := _spend()):
            return out
        r = cell().home()
        _record()
        return str({"ok": True, "gripper_cm": cm(r["gripper"])}) if feedback else str({"ok": True})
    return execute


# ---------------------------------------------------------------- scoring helpers
def release_and_park(c: Cell, settle: float = 2.0) -> None:
    """What every scorer does first: let go, lift straight up, park, and let things settle."""
    x, y, z = c.gripper_pos()
    c.grip(False)
    c.move_to(x, y, min(z + 0.06, 0.20))
    c.home()
    c.step(settle)


def usage(state: TaskState) -> dict:
    """Camera use and arm actions in this run, saved with every score."""
    calls = [tc for m in state.messages if m.role == "assistant" for tc in (m.tool_calls or [])]
    cams = Counter(tc.arguments.get("camera", "?") for tc in calls if tc.function == "look")
    return {"looks": dict(cams), "looks_total": sum(cams.values()),
            "arm_actions": sum(tc.function in ("move_gripper", "grip", "home") for tc in calls),
            "tool_calls": len(calls), "actions_used": store().get("actions_used"), "budget": store().get("budget"),
            "ran_out": store().get("actions_used", 0) >= store().get("budget", 10**9)}


def tallest_stack(c: Cell, names: list[str], block_cm: float = 3.0) -> int:
    pos = {n: c.body_pos(n) * 100 for n in names}

    def rests_on(top, below):
        dz = pos[top][2] - pos[below][2]
        off = np.hypot(pos[top][0] - pos[below][0], pos[top][1] - pos[below][1])
        return abs(dz - block_cm) < 0.6 and off < block_cm / 2

    def height_from(base, used):
        best = 1
        for other in names:
            if other not in used and rests_on(other, base):
                best = max(best, 1 + height_from(other, used | {other}))
        return best

    on_mat = [n for n in names if abs(pos[n][2] - block_cm / 2) < 0.6]
    return max((height_from(b, {b}) for b in on_mat), default=0)


# ---------------------------------------------------------------- prompt pieces
def prompt(objects: str, cameras: tuple[str, ...], feedback: bool, goal: str, actions_budget: int) -> str:
    cam_lines = "\n".join(f"- {CAMERA_TEXT[c]}." for c in cameras)
    extra = []
    if "top_cam" in cameras:
        extra.append("In top_cam, up in the picture is +x (away from the base) and left is +y (the arm's left). "
                     "The centre of the picture is the point x=20, y=0.")
    if "front_cam" in cameras:
        extra.append("front_cam faces the arm, so the arm's left (+y) appears on the RIGHT of its pictures.")
    if "wrist_cam" in cameras:
        extra.append("wrist_cam turns with the gripper.")
    if "top_cam" not in cameras:
        extra.append("There is no camera looking straight down in this task.")
    if feedback:
        move_reply = 'The reply says where the gripper tip actually ended up (gripper_cm: x, y, z in cm), or "out of reach".'
        grip_reply = "After closing, the reply says holding_something: true or false."
    else:
        move_reply = 'The reply only says whether the move happened, or "out of reach". It does not say where the tip ended up.'
        grip_reply = "The reply only confirms the command. It does not say whether you caught anything: check with the cameras."
    return f"""You control a small robot arm on a table using tools. You cannot see the table directly: everything you know comes from camera pictures and from the replies your tools return.

## Coordinates
- All distances are in centimetres.
- The arm's base is at x=0, y=0. x increases away from the base. y increases to the arm's left (negative y is to its right). z is height above the mat (the mat is z=0).
- The mat is a checkerboard. Each square is 5 cm wide.
- The gripper works best between 15 and 28 cm in front of the base. Spots outside that range may come back "out of reach".
{objects}

## Cameras
Looking does not move anything. Each picture shows the table at the moment you asked.
{cam_lines}
{" ".join(extra)}

## Actions
- look(camera): a picture from one of the cameras above.
- move_gripper(x_cm, y_cm, z_cm, jaw_deg): moves the gripper tip to that point. jaw_deg sets the line the jaws close along: 0 = along x, 90 = along y. Leave it out to keep the current setting. {move_reply}
- grip(close): True closes the jaws, False opens them. The gripper has one fixed jaw and one swinging jaw. Open, the tips are about 7 cm apart, and the swinging jaw always opens on the side away from the arm's base. {grip_reply}
- home(): lifts the arm up and back toward its base, out of most of the cameras' view.
- submit: call it when you are done.

## The goal and how it is checked
{goal}

## Budget
You have {actions_budget} actions. Every tool call counts (look, move_gripper, grip and home), except submit. When they run out, the tools stop working and you should call submit."""
