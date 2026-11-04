"""Level 2: stack four same-coloured blocks into one tower, one move at a time.

Same scene as level 1 (sort_task.py): the bins stay on the mat but are not used.
All four blocks are coloured green and start well inside the arm's working zone.

Run:
    uv run inspect eval stack_task.py --model anthropic/claude-haiku-5-5
Look at what happened:
    uv run inspect view
"""
import base64
import io
import uuid

import mujoco
import numpy as np
from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.model import ContentImage
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState
from inspect_ai.tool import tool
from inspect_ai.util import store
from PIL import Image

from cell import Cell

BLOCKS = ["block_red_1", "block_red_2", "block_blue_1", "block_blue_2"]   # body names in the scene file
GREEN = [0.2, 0.7, 0.3, 1.0]
START_XY_CM = [(17, 7), (17, -7), (25, 7), (25, -7)]   # inside the 15-28 cm zone, clear of x=21, y=0
BLOCK_CM = 3.0
BUDGET = 100                          # actions per run, counted by the tools themselves (same rule as levels 3-5)
CAMERAS = ("top_cam", "front_cam", "wrist_cam")

_cells: dict[str, Cell] = {}          # one private simulator per run


def new_stack_cell() -> Cell:
    c = Cell()
    for name, (x, y) in zip(BLOCKS, START_XY_CM):
        body = c.m.body(name)
        c.m.geom_rgba[body.geomadr[0]] = GREEN                    # all blocks look the same
        q = c.m.jnt_qposadr[body.jntadr[0]]
        c.d.qpos[q:q + 7] = [x / 100, y / 100, 0.015, 1, 0, 0, 0]  # set the starting position
    mujoco.mj_forward(c.m, c.d)
    c.step(0.5)
    return c


def cell() -> Cell:
    key = store().get("cell_id")
    if key is None:
        key = uuid.uuid4().hex
        store().set("cell_id", key)
        _cells[key] = new_stack_cell()
    return _cells[key]


def _spend() -> str | None:
    """Count one action. Once the budget is used up, every tool refuses and says so."""
    used = store().get("actions_used", 0)
    if used >= BUDGET:
        return f"You have used all {BUDGET} actions. Nothing happened. Call submit now."
    store().set("actions_used", used + 1)
    return None


def cm(xyz) -> list[float]:
    return [round(v * 100, 1) for v in xyz]


# ---------------------------------------------------------------- tools (same as level 1)
@tool
def look():
    async def execute(camera: str = "top_cam"):
        """Take a picture of the workspace.

        Args:
            camera: "top_cam" looks straight down with the arm's base at the bottom of the picture.
                "front_cam" looks at the arm from the front. "wrist_cam" is mounted on the gripper.
        """
        if camera not in CAMERAS:
            return f"There is no camera called {camera!r}. Available: {', '.join(CAMERAS)}."
        if (out := _spend()):
            return out
        buf = io.BytesIO()
        Image.fromarray(cell().image(camera)).save(buf, format="PNG")
        return ContentImage(image="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())

    return execute


@tool
def move_gripper():
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
        if not r["ok"]:
            return str({"ok": False, "reason": r["reason"], "gripper_cm": cm(r["gripper"])})
        return str({"ok": True, "gripper_cm": cm(r["gripper"]), "tilt_deg": r["tilt_deg"]})

    return execute


@tool
def grip():
    async def execute(close: bool):
        """Close or open the gripper.

        Args:
            close: True to close the jaws, False to open them.
        """
        if (out := _spend()):
            return out
        return str(cell().grip(close))

    return execute


@tool
def home():
    async def execute():
        """Lift the arm up and back so it does not block the cameras."""
        if (out := _spend()):
            return out
        return str({"ok": True, "gripper_cm": cm(cell().home()["gripper"])})

    return execute


# ---------------------------------------------------------------- the win condition
def tallest_stack(c: Cell) -> int:
    """Number of blocks in the tallest freestanding stack.

    A block counts as resting on another when its centre is one block height (3 cm, +/- 0.6 cm)
    above the other's centre and its middle sits over the other block (less than 1.5 cm off-centre).
    A stack must start from a block lying on the mat.
    """
    pos = {n: c.body_pos(n) * 100 for n in BLOCKS}           # centimetres

    def rests_on(top, below):
        dz = pos[top][2] - pos[below][2]
        off = np.hypot(pos[top][0] - pos[below][0], pos[top][1] - pos[below][1])
        return abs(dz - BLOCK_CM) < 0.6 and off < BLOCK_CM / 2

    def height_from(base, used):
        best = 1
        for other in BLOCKS:
            if other not in used and rests_on(other, base):
                best = max(best, 1 + height_from(other, used | {other}))
        return best

    on_mat = [n for n in BLOCKS if abs(pos[n][2] - BLOCK_CM / 2) < 0.6]
    return max((height_from(b, {b}) for b in on_mat), default=0)


@scorer(metrics=[mean()])
def tower_height():
    async def score(state: TaskState, target: Target):
        c = cell()
        # The tower must stand on its own: let go, lift straight up, park the arm, wait 2 seconds.
        x, y, z = c.gripper_pos()
        c.grip(False)
        c.move_to(x, y, min(z + 0.06, 0.20))
        c.home()
        c.step(2.0)
        h = tallest_stack(c)
        heights = ", ".join(f"{n}: z={c.body_pos(n)[2] * 100:.1f} cm" for n in BLOCKS)
        return Score(
            value=(h - 1) / 3 if h > 0 else 0.0,                 # 1 block = 0, 2 = 1/3, 3 = 2/3, 4 = 1
            answer=f"{h} blocks",
            explanation=f"Tallest freestanding stack: {h} of 4 blocks. {heights}",
        )

    return score


# ---------------------------------------------------------------- what the model is told
PROMPT_VERSION = "v2"                 # v2: action budget counted by the tools, replies in cm

PROMPT = """You control a small robot arm on a table using tools. You cannot see the table directly: everything you know comes from camera pictures and from the replies your tools return.

## Coordinates
- All distances are in centimetres.
- The arm's base is at x=0, y=0. x increases away from the base. y increases to the arm's left (negative y is to its right). z is height above the mat (the mat is z=0).
- The mat is a checkerboard. Each square is 5 cm wide.
- The gripper works best between 15 and 28 cm in front of the base. Spots outside that range may come back "out of reach".
- There are four green blocks. Each is a 3 cm cube, so a tower of four is 12 cm tall.
- The red and blue trays on the mat are not part of this task. Leave them alone.

## Cameras
Looking does not move anything, so look as often as you need. Each picture shows the table at the moment you asked, so take a new one after anything moves.
- top_cam looks straight down. The arm's base is at the bottom of the picture. Up in the picture is +x (away from the base) and left in the picture is +y (the arm's left). The centre of the picture is the point x=20, y=0. This is the best camera for finding where blocks are on the mat, but it cannot show how tall a tower is.
- front_cam looks back at the arm from the far side of the table, slightly from above. Because it faces the arm, the arm's left (+y) appears on the RIGHT of this picture. This is the best camera for counting how many blocks a tower has and checking that it stands straight.
- wrist_cam is mounted on the gripper and turns with it. Use it for close-up alignment, especially when lowering a block onto the tower.
The arm can hide blocks from top_cam and front_cam. If it is in the way, call home() and look again.

## Actions
- look(camera): a picture from "top_cam", "front_cam" or "wrist_cam".
- move_gripper(x_cm, y_cm, z_cm, jaw_deg): moves the gripper tip to that point. Low down (below about 8 cm) the gripper points straight down; higher up it may tilt slightly. jaw_deg sets the line the jaws close along: 0 = along x, 90 = along y. Leave it out to keep the current setting. The reply says where the gripper tip actually ended up (gripper_cm: x, y, z in cm), or "out of reach".
- grip(close): True closes the jaws, False opens them. Open jaws are about 9 cm apart. After closing, the reply says holding_something: true or false, so you know at once whether you caught something.
- home(): lifts the arm up and back toward its base, out of most of the cameras' view. Use it to clear the view, or to reset when you are stuck.
- submit: call it when the tower is finished.

## The goal and how it is checked
Stack all four blocks into a single tower, one directly on top of another. You may use any block as the bottom one.
When you call submit, the gripper is opened, lifted and parked, and after 2 seconds the blocks are measured. A block counts as part of the tower only if it rests squarely on the block below it, less than 1.5 cm off-centre. Your score is the height of the tallest tower left standing: 4 blocks is full marks, 3 or 2 blocks earn part marks, and a tower that falls over earns nothing for the blocks that fell.

## A routine that works
1. Call home(), take a top_cam picture, and work out each block's x and y from the grid.
2. Choose where the tower goes. Around x=21, y=0 works well: it is in the middle of the working zone, where the gripper can point straight down. If a block already sits there, use that block as the base and leave it in place.
3. To pick up a block: open the gripper, move above the block at z=8 with the jaws lined up with its sides, lower to z=2, and close. If holding_something is false, open, lift to z=8, look again, correct your estimate, and retry.
4. Lift the block to z=15 before moving it, so it clears the tower.
5. To place it: move above the tower at z=15, keeping the jaws lined up the same way as for the blocks below. Lower until the held block's bottom is just above the top of the tower: with k blocks already in the tower, move to z = 3 × k + 2 (5 for the second block, 8 for the third, 11 for the fourth). Then open the gripper.
6. Lift straight up to z=15 before moving away, so you do not knock the tower.
7. Call home() and check the tower with front_cam before fetching the next block. If a block landed off-centre, fix it before adding more on top.

## Budget
You have {budget} actions. Every tool call counts (look, move_gripper, grip and home), except submit. When they run out, the tools stop working and you should call submit. A finished, standing tower matters most. Use pictures to check your work, but if a move keeps failing, change your approach instead of repeating it.""".replace("{budget}", str(BUDGET))


@task
def stack_blocks():
    return Task(
        dataset=[Sample(input="Stack all four green blocks into a single tower. Call submit when you are done.")],
        solver=react(prompt=PROMPT, tools=[look(), move_gripper(), grip(), home()]),
        scorer=tower_height(),
        message_limit=3 * BUDGET + 20,    # only a safety net: the tools stop after BUDGET actions, however they are batched
        version=PROMPT_VERSION,
    )
