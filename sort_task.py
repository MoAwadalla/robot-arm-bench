"""The first benchmark task: sort four blocks into the bins of their colour, one move at a time.

Run against a real model (needs that provider's API key in your environment):
    uv run inspect eval sort_task.py --model <provider>/<model-name>
Look at what happened:
    uv run inspect view
"""
import base64
import io
import uuid

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

BUDGET = 100                          # actions per run, counted by the tools themselves (same rule as levels 3-5)
CAMERAS = ("top_cam", "front_cam", "wrist_cam")
_cells: dict[str, Cell] = {}          # one private simulator per run


def cell() -> Cell:
    key = store().get("cell_id")
    if key is None:
        key = uuid.uuid4().hex
        store().set("cell_id", key)
        _cells[key] = Cell()
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


@scorer(metrics=[mean()])
def blocks_sorted():
    async def score(state: TaskState, target: Target):
        c = cell()
        # Same as levels 2-5: let go, lift straight up, park the arm, wait 2 seconds, then measure.
        x, y, z = c.gripper_pos()
        c.grip(False)
        c.move_to(x, y, min(z + 0.06, 0.20))
        c.home()
        c.step(2.0)
        right, notes = 0, []
        for block in ["block_red_1", "block_red_2", "block_blue_1", "block_blue_2"]:
            colour = block.split("_")[1]
            bx, by, bz = c.body_pos(block)
            gx, gy, _ = c.body_pos(f"bin_{colour}")
            ok = abs(bx - gx) < 0.055 and abs(by - gy) < 0.055 and bz < 0.08    # inside the walls, resting (may sit on another block)
            right += ok
            notes.append(f"{block}: {'in' if ok else 'not in'} {colour} bin")
        return Score(value=right / 4, answer=f"{right}/4 sorted", explanation="; ".join(notes))

    return score


PROMPT_VERSION = "v3"                 # v3: action budget counted by the tools, replies in cm, gripper parked before scoring

PROMPT = """You control a small robot arm on a table using tools. You cannot see the table directly: everything you know comes from camera pictures and from the replies your tools return.

## Coordinates
- All distances are in centimetres.
- The arm's base is at x=0, y=0. x increases away from the base. y increases to the arm's left (negative y is to its right). z is height above the mat (the mat is z=0).
- The mat is a checkerboard. Each square is 5 cm wide.
- The gripper works best between 15 and 28 cm in front of the base. Spots outside that range may come back "out of reach". Both bins are within reach.
- The blocks are 3 cm cubes.

## Cameras
Looking does not move anything, so look as often as you need. Each picture shows the table at the moment you asked, so take a new one after anything moves.
- top_cam looks straight down. The arm's base is at the bottom of the picture. Up in the picture is +x (away from the base) and left in the picture is +y (the arm's left). The centre of the picture is the point x=20, y=0. This is the best camera for measuring positions with the grid.
- front_cam looks back at the arm from the far side of the table, slightly from above. Because it faces the arm, the arm's left (+y) appears on the RIGHT of this picture. Use it to judge heights and whether the gripper is level with a block.
- wrist_cam is mounted on the gripper and turns with it. Use it for close-up alignment just before you close the jaws.
The arm can hide blocks from top_cam. If it is in the way, call home() and look again.

## Actions
- look(camera): a picture from "top_cam", "front_cam" or "wrist_cam".
- move_gripper(x_cm, y_cm, z_cm, jaw_deg): moves the gripper tip to that point. Low down (below about 8 cm) the gripper points straight down; higher up it may tilt. jaw_deg sets the line the jaws close along: 0 = along x, 90 = along y. Leave it out to keep the current setting. The reply says where the gripper tip actually ended up (gripper_cm: x, y, z in cm), or "out of reach".
- grip(close): True closes the jaws, False opens them. Open jaws are about 9 cm apart. After closing, the reply says holding_something: true or false, so you know at once whether you caught something.
- home(): lifts the arm up and back toward its base, out of most of top_cam's view. Use it to clear the view, or to reset when you are stuck.
- submit: call it when the task is done. The gripper is then opened and parked, and after 2 seconds the blocks are checked.

## A routine that works
1. Call home(), take a top_cam picture, and work out each block's x and y from the grid.
2. Open the gripper and move above the block at z=8, with the jaws lined up with the block's sides.
3. Lower to z=2 and close. The middle of a block is 1.5 cm above the mat.
4. If holding_something is false: open, lift to z=8, look again, correct your estimate, and retry.
5. Lift to z=10, move above the centre of the bin, lower to about z=5, and open.
6. Travel between places at z=8 or higher. Moving low across the mat can push blocks away, and a block pushed past 28 cm may end up out of reach.
7. After placing a block, call home() and look again before choosing the next one.

## Budget
You have {budget} actions. Every tool call counts (look, move_gripper, grip and home), except submit. When they run out, the tools stop working and you should call submit. Getting every block into the right bin matters most. Use pictures to check your work, but if a move keeps failing, change your approach instead of repeating it.""".replace("{budget}", str(BUDGET))


@task
def sort_blocks():
    return Task(
        dataset=[Sample(input="Put every block into the bin of its own colour. Call submit when you are done.")],
        solver=react(prompt=PROMPT, tools=[look(), move_gripper(), grip(), home()]),
        scorer=blocks_sorted(),
        message_limit=3 * BUDGET + 20,    # only a safety net: the tools stop after BUDGET actions, however they are batched
        version=PROMPT_VERSION,
    )
