"""Levels 3 to 5: harder versions of the block tasks.

What changes from levels 1 and 2:
- Every run gets a random layout (from a seed): positions, and blocks turned at any angle.
- The prompt has no step-by-step routine. The model gets the facts and the goal, not a recipe.
- Level 3 adds more blocks and a decoy. Level 4 removes the top camera. Level 5 asks the model to
  copy a structure from a picture, and the tools stop reporting where the gripper is or whether it is holding something.
- Every score records which cameras the model used and how many arm actions it took.

Each run has a fixed action budget (BUDGET below), enforced by the tools. The simulator's exact state is saved after
every arm action (in the sample's store, under "trace"), so replays can follow the real run step by step.

Try one layout of every level on the cheapest model first:
    uv run inspect eval levels.py levels_coords.py --model anthropic/claude-haiku-5-5 --limit 1
Full run (3 layouts x 3 levels, with and without coordinates, 3 models = 54 runs):
    uv run inspect eval levels.py levels_coords.py --model anthropic/claude-haiku-5-5,anthropic/claude-sonnet-5-5,anthropic/claude-opus-5-5
"""
from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ContentText
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState
from inspect_ai.util import store

from bench import (Block, Layout, build_cell, cell, grip, home, look, make_cell, move_gripper, png, prompt,
                   random_layout, release_and_park, tallest_stack, usage)

SEEDS = [1, 2, 3]                     # layouts per level; add more seeds for a sturdier score
VERSION = "v2"
BUDGET = {3: 130, 4: 100, 5: 100}      # actions per run (every tool call but submit), enforced by the tools


def limit(level):
    """A safety net well above the budget, so a run can never spin forever."""
    return 3 * BUDGET[level] + 20


def blocks_in(c, names):
    return {n: c.body_pos(n) * 100 for n in names}


def inside_tray(c, pos, colour):
    gx, gy, _ = c.body_pos(f"bin_{colour}") * 100
    return abs(pos[0] - gx) < 5.5 and abs(pos[1] - gy) < 5.5 and pos[2] < 8


# ======================================================================= level 3
L3_COLOURS = ["red", "red", "red", "blue", "blue", "blue", "orange"]


def l3_layout(seed):
    return random_layout(seed, L3_COLOURS, yaw=40)


@scorer(metrics=[mean()])
def l3_score():
    async def score(state: TaskState, target: Target):
        c = cell()
        release_and_park(c, settle=1.0)
        blocks = store().get("blocks")
        right = decoy_in_tray = 0
        notes = []
        for b in blocks:
            pos = c.body_pos(b["name"]) * 100
            if b["colour"] == "orange":
                bad = bool(inside_tray(c, pos, "red") or inside_tray(c, pos, "blue"))
                decoy_in_tray += bad
                notes.append(f"orange: {'in a tray (-1)' if bad else 'left out'}")
            else:
                ok = bool(inside_tray(c, pos, b["colour"]))
                right += ok
                notes.append(f"{b['colour']}: {'in' if ok else 'not in'} its tray")
        value = max(0, right - decoy_in_tray) / 6
        return Score(value=value, answer=f"{right}/6 sorted", explanation="; ".join(notes),
                     metadata=usage(state) | {"sorted": right, "decoy_in_tray": decoy_in_tray})
    return score


L3_PROMPT = prompt(
    objects="- There are seven 3 cm cubes: three red, three blue and one orange. They start in random places and may be turned at any angle.\n"
            "- There is a red tray and a blue tray on the mat. Both are within reach.",
    cameras=("top_cam", "front_cam", "wrist_cam"), feedback=True,
    goal="Put every red block in the red tray and every blue block in the blue tray. Leave the orange block out of both trays.\n"
         "When you call submit, the gripper is opened and parked, and then the blocks are measured. You get one point for each "
         "red or blue block resting inside the tray of its colour, and lose one point if the orange block is inside either tray. "
         "The score is out of 6.",
    actions_budget=BUDGET[3])


@task
def level3_messy_sort():
    return Task(
        dataset=[Sample(input="Sort the red and blue blocks into the trays of their colour. Leave the orange block out. "
                              "Call submit when you are done.", metadata={"seed": s}, id=f"layout-{s}") for s in SEEDS],
        setup=build_cell(l3_layout, BUDGET[3]),
        solver=react(prompt=L3_PROMPT, tools=[look(), move_gripper(), grip(), home()]),
        scorer=l3_score(),
        message_limit=limit(3),
        version=VERSION,
    )


# ======================================================================= level 4
def l4_layout(seed):
    return random_layout(seed, ["green"] * 4, yaw=40)


@scorer(metrics=[mean()])
def tower_score():
    async def score(state: TaskState, target: Target):
        c = cell()
        release_and_park(c)
        names = [b["name"] for b in store().get("blocks")]
        h = tallest_stack(c, names)
        h = int(h)
        return Score(value=(h - 1) / 3 if h > 0 else 0.0, answer=f"tower of {h}",
                     explanation=f"Tallest freestanding stack: {h} of 4 blocks.", metadata=usage(state) | {"tower": h})
    return score


L4_CAMERAS = ("front_cam", "wrist_cam")
L4_PROMPT = prompt(
    objects="- There are four green blocks. Each is a 3 cm cube, so a tower of four is 12 cm tall. They start in random places and may be turned at any angle.\n"
            "- The red and blue trays on the mat are not part of this task.",
    cameras=L4_CAMERAS, feedback=True,
    goal="Stack all four blocks into a single tower, one directly on top of another. Any block can be the bottom one, and the tower can go anywhere you can reach.\n"
         "When you call submit, the gripper is opened, lifted and parked, and after 2 seconds the blocks are measured. A block counts only if it rests "
         "squarely on the block below it, less than 1.5 cm off-centre. The score is the height of the tallest tower left standing: 4 blocks is full marks, "
         "3 or 2 blocks earn part marks.",
    actions_budget=BUDGET[4])


@task
def level4_stack_no_top_camera():
    return Task(
        dataset=[Sample(input="Stack all four green blocks into a single tower. Call submit when you are done.",
                        metadata={"seed": s}, id=f"layout-{s}") for s in SEEDS],
        setup=build_cell(l4_layout, BUDGET[4]),
        solver=react(prompt=L4_PROMPT, tools=[look(cameras=L4_CAMERAS), move_gripper(), grip(), home()]),
        scorer=tower_score(),
        message_limit=limit(4),
        version=VERSION,
    )


# ======================================================================= level 5
TARGET = [Block("t_red", "red", 21.0, -1.5), Block("t_blue", "blue", 21.0, 1.5), Block("t_green", "green", 21.0, 0.0, z_cm=4.5)]
L5_COLOURS = ["red", "blue", "green"]


def l5_layout(seed):
    return random_layout(seed, L5_COLOURS, yaw=40, keep_clear=(21.0, 0.0, 6.0))


def target_pictures():
    """Render the finished structure once, from the two fixed cameras the model also has."""
    c = make_cell(Layout(TARGET))
    return png(c.image("top_cam")), png(c.image("front_cam"))


@scorer(metrics=[mean()])
def copy_score():
    async def score(state: TaskState, target: Target):
        c = cell()
        release_and_park(c)
        blocks = store().get("blocks")
        matched, notes = 0, []
        for t in TARGET:
            name = next(b["name"] for b in blocks if b["colour"] == t.colour)
            x, y, z = c.body_pos(name) * 100
            ok = bool(abs(x - t.x_cm) <= 1.2 and abs(y - t.y_cm) <= 1.2 and abs(z - t.z_cm) <= 0.6)
            matched += ok
            notes.append(f"{t.colour}: {'in place' if ok else f'off ({x:.1f}, {y:.1f}, {z:.1f})'}")
        return Score(value=matched / 3, answer=f"{matched}/3 in place", explanation="; ".join(notes),
                     metadata=usage(state) | {"in_place": matched})
    return score


L5_PROMPT = prompt(
    objects="- There are three 3 cm cubes: one red, one blue and one green. They start in random places and may be turned at any angle.\n"
            "- The red and blue trays on the mat are not part of this task.",
    cameras=("top_cam", "front_cam", "wrist_cam"), feedback=False,
    goal="Build the structure shown in the two pictures in the task message, in the same place on the mat and with the same colours in the same positions.\n"
         "When you call submit, the gripper is opened, lifted and parked, and after 2 seconds the blocks are measured. Each block counts as in place "
         "if it is within 1.2 cm of its position in the picture, sideways, and within 0.6 cm in height. The score is the number of blocks in place, out of 3.",
    actions_budget=BUDGET[5])


@task
def level5_copy_the_picture():
    top, front = target_pictures()
    ask = ChatMessageUser(content=[
        ContentText(text="Build this structure. The first picture is from top_cam and the second from front_cam, "
                         "taken after the structure was finished and the arm was parked. Call submit when you are done."),
        top, front])
    return Task(
        dataset=[Sample(input=[ask], metadata={"seed": s}, id=f"layout-{s}") for s in SEEDS],
        setup=build_cell(l5_layout, BUDGET[5]),
        solver=react(prompt=L5_PROMPT, tools=[look(), move_gripper(feedback=False), grip(feedback=False), home(feedback=False)]),
        scorer=copy_score(),
        message_limit=limit(5),
        version=VERSION,
    )
