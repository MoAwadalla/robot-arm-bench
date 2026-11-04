"""Levels 3 to 5 with the answers to "where is everything?" given in the task message.

Same scenes, same layouts, same tools, same scoring as levels.py. The only difference: the task message lists the
exact starting position and angle of every block (and, for level 3, the tray centres; for level 5, the target
positions). Comparing the two runs separates the two ways a model can fail:
- it scores well here but not in levels.py  ->  it struggled to work out where things are from the pictures
- it also fails here                         ->  it struggled with planning or handling the blocks

Run (one pass over the 3 layouts, 27 runs):
    uv run inspect eval levels_coords.py --model anthropic/claude-haiku-5-5,anthropic/claude-sonnet-5-5,anthropic/claude-opus-5-5
"""
from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ContentText

import levels
from bench import build_cell, grip, home, look, move_gripper

TRAYS = {"red": (16.0, 17.0), "blue": (16.0, -17.0)}           # tray centres, from so101/blocks_scene.xml
NOTE = ("These are exact positions at the start, measured in the simulator. x, y are the centre of each block in cm; "
        "angle is how far it is turned around the vertical, in degrees. Blocks you push or drop will move, so check with the cameras.")


def block_lines(layout):
    return "\n".join(f"- {b.colour} block: x={b.x_cm:g}, y={b.y_cm:g}, angle={b.yaw_deg:g}" for b in layout.blocks)


@task
def level3_messy_sort_coords():
    samples = []
    for s in levels.SEEDS:
        text = ("Sort the red and blue blocks into the trays of their colour. Leave the orange block out. Call submit when you are done.\n\n"
                f"Where everything is:\n{block_lines(levels.l3_layout(s))}\n"
                f"- red tray: centre x={TRAYS['red'][0]:g}, y={TRAYS['red'][1]:g}; blue tray: centre x={TRAYS['blue'][0]:g}, y={TRAYS['blue'][1]:g}. "
                "Each tray's floor is about 11 cm square, with walls about 2.4 cm high.\n\n" + NOTE)
        samples.append(Sample(input=text, metadata={"seed": s, "coords": True}, id=f"layout-{s}"))
    return Task(dataset=samples, setup=build_cell(levels.l3_layout, levels.BUDGET[3]),
                solver=react(prompt=levels.L3_PROMPT, tools=[look(), move_gripper(), grip(), home()]),
                scorer=levels.l3_score(), message_limit=levels.limit(3), version=levels.VERSION)


@task
def level4_stack_no_top_camera_coords():
    samples = []
    for s in levels.SEEDS:
        text = ("Stack all four green blocks into a single tower. Call submit when you are done.\n\n"
                f"Where everything is:\n{block_lines(levels.l4_layout(s))}\n\n" + NOTE)
        samples.append(Sample(input=text, metadata={"seed": s, "coords": True}, id=f"layout-{s}"))
    return Task(dataset=samples, setup=build_cell(levels.l4_layout, levels.BUDGET[4]),
                solver=react(prompt=levels.L4_PROMPT, tools=[look(cameras=levels.L4_CAMERAS), move_gripper(), grip(), home()]),
                scorer=levels.tower_score(), message_limit=levels.limit(4), version=levels.VERSION)


@task
def level5_copy_the_picture_coords():
    top, front = levels.target_pictures()
    target = "\n".join(f"- {t.colour} block: x={t.x_cm:g}, y={t.y_cm:g}, centre height z={t.z_cm:g}" for t in levels.TARGET)
    samples = []
    for s in levels.SEEDS:
        text = ("Build this structure. The first picture is from top_cam and the second from front_cam, "
                "taken after the structure was finished and the arm was parked. Call submit when you are done.\n\n"
                f"Where the blocks start:\n{block_lines(levels.l5_layout(s))}\n" + NOTE + "\n\n"
                f"Where they are in the finished structure (also exact):\n{target}")
        samples.append(Sample(input=[ChatMessageUser(content=[ContentText(text=text), top, front])],
                              metadata={"seed": s, "coords": True}, id=f"layout-{s}"))
    return Task(dataset=samples, setup=build_cell(levels.l5_layout, levels.BUDGET[5]),
                solver=react(prompt=levels.L5_PROMPT,
                             tools=[look(), move_gripper(feedback=False), grip(feedback=False), home(feedback=False)]),
                scorer=levels.copy_score(), message_limit=levels.limit(5), version=levels.VERSION)
