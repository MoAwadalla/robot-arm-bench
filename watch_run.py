"""Watch a logged model run again, live in a 3D window.

    uv run mjpython watch_run.py                     # the newest run in logs/
    uv run mjpython watch_run.py logs/<file>.eval    # a specific run
    SPEED=3 uv run mjpython watch_run.py             # 3x faster

It replays the model's tool calls, in order, in the simulator. The terminal shows what the
model wrote before each step, so you can follow its reasoning while the arm moves.
"""
import glob
import os
import sys
import time

import mujoco
from inspect_ai.log import read_eval_log

log_path = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob("logs/*.eval"), key=os.path.getmtime)
log = read_eval_log(log_path)
sample = log.samples[0]
print(f"Replaying {log_path}\nTask: {log.eval.task}   Model: {log.eval.model}")
for name, s in (sample.scores or {}).items():
    print(f"Score: {s.value}   {s.explanation}")

if "stack" in log.eval.task:
    from stack_task import new_stack_cell as make_cell
else:
    from cell import Cell as make_cell
cell = make_cell()
from pretty import prettify
prettify(cell.m)                         # nicer lighting for the window only

speed = float(os.environ.get("SPEED", "1"))
window = os.environ.get("NO_WINDOW") is None


def run_steps(viewer=None):
    if viewer is not None:
        physics_step = mujoco.mj_step
        count = [0]

        def step_and_show(m, d, *args):
            physics_step(m, d, *args)
            if d is cell.d:
                count[0] += 1
                if count[0] % 4 == 0:
                    viewer.sync()
                time.sleep(m.opt.timestep / speed)

        mujoco.mj_step = step_and_show
        cell.step(1.5)

    replies = {m.tool_call_id: m for m in sample.messages if m.role == "tool"}
    n = 0
    for msg in sample.messages:
        if msg.role != "assistant":
            continue
        if msg.text.strip():
            print(f"\nMODEL: {msg.text.strip()[:500]}")
        for tc in msg.tool_calls or []:
            n += 1
            a = tc.arguments
            logged = replies.get(tc.id)
            logged = logged.content if logged is not None and isinstance(logged.content, str) else "[picture]"
            if tc.function == "move_gripper":
                got = cell.move_to(a["x_cm"] / 100, a["y_cm"] / 100, a["z_cm"] / 100, a.get("jaw_deg"))
            elif tc.function == "grip":
                got = cell.grip(a["close"])
            elif tc.function == "home":
                got = cell.home()
            else:
                got = None                       # look or submit: nothing moves
            print(f"  [{n}] {tc.function}({a})")
            if got is not None:
                print(f"       in the run: {logged}")
                print(f"       now:        {got}")
    print("\nReplay finished.")


if window:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(cell.m, cell.d) as viewer:
        run_steps(viewer)
        print("Close the window to exit.")
        while viewer.is_running():
            cell.step(0.02)
else:
    run_steps()
