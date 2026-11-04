"""Watch the scripted pickup happen live in a 3D window.

Run on a Mac with:  uv run mjpython watch_demo.py
"""
import time

import mujoco
import mujoco.viewer

from cell import Cell
from pretty import prettify

cell = Cell()
prettify(cell.m)                         # nicer lighting for the window only

with mujoco.viewer.launch_passive(cell.m, cell.d) as viewer:
    # Show every physics step in the window, at real-time speed.
    physics_step = mujoco.mj_step
    count = 0

    def step_and_show(m, d, *args):
        global count
        physics_step(m, d, *args)
        if d is cell.d:
            count += 1
            if count % 4 == 0:          # redraw 50 times a second
                viewer.sync()
            time.sleep(m.opt.timestep)

    mujoco.mj_step = step_and_show

    cell.step(1.5)                       # a moment to look before the arm moves
    bx, by, _ = cell.body_pos("block_red_1")
    rx, ry, _ = cell.body_pos("bin_red")
    cell.move_to(bx, by, 0.08)
    cell.move_to(bx, by, 0.017)
    cell.grip(True)
    cell.move_to(bx, by, 0.10)
    cell.move_to(rx, ry, 0.10)
    cell.move_to(rx, ry, 0.045)
    cell.grip(False)
    cell.home()

    print("Done. Close the window to exit.")
    while viewer.is_running():           # keep the window open so you can look around
        cell.step(0.02)
