"""Night one: look through the cameras, then pick up a red block and drop it in the red bin.

Run:  uv run python demo.py
"""
from PIL import Image

from cell import Cell

cell = Cell()

# 1. Look. These three pictures are what the AI model will see.
for cam in ("top_cam", "front_cam", "wrist_cam"):
    Image.fromarray(cell.image(cam)).save(f"{cam}.png")
print("saved top_cam.png, front_cam.png, wrist_cam.png")

# 2. Act. Tonight we cheat and read the block's true position; the AI will have to find it by looking.
bx, by, _ = cell.body_pos("block_red_1")
rx, ry, _ = cell.body_pos("bin_red")

print("above block:", cell.move_to(bx, by, 0.08))
print("down       :", cell.move_to(bx, by, 0.017))
print("close      :", cell.grip(True))
print("lift       :", cell.move_to(bx, by, 0.10))
print("over bin   :", cell.move_to(rx, ry, 0.10))
print("lower      :", cell.move_to(rx, ry, 0.045))
print("open       :", cell.grip(False))
print("home       :", cell.home())

# 3. Score. The simulator knows exactly where everything ended up.
x, y, z = cell.body_pos("block_red_1")
in_bin = abs(x - rx) < 0.055 and abs(y - ry) < 0.055 and z < 0.03
print("red block in red bin:", in_bin)
Image.fromarray(cell.image("top_cam")).save("after.png")
print("saved after.png")
