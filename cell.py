"""The virtual cell: a simulated SO-101 arm, three cameras, blocks and bins.

Everything an AI model will later do goes through four methods of `Cell`:

    image(camera)            look through a camera
    move_to(x, y, z, jaw)    put the gripper tip somewhere
    grip(close)              close or open the gripper
    home()                   lift the arm out of the cameras' way

Units are metres. The arm's base sits at (0, 0). +x is away from the base,
+y is to the arm's left, +z is up. The mat is at z = 0.
"""
from __future__ import annotations

import mujoco
import numpy as np

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_OPEN = 1.0        # jaw angle in radians: about 9 cm between the jaw tips
GRIPPER_CLOSED = -0.17    # fully shut
REST_POINT = (0.14, 0.0, 0.18)             # where home() parks the gripper tip
IK_SEED = np.array([0.0, 0.0, 0.5, 1.2, 1.57])   # a gripper-down pose the solver starts from
DOWN = np.array([0.0, 0.0, -1.0])


class Cell:
    def __init__(self, xml_path: str = "so101/blocks_scene.xml", width: int = 640, height: int = 480):
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self._scratch = mujoco.MjData(self.m)     # a copy used only for the arm maths
        self.renderer = mujoco.Renderer(self.m, height, width)
        self.tip = self.m.site("gripperframe").id
        joint_ids = [self.m.joint(j).id for j in ARM_JOINTS]
        self.qadr = self.m.jnt_qposadr[joint_ids]
        self.dadr = self.m.jnt_dofadr[joint_ids]
        self.lo, self.hi = self.m.jnt_range[joint_ids, 0], self.m.jnt_range[joint_ids, 1]
        self.arm_act = np.array([self.m.actuator(j).id for j in ARM_JOINTS])
        self.grip_act = self.m.actuator("gripper").id
        self.grip_qadr = self.m.jnt_qposadr[self.m.joint("gripper").id]
        self.d.qpos[self.qadr] = IK_SEED
        mujoco.mj_forward(self.m, self.d)
        self.jaw_deg = 0.0                           # the jaw direction move_to() holds until told otherwise
        self.home_q = self.solve_ik(REST_POINT)[0]
        self.reset()

    # ------------------------------------------------------------------ basics
    def reset(self) -> None:
        """Put everything back where the scene file says, with the arm parked."""
        mujoco.mj_resetData(self.m, self.d)
        self.jaw_deg = 0.0
        self.d.qpos[self.qadr] = self.home_q
        self.d.ctrl[self.arm_act] = self.home_q
        self.d.qpos[self.grip_qadr] = GRIPPER_OPEN
        self.d.ctrl[self.grip_act] = GRIPPER_OPEN
        mujoco.mj_forward(self.m, self.d)
        self.step(0.5)

    def step(self, seconds: float) -> None:
        for _ in range(int(round(seconds / self.m.opt.timestep))):
            mujoco.mj_step(self.m, self.d)

    def image(self, camera: str = "top_cam") -> np.ndarray:
        """An RGB picture (height x width x 3) from 'top_cam', 'front_cam' or 'wrist_cam'."""
        self.renderer.update_scene(self.d, camera=camera)
        return self.renderer.render()

    def gripper_pos(self) -> list[float]:
        return self.d.site_xpos[self.tip].round(3).tolist()

    def body_pos(self, name: str) -> np.ndarray:
        """Ground-truth position of a block or bin. For scoring only: never show this to the model."""
        return self.d.body(name).xpos.copy()

    # ------------------------------------------------------- arm maths (IK)
    def _reach(self, pos: np.ndarray, q: np.ndarray, iters: int = 200) -> np.ndarray:
        """Adjust the first four joints so the tip is at `pos`, keeping the wrist-roll joint as given.

        Reaching the position comes first; pointing straight down is a preference the
        solver satisfies as far as the joints allow.
        """
        m, d = self.m, self._scratch
        dofs, lo, hi = self.dadr[:4], self.lo[:4], self.hi[:4]
        d.qpos[:] = self.d.qpos
        d.qpos[self.qadr] = q
        for it in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.site_xmat[self.tip].reshape(3, 3)
            e_pos = pos - d.site_xpos[self.tip]
            e_rot = np.cross(R[:, 0], DOWN)
            jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tip)
            Jp, Jr = jacp[:, dofs], jacr[:, dofs]
            Jp_pinv = Jp.T @ np.linalg.inv(Jp @ Jp.T + 1e-5 * np.eye(3))
            keep_tip_still = np.eye(4) - Jp_pinv @ Jp
            step_rot = keep_tip_still @ (Jr.T @ np.linalg.solve(Jr @ Jr.T + 1e-3 * np.eye(3), e_rot))
            gain = 0.5 if it < iters - 50 else 0.0                     # last 50 steps: position only
            dq = np.clip(Jp_pinv @ e_pos + gain * step_rot, -0.15, 0.15)
            d.qpos[self.qadr[:4]] = np.clip(d.qpos[self.qadr[:4]] + dq, lo, hi)
        return d.qpos[self.qadr].copy()

    def _pose_error(self, pos: np.ndarray, q: np.ndarray):
        """(position error in metres, tilt from vertical in degrees, rotation matrix of the tip) for joints q."""
        d = self._scratch
        d.qpos[:] = self.d.qpos
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        R = d.site_xmat[self.tip].reshape(3, 3).copy()
        err = float(np.linalg.norm(pos - d.site_xpos[self.tip]))
        tilt = float(np.degrees(np.arccos(np.clip(R[:, 0] @ DOWN, -1.0, 1.0))))
        return err, tilt, R

    def solve_ik(self, pos, jaw_deg: float | None = None):
        """Find joint angles that put the gripper tip at `pos`, pointing down as far as the joints allow.

        `jaw_deg` turns the jaws: 0 means they close along x, 90 along y. None leaves the wrist alone.
        Returns (joint_angles, position_error_in_metres, tilt_from_vertical_in_degrees).
        """
        pos = np.asarray(pos, dtype=float)
        facing = np.arctan2(pos[1], pos[0] - 0.039)                    # turn the base toward the target
        roll = self.d.qpos[self.qadr[4]]
        seeds = [self.d.qpos[self.qadr].copy(), IK_SEED.copy(), np.array([-facing, 0.0, 0.5, 1.2, roll])]
        best = None
        for seed in seeds:
            seed[4] = roll
            q = self._reach(pos, seed)
            err, tilt, _ = self._pose_error(pos, q)
            if best is None or (err < 1e-3 and tilt < best[2] - 0.5) or err < best[1] - 1e-3:
                best = (q, err, tilt)
            if err < 1e-3 and tilt < 2.0:
                break
        q, err, tilt = best
        if jaw_deg is not None:
            want = np.deg2rad(jaw_deg)
            for _ in range(4):                                         # turn the wrist, then re-reach
                _, _, R = self._pose_error(pos, q)
                have = np.arctan2(R[1, 2], R[0, 2])                    # direction the jaws close along
                turn = (want - have + np.pi / 2) % np.pi - np.pi / 2   # jaws are symmetric: shortest turn
                jacr = np.zeros((3, self.m.nv))
                mujoco.mj_jacSite(self.m, self._scratch, None, jacr, self.tip)
                about_vertical = jacr[2, self.dadr[4]]                 # how much wrist roll turns the jaws
                if abs(turn) < np.deg2rad(0.5) or abs(about_vertical) < 0.2:
                    break
                new_roll = q[4] + turn / about_vertical
                if not self.lo[4] <= new_roll <= self.hi[4]:           # go the other way round
                    new_roll = q[4] + (turn - np.sign(turn) * np.pi) / about_vertical
                q[4] = np.clip(new_roll, self.lo[4], self.hi[4])
                q = self._reach(pos, q, iters=80)
            err, tilt, _ = self._pose_error(pos, q)
        return q, err, tilt

    # ----------------------------------------------------------------- actions
    def _drive(self, q: np.ndarray, seconds: float) -> None:
        start = self.d.ctrl[self.arm_act].copy()
        n = int(round(seconds / self.m.opt.timestep))
        for i in range(n):
            a = (i + 1) / n
            a = a * a * (3 - 2 * a)                                    # ease in and out
            self.d.ctrl[self.arm_act] = (1 - a) * start + a * q
            mujoco.mj_step(self.m, self.d)
        self.step(0.4)

    def move_to(self, x: float, y: float, z: float, jaw_deg: float | None = None, seconds: float = 1.5) -> dict:
        """Move the gripper tip to (x, y, z).

        `jaw_deg` turns the jaws: 0 closes them along x, 90 along y. Leave it out to keep the jaws as they are.
        """
        if jaw_deg is not None:
            self.jaw_deg = float(jaw_deg)
        q, err, tilt = self.solve_ik([x, y, z], self.jaw_deg)
        if err > 0.005:
            return {"ok": False, "reason": "out of reach", "gripper": self.gripper_pos()}
        self._drive(q, seconds)
        return {"ok": True, "gripper": self.gripper_pos(), "tilt_deg": round(tilt, 1)}

    def grip(self, close: bool) -> dict:
        self.d.ctrl[self.grip_act] = GRIPPER_CLOSED if close else GRIPPER_OPEN
        self.step(0.6)
        jaw = float(self.d.qpos[self.grip_qadr])
        return {"closed": close, "holding_something": bool(close and jaw > GRIPPER_CLOSED + 0.08)}

    def home(self) -> dict:
        """Park the arm up and back so it does not block the cameras."""
        self._drive(self.home_q, 1.5)
        return {"ok": True, "gripper": self.gripper_pos()}
