"""Nicer visuals for the watch scripts. Changes only how things look, never the physics,
and is not used for the pictures the model sees."""
import mujoco
import numpy as np


def prettify(m: mujoco.MjModel) -> None:
    q = m.vis.quality
    q.shadowsize = 8192          # sharp shadows
    q.offsamples = 8             # smooth edges in saved pictures
    m.vis.headlight.ambient[:] = 0.22
    m.vis.headlight.diffuse[:] = 0.25
    m.vis.headlight.specular[:] = 0.05
    m.vis.map.shadowclip = 0.8   # fit the shadow map to the table, not the whole world
    # One angled key light that casts shadows, like a desk lamp above and to the side
    key = np.array([0.45, 0.35, -1.0]); key /= np.linalg.norm(key)
    for i in range(m.nlight):
        m.light_dir[i] = key
        m.light_pos[i] = np.array([0.2, 0.0, 0.0]) - 1.2 * key
        m.light_castshadow[i] = 1
        m.light_diffuse[i] = 0.75
        m.light_specular[i] = 0.25
    # Glossy plastic parts and a slightly reflective mat
    for i in range(m.nmat):
        m.mat_specular[i] = 0.45
        m.mat_shininess[i] = 0.55
    m.mat_reflectance[m.material("mat").id] = 0.12
