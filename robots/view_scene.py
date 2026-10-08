"""Visualize a MuJoCo scene XML without training or ROS.

Holds the robot at the "init_state" keyframe (kinematic: no physics stepping, so it
never falls) and optionally bounces the mocap ping-pong ball. Use it to check that
the table, ball and robot are placed correctly.

    # interactive window (needs a display; WSLg works on Windows 11)
    uv run python robots/view_scene.py

    # headless / WSL without a display: write PNGs from several cameras
    MUJOCO_GL=egl uv run python robots/view_scene.py --render scene.png

    # also run physics for N seconds (robot will collapse: no controller) to
    # sanity-check table/robot collisions
    uv run python robots/view_scene.py --physics
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

DEFAULT_XML = Path(__file__).parent / "deployment" / "g1_27dof_deploy_tt.xml"


def load(xml: Path):
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def report(model, data):
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nmocap={model.nmocap} ngeom={model.ngeom}")
    for kind, name in [
        (mujoco.mjtObj.mjOBJ_GEOM, "table"),
        (mujoco.mjtObj.mjOBJ_SITE, "center"),
        (mujoco.mjtObj.mjOBJ_SITE, "pp_ball"),
        (mujoco.mjtObj.mjOBJ_SITE, "racket_contact"),
    ]:
        i = mujoco.mj_name2id(model, kind, name)
        if i < 0:
            print(f"  {name:15s} MISSING")
            continue
        pos = data.geom_xpos[i] if kind == mujoco.mjtObj.mjOBJ_GEOM else data.site_xpos[i]
        print(f"  {name:15s} world pos = {np.round(pos, 3)}")
    t = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    if t >= 0:
        top = data.geom_xpos[t][2] + model.geom_size[t][2]
        near = data.geom_xpos[t][0] - model.geom_size[t][0]
        print(f"  table top z = {top:.3f} m, near edge x = {near:.3f} m")
    # Report any initial contacts between the robot and the table.
    for c in data.contact[: data.ncon]:
        names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in (c.geom1, c.geom2)}
        if "table" in names:
            print(f"  WARNING: initial contact with table: {names}")


def table_surface(model, data):
    """Return (top_z, x_min, x_max, y_min, y_max) of the table's top face."""
    t = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    c, h = data.geom_xpos[t], model.geom_size[t]
    return c[2] + h[2], c[0] - h[0], c[0] + h[0], c[1] - h[1], c[1] + h[1]


def ball_radius(model):
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pp_ball_geom")
    return float(model.geom_size[g][0])


def ball_motion(model, data, t, period=0.8, apex=0.25):
    """Kinematic parabolic bounce that touches the table top exactly.

    The ball centre is at (table_top + radius) at each contact, so the ball surface
    meets the table surface. It travels along x over the table's length and back.
    """
    if model.nmocap == 0:
        return
    top, x0, x1, y0, y1 = table_surface(model, data)
    r = ball_radius(model)
    phase = (t / period) % 1.0
    z = top + r + 4.0 * apex * phase * (1.0 - phase)
    # triangle wave in x: far end of the table and back, staying 0.1 m inside the edges
    u = (t / (period * 6)) % 1.0
    tri = 1.0 - abs(2.0 * u - 1.0)
    x = (x0 + 0.1) + tri * (x1 - x0 - 0.2)
    y = 0.5 * (y0 + y1)
    data.mocap_pos[0] = [x, y, z]


def render(model, data, out: Path):
    import imageio.v2 as imageio  # lazy: only needed for --render

    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
    r = mujoco.Renderer(model, height=720, width=1280)
    views = {
        "side": dict(azimuth=90, elevation=-15, distance=4.5, lookat=[1.0, 0, 0.8]),
        "behind_robot": dict(azimuth=0, elevation=-20, distance=3.5, lookat=[1.2, 0, 0.8]),
        "top": dict(azimuth=90, elevation=-89, distance=5.0, lookat=[1.0, 0, 0.5]),
    }
    for name, v in views.items():
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation, cam.distance = v["azimuth"], v["elevation"], v["distance"]
        cam.lookat[:] = v["lookat"]
        r.update_scene(data, camera=cam)
        path = out.with_name(f"{out.stem}_{name}{out.suffix}")
        imageio.imwrite(path, r.render())
        print("wrote", path)
    r.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, default=DEFAULT_XML)
    ap.add_argument("--render", type=Path, default=None, help="write PNGs instead of opening a window")
    ap.add_argument("--physics", action="store_true", help="step physics (robot is uncontrolled)")
    ap.add_argument("--no-ball-motion", action="store_true")
    args = ap.parse_args()

    model, data = load(args.xml)
    report(model, data)

    if args.render:
        render(model, data, args.render)
        return

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0
            if not args.no_ball_motion:
                ball_motion(model, data, t)
            if args.physics:
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)  # kinematic: robot stays at keyframe
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()