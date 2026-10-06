import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, home of the shared packages
import wsl_cuda  # noqa: F401  (must come before genesis)
import argparse
import os
import pickle
import time
import imageio
import numpy as np
import torch
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from envs import Go2EnvConstruction
from navigation.goal_click_plugin import GoalClickPlugin
from navigation import Navigator, OccupancyGrid, VelocityController

# Debug drawing: height above the ground and RGBA color of each element
DRAW_HEIGHT = 0.08
PATH_COLOR = (1.0, 0.45, 0.0, 0.9)
WAYPOINT_COLOR = (1.0, 0.85, 0.2, 0.7)
TARGET_COLOR = (0.1, 0.6, 1.0, 0.9)
GOAL_COLOR = (0.1, 0.85, 0.3, 0.9)

# A robot whose base stays this low (m) for this many steps is lying on the ground
FALLEN_HEIGHT = 0.15
FALLEN_STEPS = 50

# Recording camera offset from the robot in follow view (m)
FOLLOW_CAM_OFFSET = np.array([2.5, 0.5, 3.0])


def main():
    parser = argparse.ArgumentParser(description="Click anywhere in the construction site and the Go2 walks there")
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking-v2-rough-random")
    parser.add_argument("-c", "--ckpt", type=int, default=34996)
    parser.add_argument("-d", "--device", type=str, default="cuda:0", choices=["cuda:0", "cpu"])
    parser.add_argument("--site_size", type=float, default=12.0, help="Side of the walled site in meters")
    parser.add_argument("--num_obstacles", type=int, default=25,
                        help="Obstacles to place; fewer fit if the site is full")
    parser.add_argument("--obstacle_seed", type=int, default=0)
    parser.add_argument("--min_gap", type=float, default=0.8, help="Minimum gap between obstacles in meters")
    parser.add_argument("--roughness", type=float, default=0.04, help="Max bump height in meters")
    parser.add_argument("--terrain_seed", type=int, default=0)
    parser.add_argument("--inflation", type=float, default=0.25,
                        help="Clearance kept between the robot's center and obstacles in meters")
    parser.add_argument("--max_speed", type=float, default=1.0, help="Top forward speed in m/s")
    parser.add_argument("--no_realtime", action="store_true", help="Simulate as fast as possible")
    parser.add_argument("--record", type=str, default=None, metavar="PATH",
                        help="Record the run to this MP4 file (e.g. nav.mp4)")
    parser.add_argument("--record_view", type=str, default="follow", choices=["follow", "overview"],
                        help="Recording camera: follow the robot, or a fixed view of the whole site")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(f"CUDA was requested with {device}, but CUDA is not available")
    gs.init(backend=gs.gpu if device.type == "cuda" else gs.cpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}
    env_cfg["termination_if_roll_greater_than"] = 50  # degree
    env_cfg["termination_if_pitch_greater_than"] = 50  # degree

    env = Go2EnvConstruction(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        device=device,
        add_camera=args.record is not None,
        camera_gui=False,  # the viewer is already open, no extra camera window
        site_size=args.site_size,
        num_obstacles=args.num_obstacles,
        obstacle_seed=args.obstacle_seed,
        min_gap=args.min_gap,
        roughness=args.roughness,
        seed=args.terrain_seed,
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=str(device))
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=str(device))

    # The overhead camera: the site as seen from the environment's own state
    grid = OccupancyGrid.from_env(env, inflation=args.inflation)
    navigator = Navigator(grid, controller=VelocityController(max_lin_vel_x=args.max_speed))
    clicks = env.scene.viewer.add_plugin(GoalClickPlugin())
    s = args.site_size
    env.scene.viewer.set_camera_pose(pos=np.array([0.9 * s, -0.9 * s, 0.75 * s]), lookat=np.zeros(3))
    print(f"Site {s:g} x {s:g} m with {len(env.obstacles)} obstacles.")
    print("Click on the ground to send the robot there. Drag to orbit, scroll to zoom, close the window to exit.")

    writer = None
    if args.record is not None:
        if args.record_view == "overview":
            env.cam_0.set_pose(pos=(0.9 * s, -0.9 * s, 0.75 * s), lookat=(0.0, 0.0, 0.0))
        writer = imageio.get_writer(args.record, fps=int(round(1 / env.dt)), codec="libx264", quality=8,
                                    macro_block_size=1, ffmpeg_log_level="error")
        print(f"Recording the {args.record_view} view to {args.record}")

    # No episode timeout, so every reset is a real fall.
    env.max_episode_length = 10**9
    obs = env.reset().to(device)
    height = reward_cfg["base_height_target"]
    goal_start_step, low_steps, step = 0, 0, 0
    target_marker, drawn_target_idx = None, None
    try:
        with torch.no_grad():
            while env.scene.viewer.is_alive():
                tick = time.perf_counter()
                position = env.base_pos[0, :2].cpu().numpy()

                goal = clicks.pop_goal()
                if goal is not None:
                    target_marker, drawn_target_idx = None, None
                    env.scene.clear_debug_objects()
                    if navigator.set_goal(position, goal):
                        goal_start_step = step
                        draw_route(env.scene, navigator)
                        report_goal(goal, navigator)
                    else:
                        print(f"Goal ({goal[0]:+.2f}, {goal[1]:+.2f}) cannot be reached")

                was_reached = navigator.reached
                vx, vy, wz = navigator.compute_command(position, env.base_euler[0, 2].item())
                if navigator.reached and not was_reached:
                    seconds = (step - goal_start_step) * env.dt
                    print(f"  reached in {seconds:.1f} s, {navigator.num_replans} replans")
                    env.scene.clear_debug_objects()
                    target_marker, drawn_target_idx = None, None

                # Highlight the current intermediate target
                if not navigator.reached and navigator.target is not None and navigator.target_idx != drawn_target_idx:
                    if target_marker is not None:
                        env.scene.clear_debug_object(target_marker)
                    target_marker = env.scene.draw_debug_sphere(lift(navigator.target), radius=0.07,
                                                                color=TARGET_COLOR)
                    drawn_target_idx = navigator.target_idx

                env.commands = torch.tensor([[vx, vy, wz, height, 0.0]], dtype=torch.float, device=env.device)
                obs, _, dones, _ = env.step(policy(obs).to(env.device), is_train=False)
                obs = obs.to(device)
                step += 1

                # A robot lying flat never exceeds the tilt limit, so also reset when the base stays low.
                low_steps = low_steps + 1 if env.base_pos[0, 2].item() < FALLEN_HEIGHT else 0
                fell = bool(dones.any()) or low_steps >= FALLEN_STEPS
                if low_steps >= FALLEN_STEPS:
                    obs = env.reset().to(device)
                    low_steps = 0
                if fell:
                    print("  fell, respawned at the start")
                    if navigator.goal is not None and not navigator.reached:
                        # Head for the same goal again from the spawn point
                        env.scene.clear_debug_objects()
                        target_marker, drawn_target_idx = None, None
                        navigator.set_goal(env.base_pos[0, :2].cpu().numpy(), navigator.goal)
                        draw_route(env.scene, navigator)

                if writer is not None:
                    if args.record_view == "follow":
                        base_pos = env.base_pos[0].cpu().numpy()
                        env.cam_0.set_pose(pos=base_pos + FOLLOW_CAM_OFFSET, lookat=base_pos)
                    writer.append_data(np.ascontiguousarray(env.cam_0.render(rgb=True)[0]))

                if not args.no_realtime:
                    time.sleep(max(0.0, env.dt - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        # Finalize the video even if the run is interrupted with Ctrl+C.
        if writer is not None:
            writer.close()
            print(f"Saved video to {args.record}")


# Point (x, y) raised to the drawing height
def lift(point):
    return np.array([point[0], point[1], DRAW_HEIGHT])


# Draw the planned path, its intermediate targets and the goal pole
def draw_route(scene, navigator):
    path = np.array([lift(p) for p in navigator.path])
    scene.draw_debug_trajectory(path, radius=0.015, color=PATH_COLOR)
    if len(navigator.waypoints) > 1:
        scene.draw_debug_spheres(np.array([lift(p) for p in navigator.waypoints[:-1]]), radius=0.035,
                                 color=WAYPOINT_COLOR)
    end = path[-1]
    scene.draw_debug_line(end, end + np.array([0.0, 0.0, 1.0]), radius=0.02, color=GOAL_COLOR)
    scene.draw_debug_sphere(end + np.array([0.0, 0.0, 1.0]), radius=0.08, color=GOAL_COLOR)


def report_goal(goal, navigator):
    end = navigator.path[-1]
    length = np.sum(np.linalg.norm(np.diff(navigator.path, axis=0), axis=1))
    text = f"Goal ({goal[0]:+.2f}, {goal[1]:+.2f})"
    if np.linalg.norm(end - goal) > 1e-6:
        text += f" is too close to an obstacle, heading to the nearest free spot ({end[0]:+.2f}, {end[1]:+.2f})"
    print(f"{text}\n  path {length:.1f} m, {len(navigator.waypoints)} targets")


if __name__ == "__main__":
    main()
