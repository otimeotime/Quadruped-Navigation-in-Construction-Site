import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, home of the shared packages
import argparse
import cv2
import torch
import genesis as gs
from envs import Go2EnvConstruction
from train.go2_train import get_cfgs


def main():
    parser = argparse.ArgumentParser(description="Take a look at the construction site environment")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", choices=["cuda:0", "cpu"])
    parser.add_argument("--headless", action="store_true", help="Save overview images instead of opening a viewer")
    parser.add_argument("--out", type=str, default="construction_site", help="Image name prefix in headless mode")
    parser.add_argument("--site_size", type=float, default=12.0, help="Side of the walled site in meters")
    parser.add_argument("--num_obstacles", type=int, default=25,
                        help="Obstacles to place; fewer fit if the site is full")
    parser.add_argument("--obstacle_seed", type=int, default=0)
    parser.add_argument("--min_gap", type=float, default=0.8, help="Minimum gap between obstacles in meters")
    parser.add_argument("--roughness", type=float, default=0.04, help="Max bump height in meters")
    parser.add_argument("--terrain_seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    gs.init(backend=gs.gpu if device.type == "cuda" else gs.cpu, logging_level="warning")

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    reward_cfg["reward_scales"] = {}
    env = Go2EnvConstruction(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.headless,
        device=device,
        add_camera=args.headless,
        site_size=args.site_size,
        num_obstacles=args.num_obstacles,
        obstacle_seed=args.obstacle_seed,
        min_gap=args.min_gap,
        roughness=args.roughness,
        seed=args.terrain_seed,
    )
    print(f"Site {args.site_size:g} x {args.site_size:g} m, {len(env.obstacles)} of {args.num_obstacles} obstacles placed:")
    for obstacle in env.obstacles:
        size = " x ".join(f"{v:.2f}" for v in obstacle["size"])
        print(f"  {obstacle['kind']:<15} at ({obstacle['x']:+6.2f}, {obstacle['y']:+6.2f})  size {size} m")

    env.reset()
    # Zero actions hold the default standing pose, so the robot just stands at the spawn point.
    zero_actions = torch.zeros((1, env.num_actions), device=env.device)
    s = args.site_size
    overview_pos, overview_lookat = (0.9 * s, -0.9 * s, 0.75 * s), (0.0, 0.0, 0.0)

    if args.headless:
        for _ in range(50):  # let the robot settle on the ground
            env.step(zero_actions, is_train=False)
        # (position, look-at point, up direction) of each view
        views = {
            "overview": (overview_pos, overview_lookat, (0.0, 0.0, 1.0)),
            "top": ((0.0, 0.0, 1.5 * s), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            "robot": ((2.5, -2.5, 1.5), (0.0, 0.0, 0.3), (0.0, 0.0, 1.0)),
        }
        for name, (pos, lookat, up) in views.items():
            env.cam_0.set_pose(pos=pos, lookat=lookat, up=up)
            rgb = env.cam_0.render(rgb=True)[0]
            path = f"{args.out}_{name}.png"
            cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(f"Saved {path}")
        return

    env.scene.viewer.set_camera_pose(pos=overview_pos, lookat=overview_lookat)
    print("Viewer open: drag to orbit, scroll to zoom. Close the window or press Ctrl+C to exit.")
    try:
        with torch.no_grad():
            while env.scene.viewer.is_alive():
                env.step(zero_actions, is_train=False)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
