import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, home of the shared packages
import argparse
import os
import pickle
import cv2
import imageio
import torch
from envs import Go2Env, Go2EnvRough
from rsl_rl.runners import OnPolicyRunner
import numpy as np
import genesis as gs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-c", "--ckpt", type=int, default=200)
    parser.add_argument("-d", "--device", type=str, default="cuda:0", choices=["cuda:0", "cpu"])
    parser.add_argument("--headless", action="store_true", help="Run without a viewer and record an MP4 instead")
    parser.add_argument("--video", type=str, default="go2_eval.mp4", help="Output video path in headless mode")
    parser.add_argument("--steps", type=int, default=None,
                        help="Steps to record in headless mode (default: one full tour, or 1000 for sweep)")
    parser.add_argument("--commands", type=str, default="tour", choices=["tour", "sweep"],
                        help="tour: every command type over the full training ranges; sweep: forward speed only")
    parser.add_argument("--segment_s", type=float, default=4.0, help="Seconds per tour segment")
    parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "rough", "random"],
                        help="flat: plane; rough: one fixed bump field; "
                             "random: tiles of random roughness with random spawns (domain randomization)")
    parser.add_argument("--roughness", type=float, default=0.04,
                        help="Max bump height in meters (heights span +/- this value); "
                             "with random terrain, each tile's roughness is drawn from [0, roughness]")
    parser.add_argument("--terrain_seed", type=int, default=None,
                        help="Terrain seed (default: 0 for rough, a new one every run for random)")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(f"CUDA was requested with {device}, but CUDA is not available")
    gs.init(
        backend=gs.constants.backend.gpu if device.type == "cuda" else gs.constants.backend.cpu,
        logging_level="warning",
    )

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env_cfg["termination_if_roll_greater_than"] =  50  # degree
    env_cfg["termination_if_pitch_greater_than"] = 50  # degree

    env_kwargs = dict(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.headless,
        device=device,
        add_camera=args.headless,
    )
    if args.terrain == "flat":
        env = Go2Env(**env_kwargs)
    else:
        seed = args.terrain_seed
        if seed is None and args.terrain == "rough":
            seed = 0  # the same bump field every run, so results are comparable
        env = Go2EnvRough(roughness=args.roughness, randomize=args.terrain == "random", seed=seed, **env_kwargs)
        print(f"Terrain: {args.terrain}, roughness {args.roughness} m, seed {env.terrain_seed}")

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=str(device))
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=str(device))

    # No episode timeout during evaluation, so every reset is a real fall.
    env.max_episode_length = 10**9
    obs = env.reset().to(device)

    if args.commands == "tour":
        segments = build_tour(command_cfg, reward_cfg)
        seg_steps = int(args.segment_s / env.dt)
        cycle_steps = seg_steps * len(segments)
    else:
        segments, seg_steps, cycle_steps = None, None, 600
    num_steps = args.steps if args.steps is not None else (cycle_steps if args.commands == "tour" else 1000)

    writer = None
    if args.headless:
        # Keep the camera's original offset from the robot so it stays in frame.
        cam_offset = np.array([2.5, 0.5, 3.0])
        writer = imageio.get_writer(args.video, fps=int(round(1 / env.dt)), codec="libx264", quality=8,
                                    macro_block_size=1, ffmpeg_log_level="error")

    summary = []  # (name, command, mean achieved, falls) per tour segment
    seg_meas, seg_falls = [], 0
    low_steps = 0
    step = 0
    try:
        with torch.no_grad():
            while not args.headless or step < num_steps:
                name, cmd = command_at(step, args.commands, segments, seg_steps, cycle_steps, command_cfg, reward_cfg)
                env.commands = torch.tensor([cmd + [0.0]], device=env.device)
                obs, rews, dones, infos = env.step(policy(obs).to(env.device), is_train=False)
                obs = obs.to(device)

                achieved = [
                    env.base_lin_vel[0, 0].item(),
                    env.base_lin_vel[0, 1].item(),
                    env.base_ang_vel[0, 2].item(),
                    env.base_pos[0, 2].item(),
                ]
                # A robot lying flat never exceeds the tilt limit, so also reset
                # when the base stays near the ground.
                low_steps = low_steps + 1 if achieved[3] < 0.15 else 0
                fell = bool(dones.any()) or low_steps >= 50
                if low_steps >= 50:
                    obs = env.reset().to(device)
                    low_steps = 0
                if fell:
                    seg_falls += 1
                    print(f"step {step:5d} | {name}: fell, episode reset")

                if args.commands == "tour":
                    # Skip the first second of each segment while the gait adapts.
                    if step % seg_steps >= int(1.0 / env.dt):
                        seg_meas.append(achieved)
                    if (step + 1) % seg_steps == 0:
                        mean = np.mean(seg_meas, axis=0) if seg_meas else np.full(4, np.nan)
                        summary.append((name, cmd, mean, seg_falls))
                        print(f"{name:<22} cmd {fmt(cmd)} | achieved {fmt(mean)} | falls {seg_falls}")
                        seg_meas, seg_falls = [], 0
                elif step % 50 == 0:
                    print(f"step {step:5d} | cmd vx {cmd[0]:4.2f} | vx {achieved[0]:+5.2f} | height {achieved[3]:.2f}")

                if writer is not None:
                    base_pos = env.base_pos[0].cpu().numpy()
                    env.cam_0.set_pose(pos=base_pos + cam_offset, lookat=base_pos)
                    rgb = env.cam_0.render(rgb=True)[0]
                    writer.append_data(draw_overlay(np.ascontiguousarray(rgb), name, cmd, achieved, step, num_steps, env.dt))
                step += 1
    finally:
        # Finalize the video even if the run is interrupted with Ctrl+C.
        if writer is not None:
            writer.close()
            print(f"Saved video to {args.video}")
        if summary:
            print_summary(summary)


def build_tour(command_cfg, reward_cfg):
    """Segments covering every command over the full training ranges: (name, [vx, vy, wz, height])."""
    h0 = reward_cfg["base_height_target"]
    vx_lo, vx_hi = command_cfg["lin_vel_x_range"]
    vy_lo, vy_hi = command_cfg["lin_vel_y_range"]
    wz_lo, wz_hi = command_cfg["ang_vel_range"]
    h_lo, h_hi = command_cfg["height_range"]
    return [
        ("stand", [0.0, 0.0, 0.0, h0]),
        ("walk forward", [0.5 * vx_hi, 0.0, 0.0, h0]),
        ("run forward (max)", [vx_hi, 0.0, 0.0, h0]),
        ("walk backward (max)", [vx_lo, 0.0, 0.0, h0]),
        ("strafe left (max)", [0.0, vy_hi, 0.0, h0]),
        ("strafe right (max)", [0.0, vy_lo, 0.0, h0]),
        ("turn left (max)", [0.0, 0.0, wz_hi, h0]),
        ("turn right (max)", [0.0, 0.0, wz_lo, h0]),
        ("crouch (min height)", [0.0, 0.0, 0.0, h_lo]),
        ("stand tall (max height)", [0.0, 0.0, 0.0, h_hi]),
        ("walk + turn left", [0.5 * vx_hi, 0.0, 0.5 * wz_hi, h0]),
        ("walk + turn right", [0.5 * vx_hi, 0.0, 0.5 * wz_lo, h0]),
        ("walk + strafe left", [0.5 * vx_hi, 0.5 * vy_hi, 0.0, h0]),
        # Training halves the velocity commands at the height extremes.
        ("crouched walk", [0.25 * vx_hi, 0.0, 0.0, h_lo]),
        ("tall walk", [0.25 * vx_hi, 0.0, 0.0, h_hi]),
        ("crouched turn", [0.0, 0.0, 0.5 * wz_hi, h_lo]),
    ]


def command_at(step, mode, segments, seg_steps, cycle_steps, command_cfg, reward_cfg):
    """Return (segment name, [vx, vy, wz, height]) for this step."""
    if mode == "tour":
        return segments[(step // seg_steps) % len(segments)]
    vx_lo, vx_hi = 0.5, command_cfg["lin_vel_x_range"][1]
    lin_x = vx_lo + (vx_hi - vx_lo) * (np.sin(2 * np.pi * step / cycle_steps) + 1) / 2
    return "forward sweep", [float(lin_x), 0.0, 0.0, reward_cfg["base_height_target"]]


def fmt(values):
    vx, vy, wz, h = values
    return f"vx {vx:+5.2f} vy {vy:+5.2f} wz {wz:+5.2f} h {h:.2f}"


def draw_overlay(frame, name, cmd, achieved, step, num_steps, dt):
    """Draw the current segment and commanded vs. achieved values on a frame."""
    font, white, grey = cv2.FONT_HERSHEY_SIMPLEX, (255, 255, 255), (170, 170, 170)
    panel = frame.copy()
    cv2.rectangle(panel, (20, 20), (760, 250), (0, 0, 0), -1)
    frame = cv2.addWeighted(panel, 0.55, frame, 0.45, 0)
    cv2.putText(frame, name, (40, 70), font, 1.3, white, 3, cv2.LINE_AA)
    cv2.putText(frame, f"{'':<9}{'vx':>7}{'vy':>7}{'wz':>7}{'h':>7}", (40, 120), cv2.FONT_HERSHEY_PLAIN, 2.0, grey, 2, cv2.LINE_AA)
    for row, (label, vals) in enumerate([("command", cmd), ("actual", achieved)]):
        text = f"{label:<9}" + "".join(f"{v:>+7.2f}" for v in vals[:3]) + f"{vals[3]:>7.2f}"
        cv2.putText(frame, text, (40, 165 + 45 * row), cv2.FONT_HERSHEY_PLAIN, 2.0, white, 2, cv2.LINE_AA)
    cv2.putText(frame, f"t = {step * dt:5.1f} s / {num_steps * dt:.0f} s", (1600, 60), font, 1.0, white, 2, cv2.LINE_AA)
    return frame


def print_summary(summary):
    print()
    print(f"{'segment':<24}{'command (vx, vy, wz, h)':<34}{'achieved':<34}falls")
    for name, cmd, mean, falls in summary:
        print(f"{name:<24}{fmt(cmd):<34}{fmt(mean):<34}{falls}")


if __name__ == "__main__":
    main()
