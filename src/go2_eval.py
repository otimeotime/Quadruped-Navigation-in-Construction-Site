import argparse
import os
import pickle
import torch
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner
import numpy as np
import genesis as gs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-c", "--ckpt", type=int, default=200)
    parser.add_argument("-d", "--device", type=str, default="cuda:0", choices=["cuda:0", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(f"CUDA was requested with {device}, but CUDA is not available")
    gs.init(backend=gs.constants.backend.gpu if device.type == "cuda" else gs.constants.backend.cpu)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env_cfg["termination_if_roll_greater_than"] =  50  # degree
    env_cfg["termination_if_pitch_greater_than"] = 50  # degree

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        device=device,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=str(device))
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=str(device))

    obs = env.reset().to(device)
    
    env.commands = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0]], device=env.device)
    iter = 0
    lin_x_range = [0.5, 4.0]
    with torch.no_grad():
        while True:
            actions = policy(obs)
            # lin_x = lin_x_range[0] + (lin_x_range[1] - lin_x_range[0]) * (iter % 300) / 300
            lin_x = lin_x_range[0] + (lin_x_range[1] - lin_x_range[0]) * (np.sin(2 * np.pi * iter / 600) + 1) / 2
            lin_x = float(lin_x)
            print(lin_x)
            env.commands = torch.tensor([[lin_x, 0.0, 0.0, 0.0, 0.0]], device=env.device)
            obs, rews, dones, infos = env.step(actions.to(env.device), is_train=False)
            obs = obs.to(device)
            iter += 1
            if dones.any():
                iter = 0

if __name__ == "__main__":
    main()
