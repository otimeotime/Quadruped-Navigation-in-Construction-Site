import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, home of the shared packages
import wsl_cuda  # noqa: F401  (must come before genesis)
import os
import argparse
import contextlib
import datetime
import io
import pickle
import shutil
import statistics
from envs import Go2Env, Go2EnvRough
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.logger import Logger
import genesis as gs


class CompactLogger(Logger):
    """rsl_rl Logger that keeps all TensorBoard logging but prints one line per
    iteration, plus a reward breakdown every `breakdown_interval` iterations."""

    breakdown_interval = 100

    def log(self, it, start_it, total_it, collect_time, learn_time, loss_dict, learning_rate, action_std, **kwargs):
        if self.writer is None:
            return
        # The parent clears the episode extras, so average the reward terms first.
        terms = {}
        for ep_info in self.ep_extras:
            for key, value in ep_info.items():
                terms.setdefault(key, []).append(float(value))
        # Let the parent write TensorBoard scalars, but drop its long console block.
        with contextlib.redirect_stdout(io.StringIO()):
            super().log(it, start_it, total_it, collect_time, learn_time, loss_dict, learning_rate, action_std, **kwargs)

        done_it = it + 1 - start_it
        eta = self.tot_time / done_it * (total_it - start_it - done_it)
        fps = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size / (collect_time + learn_time)
        rew = f"{statistics.mean(self.rewbuffer):7.2f}" if self.rewbuffer else "      -"
        ep_len = f"{statistics.mean(self.lenbuffer):5.0f}" if self.lenbuffer else "    -"
        width = len(str(total_it))
        print(
            f"\033[1mit {it:>{width}}/{total_it}\033[0m"
            f" | rew {rew} | len {ep_len}"
            f" | std {action_std.mean().item():.2f} | lr {learning_rate:.1e}"
            f" | v-loss {loss_dict.get('value', float('nan')):.4f}"
            f" | {fps / 1000:4.0f}k sps"
            f" | {datetime.timedelta(seconds=int(self.tot_time))} < {datetime.timedelta(seconds=int(eta))}"
        )
        if it % self.breakdown_interval == 0 or done_it == total_it - start_it:
            # Skip terms that are inactive (e.g. jump rewards while jumping is disabled).
            parts = [
                f"{key.removeprefix('rew_')} {statistics.mean(values):+.3f}"
                for key, values in terms.items()
                if abs(statistics.mean(values)) > 1e-6
            ]
            print("    reward terms: " + "  ".join(parts))

def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.005,  # lower so the action noise can shrink
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "activation": "elu",
            "hidden_dims": [512, 256, 128],
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "activation": "elu",
            "hidden_dims": [512, 256, 128],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        # base linear/angular velocity, projected gravity, all five commands,
        # joint position/velocity, and previous action
        "num_obs": 50,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "jump_upward_velocity": 1.2,  
        "jump_reward_steps": 50,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,  # raised from 0.2 so turning is worth learning
            "lin_vel_z": -1.0,
            "base_height": -100.0,  # raised from -50 so height commands are followed
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            # "jump": 4.0,
            "jump_height_tracking": 0.5,
            "jump_height_achievement": 10,
            "jump_speed" : 1.0,
            "jump_landing": 0.08,
        },
    }
    command_cfg = {
        "num_commands": 5,  # [lin_vel_x, lin_vel_y, ang_vel, height, jump]
        "lin_vel_x_range": [-1.0, 3.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-1.0, 1.0],
        # "lin_vel_x_range": [0.0, 0.0],
        # "lin_vel_y_range": [0.0, 0.0],
        # "ang_vel_range": [0.0, 0.0],
        "height_range": [0.2, 0.4],
        "jump_range": [0.5, 1.5],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("-i", "--max_iterations", type=int, default=10000)
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="device to use: 'cpu' or 'cuda:0'")
    parser.add_argument("--resume", type=str, default=None, help="experiment name to fine-tune from, e.g. go2-walking")
    parser.add_argument("--ckpt", type=int, default=None, help="checkpoint iteration to fine-tune from (with --resume)")
    parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "rough", "random"],
                        help="flat: plane; rough: one fixed bump field; "
                             "random: tiles of random roughness with random spawns (domain randomization)")
    parser.add_argument("--roughness", type=float, default=0.04,
                        help="Max bump height in meters (heights span +/- this value); "
                             "with random terrain, each tile's roughness is drawn from [0, roughness]")
    parser.add_argument("--terrain_seed", type=int, default=None,
                        help="Terrain seed (default: 0 for rough, a new one every run for random)")
    args = parser.parse_args()
    if args.resume is not None:
        if args.ckpt is None:
            parser.error("--resume requires --ckpt")
        # The log directory of exp_name is wiped below, so it must not be the source.
        if args.resume == args.exp_name:
            parser.error("--resume must differ from --exp_name, otherwise the source checkpoint is deleted")
        resume_path = f"logs/{args.resume}/model_{args.ckpt}.pt"
        if not os.path.isfile(resume_path):
            parser.error(f"checkpoint not found: {resume_path}")

    backend = gs.constants.backend.gpu if args.device.lower() == "cuda:0" else gs.constants.backend.cpu
    gs.init(logging_level="warning", backend=backend)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    # Record the training ground in cfgs.pkl; Go2Env ignores these keys.
    env_cfg["terrain"] = args.terrain
    if args.terrain != "flat":
        env_cfg["roughness"] = args.roughness

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env_kwargs = dict(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=args.device,
    )
    if args.terrain == "flat":
        env = Go2Env(**env_kwargs)
    else:
        seed = args.terrain_seed
        if seed is None and args.terrain == "rough":
            seed = 0  # the same bump field every run, so results are comparable
        env = Go2EnvRough(roughness=args.roughness, randomize=args.terrain == "random", seed=seed, **env_kwargs)
        print(f"Terrain: {args.terrain}, roughness {args.roughness} m, seed {env.terrain_seed}")
        env_cfg["terrain_seed"] = env.terrain_seed

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)
    # Swap in the compact console output; the logger's state is unchanged.
    runner.logger.__class__ = CompactLogger
    if args.resume is not None:
        # Loads the actor, critic and optimizer; iterations continue from the checkpoint.
        runner.load(resume_path)
        print(f"Fine-tuning from {resume_path}")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()
