# Quadruped Navigation in Construction Site

> A personal locomotion project to get familiar with robotics.
> The project contains 2 parts: training a locomotion policy + navigation program.

---

## 1. Project overview
The goal of this project is to make a **quadruped** able to walk freely in a simulated construction site.
At the moment, the walking policy is trained from scratch. And there will be a navigation script that helps the robot navigates around the site.

**Demos**
- [Locomotion policy on random rough terrain (bumps up to 0.08 m)](go2_eval_random_0.08.mp4): the evaluation tour of every command type.
- [Navigation in the construction site](navigation_demo.mp4): click a goal and the robot plans a path around the obstacles and walks there.


## 2. Quick start
Requirements: Linux or WSL2, Python 3.12, and an NVIDIA GPU with driver >= 570. The simulation scripts also run on the CPU with `-d cpu`, just much slower.
Building the rough terrain and the construction site takes about 5 GB of RAM. WSL2 only gets half of the Windows RAM by default, so on a 16 GB machine raise it in `%UserProfile%\.wslconfig` (`[wsl2]` then `memory=12GB`) and run `wsl --shutdown`.

**Install**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
No CUDA toolkit is needed: the torch wheel ships its own CUDA runtime.

Run every command from the repository root. Scripts read and write checkpoints in `logs/<exp_name>/`, and that folder is not tracked by git, so you need to train a policy first.

**Train the walking policy**
```bash
# Flat ground
python src/train/go2_train.py -e go2-walking -i 10000

# Random rough terrain, fine-tuned from the flat policy (this is what the navigation uses)
python src/train/go2_train.py -e go2-walking-rough --terrain random --roughness 0.08 \
    --resume go2-walking --ckpt 9999 -i 25000
```
A checkpoint `model_<iteration>.pt` is saved every 100 iterations, plus one at the last iteration. Watch the training curves with `tensorboard --logdir logs`.

**Evaluate the policy**
Pretrained policy for who just want to play around the environment

```bash
# Open a viewer and run a tour of every command type (walk, strafe, turn, crouch...)
python src/train/go2_eval.py -e go2-walking -c 9999

# Headless: record the tour to an MP4 instead
python src/train/go2_eval.py -e go2-walking-rough -c 34998 --terrain random --headless --video tour.mp4
```

**Drive it with the keyboard**
```bash
python src/train/go2_eval_teleop.py -e go2-walking -c 9999
```
| Key | Command |
|---|---|
| `w` / `s` | forward / backward speed +- 0.1 m/s |
| `a` / `d` | left / right speed +- 0.1 m/s |
| `q` / `e` | turn left / right +- 0.1 rad/s |
| `r` / `f` | raise / lower the body by 0.1 m |
| `j` | jump; `u` / `m` raise / lower the jump height |
| `Esc` or `8` | quit |

With `--save-data True`, the camera frames and commands are saved to `images_buffer.pkl` and `commands_buffer.pkl` when you quit, and `python src/train/create_video_with_overlay.py` turns them into `output_video.mp4` with a joystick overlay.

**Navigate the construction site**
The pretrained policy is located in logs/go2-walking-v2-rough-random.
```bash
# Look at the site without a robot policy (--headless saves PNG views instead)
python src/navigation/view_construction_site.py

# Click on the ground and the robot walks there
python src/navigation/go2_eval_navigation.py -e go2-walking-rough -c 34998
```
In the navigation viewer, a left click on the ground sets the goal. Drag to orbit, scroll to zoom, and close the window to exit. The planned path and its intermediate targets are drawn in the scene.

---

## 3. Configuration knobs
**Command-line flags.** Run any script with `-h` for the full list.

| Flag | Scripts | Meaning |
|---|---|---|
| `-e`, `--exp_name` | all | Experiment name; checkpoints live in `logs/<exp_name>/`. Training wipes this folder first. |
| `-c`, `--ckpt` | eval, teleop, navigation | Checkpoint iteration to load (`model_<ckpt>.pt`) |
| `-d`, `--device` | all | `cuda:0` (default) or `cpu` |
| `-B`, `--num_envs` | train | Parallel environments (default 4096; lower it on a small GPU) |
| `-i`, `--max_iterations` | train | PPO iterations to run |
| `--resume`, `--ckpt` | train | Fine-tune from another experiment's checkpoint |
| `--terrain` | train, eval | `flat`, `rough` (one fixed bump field) or `random` (tiles of random roughness) |
| `--roughness` | train, eval, site scripts | Max bump height in meters |
| `--terrain_seed` | train, eval, site scripts | Seed of the bump field |
| `--commands` | eval | `tour` (every command type) or `sweep` (forward speed only) |
| `--headless`, `--video`, `--steps` | eval | Record an MP4 instead of opening a viewer |
| `--site_size`, `--num_obstacles`, `--obstacle_seed`, `--min_gap` | site scripts | Size of the walled site, and how many obstacles are placed and how far apart |
| `--inflation` | navigation | Clearance kept between the robot's center and obstacles, in meters |
| `--max_speed` | navigation | Top forward speed in m/s |
| `--no_realtime` | navigation | Simulate as fast as possible |

**Training configuration.** Edit the dictionaries in [src/train/go2_train.py](src/train/go2_train.py):
- `get_train_cfg()`: PPO hyperparameters and the actor/critic network sizes.
- `get_cfgs()`: the environment (PD gains, default joint angles, termination angles, episode length), the observation scales, the reward terms and their weights, and the sampling ranges of the commands (speed, turn rate, body height, jump height).

These are saved to `logs/<exp_name>/cfgs.pkl` at training time, and the eval and navigation scripts load them from there, so a checkpoint always runs with the settings it was trained with.

---

## 4. Repository layout
```
src/
├── wsl_cuda.py                   # Lets Genesis find the GPU driver on WSL2; imported by every GPU script
├── envs/                         # Genesis environments of the Unitree Go2
│   ├── go2_env.py                #   Flat ground: observations, rewards, commands, resets
│   ├── go2_env_rough.py          #   Same robot on a bumpy height field
│   └── go2_env_construction.py   #   Rough ground inside walls, filled with blocks, barriers and pipes
├── train/                        # Locomotion policy
│   ├── go2_train.py              #   PPO training (rsl_rl) and all training configs
│   ├── go2_eval.py               #   Command tour in a viewer, or recorded to MP4
│   ├── go2_eval_teleop.py        #   Keyboard teleoperation
│   └── create_video_with_overlay.py  # Teleop recording -> MP4 with a joystick overlay
└── navigation/                   # Goal -> path -> velocity commands for the policy
    ├── occupancy_grid.py         #   2D grid of the site, read from the environment (idealized overhead camera)
    ├── path_planner.py           #   A* on the grid, then string pulling to straighten the path
    ├── waypoint_generator.py     #   Splits the path into evenly spaced intermediate targets
    ├── velocity_controller.py    #   Heading controller: turn toward the target, slow down while turning
    ├── navigator.py              #   Ties the three steps above together and replans when needed
    ├── goal_sampler.py           #   Random reachable goals in the site
    ├── goal_click_plugin.py      #   Viewer plugin: a mouse click on the ground becomes a goal
    ├── go2_eval_navigation.py    #   Click-to-navigate demo
    └── view_construction_site.py #   Look at the site, or save PNG views of it
logs/                             # Checkpoints and TensorBoard logs (not tracked by git)
requirements.txt                  # Pinned Python dependencies
```

---

## 5. Limitations
These are limitations of the current project situation, and will be the future directions to work on.
1. Online RL only: There should be another offline RL approach for this project to make a comparison.
2. The navigation is not practical: To make the quadruped walk to the target coordinate, this project uses A* algorithm with full environment information. Which is not realistic, since it should be percepted from the quadruped sensors and the environment should be partially observable.

---

## 6. License
MIT our our code. Pre-trained policy and Go2 meshes inherit upstream licences.