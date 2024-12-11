# CleanRL (Clean Implementation of RL Algorithms)

Thanks for cleanrl (https://github.com/vwxyzjn/cleanrl) from CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms

This is my lab to implement Double DQN into cleanrl - UIT.


## Get started

Prerequisites:
* Python >=3.7.1,<3.11
* [Poetry 1.2.1+](https://python-poetry.org)

To run experiments locally, give the following a try:

```bash
git clone https://github.com/Laptq201/lab_double_dqn.git && cd cleanrl
poetry install

# alternatively, you could use `poetry shell` and do
# `python run cleanrl/ppo.py`
poetry run python dqn.py --seed 22520750 \
                                --no_cuda \
                                --env-id CartPole-v0 \
                                --total-timesteps 100000 \
                                --capture_video

# open another terminal and enter `cd cleanrl/cleanrl`
tensorboard --logdir runs
```

To use experiment tracking with wandb, run
```bash
wandb login # only required for the first time
poetry run python dqn.py --seed 22520750 \
                                --no_cuda \
                                --env-id CartPole-v0 \
                                --total-timesteps 100000 \
                                --track \
                                --wandb-project-name cartpole \
                                --capture_video
```

If you are not using `poetry`, you can install CleanRL with `requirements.txt`:

```bash
# core dependencies
pip install -r requirements/requirements.txt

# optional dependencies
pip install -r requirements/requirements-atari.txt
pip install -r requirements/requirements-mujoco.txt
pip install -r requirements/requirements-mujoco_py.txt
pip install -r requirements/requirements-procgen.txt
pip install -r requirements/requirements-envpool.txt
pip install -r requirements/requirements-pettingzoo.txt
pip install -r requirements/requirements-jax.txt
pip install -r requirements/requirements-docs.txt
pip install -r requirements/requirements-cloud.txt
pip install -r requirements/requirements-memory_gym.txt
```


To run training scripts in other games:
```bash
poetry shell

# classic control
python cleanrl/dqn.py --env-id CartPole-v1
python cleanrl/ddqn.py --env-id CartPole-v1


# atari
poetry install -E atari
python cleanrl/dqn_atari.py --env-id BreakoutNoFrameskip-v4
python cleanrl/ddqn_atari.py --env-id BreakoutNoFrameskip-v4

```

To run video with atari (require: parameter DQN model.pth)
- Change the directory (for save model) in the testing_video.py
```
python testing_video.py
```
