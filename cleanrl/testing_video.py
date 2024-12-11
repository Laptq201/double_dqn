# import gymnasium as gym
# import torch
# from stable_baselines3.common.atari_wrappers import (
#     ClipRewardEnv,
#     EpisodicLifeEnv,
#     FireResetEnv,
#     MaxAndSkipEnv,
#     NoopResetEnv,
# )
# from torch import nn
# import random
# import numpy as np

# # Define QNetwork as used during training


# class QNetwork(nn.Module):
#     def __init__(self, action_space_n):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Conv2d(4, 32, 8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, stride=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(3136, 512),
#             nn.ReLU(),
#             nn.Linear(512, action_space_n),
#         )

#     def forward(self, x):
#         return self.network(x / 255.0)


# # Load the environment setup function
# def make_env(env_id):
#     seed = 22520750
#     env = gym.make(env_id, render_mode="human")
#     # env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

#     env = gym.wrappers.RecordEpisodeStatistics(env)

#     env = NoopResetEnv(env, noop_max=30)
#     env = MaxAndSkipEnv(env, skip=4)
#     env = EpisodicLifeEnv(env)
#     if "FIRE" in env.unwrapped.get_action_meanings():
#         env = FireResetEnv(env)
#     env = ClipRewardEnv(env)
#     env = gym.wrappers.ResizeObservation(env, (84, 84))
#     env = gym.wrappers.GrayScaleObservation(env)
#     env = gym.wrappers.FrameStack(env, 4)

#     env.action_space.seed(seed)
#     return env


# # Load the trained model
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # env_id = "BreakoutNoFrameskip-v4"
# # env = make_env(env_id)
# # q_network = QNetwork(env.action_space.n).to(device)

# # # Load weights
# # model_path = "/home/lapquang/Downloads/dqn_atari.cleanrl_model"
# # q_network.load_state_dict(torch.load(model_path, map_location=device))
# # q_network.eval()

# # # Watch the agent play
# # # Watch the agent play
# # # For Gymnasium, reset returns a tuple (obs, info)
# # epsilon = 0.05  # Small exploration rate for evaluation
# # obs = env.reset(seed=1)[0]
# # done = False
# # while not done:
# #     env.render()
# #     with torch.no_grad():
# #         if random.random() < epsilon:
# #             action = env.action_space.sample()  # Random action
# #         else:
# #             obs_tensor = torch.tensor(
# #                 obs, dtype=torch.float32).unsqueeze(0).to(device)
# #             q_values = q_network(obs_tensor)
# #             action = torch.argmax(q_values, dim=1).item()
# #     obs, reward, done, _, info = env.step(action)

# # env.close()
# def make_env_for_eval(env_id, seed):
#     env = gym.make(env_id, render_mode="human")  # Render for human viewing
#     env = gym.wrappers.RecordEpisodeStatistics(env)  # Track episode stats
#     env = NoopResetEnv(env, noop_max=30)
#     env = MaxAndSkipEnv(env, skip=4)
#     if "FIRE" in env.unwrapped.get_action_meanings():
#         env = FireResetEnv(env)
#     env = ClipRewardEnv(env)
#     env = gym.wrappers.ResizeObservation(env, (84, 84))
#     env = gym.wrappers.GrayScaleObservation(env)
#     env = gym.wrappers.FrameStack(env, 4)
#     env.action_space.seed(seed)
#     return env

# # env.close()


# def evaluate_dqn(env, q_network, device, num_episodes=10):
#     total_rewards = []
#     for episode in range(num_episodes):
#         obs = env.reset(seed=episode)[0]
#         done = False
#         total_reward = 0
#         i = 0
#         while not done:
#             env.render()
#             with torch.no_grad():
#                 obs_tensor = torch.tensor(
#                     obs, dtype=torch.float32).unsqueeze(0).to(device)
#                 q_values = q_network(obs_tensor)
#                 action = torch.argmax(q_values, dim=1).item()
#             obs, reward, done, _, info = env.step(action)
#             total_reward += reward
#         total_rewards.append(total_reward)
#         print(f"Episode {episode + 1}: Reward = {total_reward}")
#     return np.mean(total_rewards), np.std(total_rewards)


# # Example usage
# if __name__ == "__main__":
#     env = make_env_for_eval("BreakoutNoFrameskip-v4", seed=22520750)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     q_network = QNetwork(env.action_space.n).to(device)
#     model_path = "/home/lapquang/Downloads/dqn_atari.cleanrl_model"
#     q_network.load_state_dict(torch.load(model_path, map_location=device))
#     q_network.eval()
#     avg_reward, std_reward = evaluate_dqn(
#         env, q_network, device, num_episodes=10)
#     print(f"Average Reward over 10 episodes: {avg_reward} ± {std_reward}")
#     env.close()
# # import random
# # from typing import Callable

# # import gymnasium as gym
# # import numpy as np
# # import torch


# # def evaluate(
# #     model_path: str,
# #     make_env: Callable,
# #     env_id: str,
# #     eval_episode: int,
# #     run_name: str,
# #     Model: torch.nn.Module,
# #     device: torch.device = torch.device("cpu"),
# #     epsilon: float = 0.05,
# #     capture_video: bool = True
# # ):
# #     envs = gym.vector.SyncVectorEnv(
# #         [make_env(env_id, 0, 0, capture_video, run_name)])
# #     model = Model(envs).to(device)
# #     model.load_state_dict(torch.load(model_path, map_location=device))
# #     model.eval()

# #     obs, _ = envs.reset()
# #     episodic_returns = []
# #     while len(episodic_returns) < eval_episode:
# #         if random.random() < epsilon:
# #             actions = np.array([envs.single_action_space.sample()
# #                                for _ in range(envs.num_envs)])
# #         else:
# #             q_values = model(torch.Tensor(obs).to(device))
# #             actions = torch.argmax(q_values, dim=1).cpu().numpy()
# #         next_obs, _, _, _, infos = envs.step(actions)
# #         if "final_info" in infos:
# #             for info in infos["final_info"]:
# #                 if "episode" not in info:
# #                     continue
# #                 print(
# #                     f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
# #                 episodic_returns += [info["episode"]["r"]]
# #         obs = next_obs

# #     return episodic_returns


# # if __name__ == "__main__":
# #     from huggingface_hub import hf_hub_download

# #     from dqn_atari import QNetwork, make_env

# #     model_path = "/home/lapquang/Downloads/dqn_atari.cleanrl_model"
# #     # model_path = ".pth"
# #     evaluate(
# #         model_path,
# #         make_env,
# #         "BreakoutNoFrameskip-v4",
# #         eval_episode=0,
# #         run_name=f"eval",
# #         Model=QNetwork,
# #         device="cpu",
# #         capture_video=False
# #     )
import gymnasium as gym
import torch
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from torch import nn
import random
import numpy as np

# Define QNetwork as used during training


class QNetwork(nn.Module):
    def __init__(self, action_space_n):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space_n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


# Load the environment setup function
def make_env(env_id, seed=22520750, record_video=False, run_name=""):
    env = gym.make(env_id, render_mode="human")

    # Wrap for recording video only every 5th episode
    if record_video:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

    env = gym.wrappers.RecordEpisodeStatistics(env)

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)

    env.action_space.seed(seed)
    return env


# Load the trained model
env = make_env("BreakoutNoFrameskip-v4", seed=22520750, record_video=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     q_network = QNetwork(env.action_space.n).to(device)
#     model_path = "/home/lapquang/Downloads/dqn_atari.cleanrl_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_network = QNetwork(action_space_n=env.action_space.n).to(device)

# Load your pre-trained model (assuming the model is saved as 'dqn_model.pth')
q_network.load_state_dict(torch.load(
    "/home/lapquang/Downloads/dqn_atari.cleanrl_model", map_location=device))
q_network.eval()

# Create the environment
env = make_env('BreakoutNoFrameskip-v4')

# Number of episodes to visualize
num_episodes = 30

# Loop over episodes to visualize the model
for episode in range(num_episodes):
    # Get the first observation from the reset
    obs = env.reset(seed=episode)[0]
    done = False
    total_reward = 0
    while not done:
        # Render the environment to visualize
        env.render()

        # Preprocess observation and feed it into the QNetwork to get Q-values
        obs_tensor = torch.tensor(
            obs, dtype=torch.float32).unsqueeze(0).to(device)

        # Get action from the QNetwork (use greedy action, i.e., pick the highest Q-value)
        with torch.no_grad():
            q_values = q_network(obs_tensor)
            action = torch.argmax(q_values, dim=1).item()

        # Take the action in the environment
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

    # Print the total reward for the episode
    if episode + 1 % 5 == 0:
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
