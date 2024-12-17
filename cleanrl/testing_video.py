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


env = make_env("BreakoutNoFrameskip-v4", seed=22520670, record_video=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     q_network = QNetwork(env.action_space.n).to(device)
#     model_path = "/home/lapquang/Downloads/dqn_atari.cleanrl_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_network = QNetwork(action_space_n=env.action_space.n).to(device)

# Load your pre-trained model (assuming the model is saved as 'dqn_model.pth')
q_network.load_state_dict(torch.load(
    "/home/lapquang/Desktop/Atari-BreakOut/22520752_ddqn/ddqn_atari.cleanrl_model", map_location=device))
q_network.eval()

env = make_env('BreakoutNoFrameskip-v4')

num_episodes = 30

epsilon = 0.02  # Exploration rate 

# Loop over episodes to visualize the model
for episode in range(num_episodes):
    # Get the first observation from the reset
    obs, _ = env.reset(seed=episode)
    done = False
    total_reward = 0
    while not done:
        env.render()
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        # Action selection (epsilon-greedy)
        if random.random() < epsilon:
            action = env.action_space.sample()  # Choose a random action
        else:
            with torch.no_grad():
                q_values = q_network(obs_tensor)
                action = torch.argmax(q_values, dim=1).item()  # Choose the action with the highest Q-value

        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        #print(f"Episode {episode + 1}, Action: {action}, Reward: {reward}, Done: {done}")
    # Print the total reward for the episode
    if (episode + 1) % 1 == 0:
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()