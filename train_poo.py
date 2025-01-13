import os

from SnakeEnv import SnakeGame
from stable_baselines3 import PPO

env = SnakeGame(render_mode="human1")
model = None
# if not os.path.exists("PPO_snake.zip"):
#     model = PPO("MlpPolicy", env, verbose=1)
#     model.learn(total_timesteps=10_000)
#     model.save("PPO_snake")
for _ in range(100000):
    model = PPO.load("PPO_snake", env=env)
    model.learn(total_timesteps=10_000)
    model.save("PPO_snake")