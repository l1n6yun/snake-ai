import time
import imageio
import numpy as np

from SnakeEnv import SnakeGame
from stable_baselines3 import PPO

env = SnakeGame(render_mode="human")
model = PPO.load("ppo_snake", env=env)

images = []
obs, _ = env.reset()
img = env.render()
for _ in range(1000):
    images.append(img)
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    img = env.render()
    if done:
        break
    # time.sleep(0.05)

imageio.mimsave("ppo_snake.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29, loop=1, duration=0.1)