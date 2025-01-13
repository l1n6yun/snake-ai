import random

import gymnasium as gym
import numpy as np
import pygame
from matplotlib import pyplot as plt


class SnakeGame(gym.Env):
    def __init__(self, seed=10809, render_mode=None):
        self.no_food_step = None
        random.seed(seed)
        self.rows = 6
        self.cols = 6
        self.block_size = 20

        self.score = None
        self.food = None
        self.snake = None

        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.rows * self.block_size, self.cols * self.block_size))
        else:
            self.screen = None

        self.observation_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(self.rows, self.cols),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(4)

    def reset(self, seed=10809):
        random.seed(seed)

        self.snake = [(1, 2), (2, 2), (3, 2)]
        self.food = self._generate_food()
        self.score = 0
        self.no_food_step = 0

        observation = self._generate_observation()
        return observation, {}

    def step(self, action):
        new_head = None
        self.no_food_step += 1

        if action == 0:
            new_head = (self.snake[0][0], self.snake[0][1] - 1)
        elif action == 1:
            new_head = (self.snake[0][0], self.snake[0][1] + 1)
        elif action == 2:
            new_head = (self.snake[0][0] - 1, self.snake[0][1])
        elif action == 3:
            new_head = (self.snake[0][0] + 1, self.snake[0][1])


        done = new_head in self.snake or new_head[0] < 0 or new_head[0] >= self.rows or new_head[1] < 0 or new_head[
            1] >= self.cols

        reward = 0


        if  len(self.snake) != self.rows * self.cols:
            if done:
                reward -= 10
            else:
                if new_head == self.food:
                    self.snake.insert(0, new_head)
                    self.food = self._generate_food()
                    self.score += 1
                    reward += 1
                    self.no_food_step = 0
                else:
                    self.snake.insert(0, new_head)
                    self.snake.pop()

                    if np.linalg.norm(np.array(self.snake[0])-np.array(self.food)) < np.linalg.norm(np.array(self.snake[1])-np.array(self.food)):
                        self.score += 0.1
                        reward += 0.1
                    else:
                        self.score -= 0.13
                        if self.no_food_step > 100:
                            done = True
                        reward -= min(0.05 * self.no_food_step,1)
        else:
            done = True
            reward = 1000

        obs = self._generate_observation()

        if self.screen is not None:
            self.render()

        return obs, reward, done, {}, {}

    def render(self):
        if self.screen is None:
            return

        self.screen.fill((30, 31, 34))

        pygame.font.init()
        font = pygame.font.SysFont('Arial', 14)
        text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(text, (0, 0))

        color = np.linspace(0.8, 0.5, len(self.snake))
        for i, c in zip(self.snake, color):
            pygame.draw.rect(self.screen, np.array((42, 172, 184)) * c,
                             (i[0] * self.block_size, i[1] * self.block_size, self.block_size, self.block_size))

        pygame.draw.rect(self.screen, np.array((42, 172, 184)), (
            self.snake[0][0] * self.block_size, self.snake[0][1] * self.block_size, self.block_size, self.block_size))

        pygame.draw.rect(self.screen, (199, 101, 56), (
            self.food[0] * self.block_size, self.food[1] * self.block_size, self.block_size, self.block_size))

        pygame.display.update()
        return pygame.surfarray.array3d(self.screen)

    def _generate_observation(self):
        obs = np.zeros((self.rows, self.cols))
        obs[tuple(np.transpose(self.snake))] = np.linspace(0.8, 0.2, len(self.snake), dtype=np.float32)
        obs[tuple(self.snake[0])] = 1.0
        obs[tuple(self.food)] = -1.0
        return obs

    def _generate_food(self):
        return random.choice(
            [(row, col) for row in range(self.rows) for col in range(self.cols) if (row, col) not in self.snake])


if __name__ == "__main__":
    snakeGame = SnakeGame(render_mode="human")
    snakeGame.reset()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                action = -1
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3

                if action != -1:
                    obs, reward, done, _, info = snakeGame.step(action)
                    plt.imshow(obs.T, interpolation='nearest')
                    plt.show()
                    if done:
                        snakeGame.reset()
        # 60帧
        pygame.time.delay(1000 // 60)
        snakeGame.render()
