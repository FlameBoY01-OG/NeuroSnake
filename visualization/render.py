# visualization/render.py
import pygame
import sys
import time
import torch
from env.snake_env import SnakeEnv
from model.agent import DQNAgent
import numpy as np

CELL_SIZE = 25
FPS = 12
BG_COLOR = (18, 18, 18)
GRID_COLOR = (40, 40, 40)
SNAKE_COLOR = (0, 190, 0)
HEAD_COLOR = (0, 255, 0)
FOOD_COLOR = (200, 0, 0)
MODEL_PATH = "model_checkpoints/policy_final.pth"

def draw_grid(screen, env):
    for x in range(env.width):
        for y in range(env.height):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

def draw_snake(screen, env):
    for i, p in enumerate(env.snake):
        color = HEAD_COLOR if i == 0 else SNAKE_COLOR
        rect = pygame.Rect(p.x * CELL_SIZE, p.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, color, rect)

def draw_food(screen, env):
    if env.food is None:
        return
    rect = pygame.Rect(env.food.x * CELL_SIZE, env.food.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, FOOD_COLOR, rect)

def main():
    pygame.init()
    env = SnakeEnv(width=20, height=20)
    screen = pygame.display.set_mode((env.width * CELL_SIZE, env.height * CELL_SIZE))
    pygame.display.set_caption("NeuroSnake — DQN Eval")
    clock = pygame.time.Clock()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(device=device)
    # load weights onto device
    agent.load(MODEL_PATH)

    state = env.reset()
    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = agent.select_action(state, eps=0.0)
        state, reward, done = env.step(action)
        if done:
            time.sleep(0.4)
            state = env.reset()

        screen.fill(BG_COLOR)
        draw_grid(screen, env)
        draw_snake(screen, env)
        draw_food(screen, env)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
