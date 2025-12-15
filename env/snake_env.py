# env/snake_env.py
import random
import numpy as np
from enum import Enum

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Point:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
    def __eq__(self, o):
        return isinstance(o, Point) and self.x == o.x and self.y == o.y
    def __hash__(self):
        return hash((self.x, self.y))
    def __repr__(self):
        return f"Point({self.x},{self.y})"

class SnakeEnv:
    """
    Returns a 17-dim float32 state:
    [danger_up, danger_down, danger_left, danger_right,
     dir_up, dir_down, dir_left, dir_right,
     food_left, food_right, food_up, food_down,
     head_x_norm, head_y_norm,
     food_x_norm, food_y_norm,
     length_norm]
    """
    def __init__(self, width=10, height=10):
        self.width = int(width)
        self.height = int(height)
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [
            self.head,
            Point(self.head.x - 1, self.head.y),
            Point(self.head.x - 2, self.head.y),
        ]
        self.score = 0
        self._place_food()
        return self.get_state()

    def _place_food(self):
        free = [(x, y) for x in range(self.width) for y in range(self.height) if Point(x, y) not in self.snake]
        if not free:
            self.food = None
            return
        x, y = random.choice(free)
        self.food = Point(x, y)

    def step(self, action):
        # action is int 0..3 mapping to Direction
        self._move(action)
        self.snake.insert(0, self.head)

        if self._collision(self.head):
            return self.get_state(), -10.0, True

        reward = -0.01  # step penalty
        if self.food is not None and self.head == self.food:
            reward = 10.0
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        return self.get_state(), reward, False

    def _collision(self, p):
        return (
            p.x < 0 or p.x >= self.width or
            p.y < 0 or p.y >= self.height or
            p in self.snake[1:]
        )

    def _move(self, action):
        self.direction = Direction(action)
        x, y = self.head.x, self.head.y
        if self.direction == Direction.UP:
            y -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.RIGHT:
            x += 1
        self.head = Point(x, y)

    def _danger(self, p):
        return p.x < 0 or p.x >= self.width or p.y < 0 or p.y >= self.height or p in self.snake

    def get_state(self):
        h = self.head
        # dangers in four directions
        danger_up = self._danger(Point(h.x, h.y - 1))
        danger_down = self._danger(Point(h.x, h.y + 1))
        danger_left = self._danger(Point(h.x - 1, h.y))
        danger_right = self._danger(Point(h.x + 1, h.y))

        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT

        if self.food is None:
            food_left = food_right = food_up = food_down = False
            food_x_norm = 0.0
            food_y_norm = 0.0
        else:
            food_left = self.food.x < h.x
            food_right = self.food.x > h.x
            food_up = self.food.y < h.y
            food_down = self.food.y > h.y
            food_x_norm = (self.food.x) / max(1, self.width - 1)
            food_y_norm = (self.food.y) / max(1, self.height - 1)

        head_x_norm = h.x / max(1, self.width - 1)
        head_y_norm = h.y / max(1, self.height - 1)
        length_norm = len(self.snake) / float(self.width * self.height)

        state = [
            float(danger_up), float(danger_down), float(danger_left), float(danger_right),
            float(dir_u), float(dir_d), float(dir_l), float(dir_r),
            float(food_left), float(food_right), float(food_up), float(food_down),
            float(head_x_norm), float(head_y_norm),
            float(food_x_norm), float(food_y_norm),
            float(length_norm),
        ]
        return np.array(state, dtype=np.float32)
