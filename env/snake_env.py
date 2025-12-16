# env/snake_env.py
import random
from typing import Optional, List
import numpy as np
from enum import Enum

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Point:
    __slots__ = ("x", "y")
    def __init__(self, x: int, y: int):
        self.x = int(x)
        self.y = int(y)
    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y
    def __hash__(self):
        return hash((self.x, self.y))
    def __repr__(self):
        return f"Point({self.x},{self.y})"

class SnakeEnv:
    """
    Grid-based Snake environment with randomized start & RNG isolation.
    """

    def __init__(self, width: int = 20, height: int = 20, rng_seed: Optional[int] = None):
        self.width = int(width)
        self.height = int(height)
        # use a local numpy Generator for food placement & randomness
        if rng_seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(rng_seed)
        self.reset()

    def reset(self, randomize_start: bool = True):
        """
        randomize_start=True -> random head position & direction.
        Returns initial state (numpy array).
        """
        if randomize_start:
            pad = max(2, min(self.width, self.height) // 6)
            hx = int(self.rng.integers(pad, self.width - pad))
            hy = int(self.rng.integers(pad, self.height - pad))
            self.head = Point(hx, hy)
            self.direction = self.rng.choice(list(Direction))
        else:
            self.direction = Direction.RIGHT
            self.head = Point(self.width // 2, self.height // 2)

        # initial snake of length 3 placed opposite to movement
        self.snake: List[Point] = [self.head]
        if self.direction == Direction.RIGHT:
            self.snake.append(Point(self.head.x - 1, self.head.y))
            self.snake.append(Point(self.head.x - 2, self.head.y))
        elif self.direction == Direction.LEFT:
            self.snake.append(Point(self.head.x + 1, self.head.y))
            self.snake.append(Point(self.head.x + 2, self.head.y))
        elif self.direction == Direction.UP:
            self.snake.append(Point(self.head.x, self.head.y + 1))
            self.snake.append(Point(self.head.x, self.head.y + 2))
        else:  # DOWN
            self.snake.append(Point(self.head.x, self.head.y - 1))
            self.snake.append(Point(self.head.x, self.head.y - 2))

        self.score = 0
        self.food: Optional[Point] = None
        self._place_food()
        return self.get_state()

    def _place_food(self):
        free_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if Point(x, y) not in self.snake
        ]
        if not free_cells:
            self.food = None
            return
        x, y = free_cells[int(self.rng.integers(0, len(free_cells)))]
        self.food = Point(int(x), int(y))

    def step(self, action: int):
        # safe convert to Direction
        try:
            new_dir = Direction(action)
        except Exception:
            new_dir = self.direction

        # prevent immediate 180-degree reversal
        if (self.direction == Direction.UP and new_dir == Direction.DOWN) or \
           (self.direction == Direction.DOWN and new_dir == Direction.UP) or \
           (self.direction == Direction.LEFT and new_dir == Direction.RIGHT) or \
           (self.direction == Direction.RIGHT and new_dir == Direction.LEFT):
            new_dir = self.direction

        self.direction = new_dir
        
        # Store previous distance to food for reward shaping
        if self.food is not None:
            prev_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

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
        self.snake.insert(0, self.head)

        reward = 0.0
        done = False

        if self._collision():
            reward = -10.0
            done = True
            return self.get_state(), reward, done

        if self.food is not None and self.head.x == self.food.x and self.head.y == self.food.y:
            reward = 10.0
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
            reward = 0

        return self.get_state(), reward, done

    def _collision(self) -> bool:
        if self.head.x < 0 or self.head.x >= self.width or self.head.y < 0 or self.head.y >= self.height:
            return True
        if self.head in self.snake[1:]:
            return True
        return False

    def _is_collision(self, point: Point) -> bool:
        if point.x < 0 or point.x >= self.width or point.y < 0 or point.y >= self.height:
            return True
        if point in self.snake:
            return True
        return False

    def get_state(self):
        """
        Simple but effective state representation: 11 features.
        This is proven to work for Snake.
        """
        head = self.head
        # Direction booleans
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # Points around head
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)

        # Danger in each relative direction
        danger_straight = (
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d))
        )

        danger_right = (
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d))
        )

        danger_left = (
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d))
        )

        # Food direction (simple binary)
        if self.food is None:
            food_left = food_right = food_up = food_down = 0
        else:
            food_left = 1 if self.food.x < head.x else 0
            food_right = 1 if self.food.x > head.x else 0
            food_up = 1 if self.food.y < head.y else 0
            food_down = 1 if self.food.y > head.y else 0

        state = [
            # Danger (3)
            float(danger_straight),
            float(danger_right),
            float(danger_left),
            
            # Direction (4)
            float(dir_l),
            float(dir_r),
            float(dir_u),
            float(dir_d),
            
            # Food location (4)
            float(food_left),
            float(food_right),
            float(food_up),
            float(food_down),
        ]
        return np.array(state, dtype=np.float32)
