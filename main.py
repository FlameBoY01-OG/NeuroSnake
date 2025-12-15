from env.snake_env import SnakeEnv

env = SnakeEnv()
state = env.reset()
print("Initial state:", state)

for _ in range(5):
    s, r, d = env.step(3)  # move RIGHT
    print(s, r, d)
