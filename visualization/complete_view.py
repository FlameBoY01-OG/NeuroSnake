import pygame
import sys
import os
import torch
import numpy as np
from env.snake_env import SnakeEnv
from model.agent import DQNAgent

CELL_SIZE = 20
MODEL_PATH = os.environ.get('MODEL_PATH', "model_checkpoints/policy_final.pth")

CAT = {
    "bg": (20, 20, 26),
    "panel": (26, 26, 34),
    "text": (230, 230, 235),
    "green": (166, 227, 161),
    "peach": (250, 179, 135),
    "lavender": (196, 196, 255),
    "mauve": (198, 160, 246),
}

def draw_complete_visualization(screen, env, agent, state, q_values, action, score, fps):
    screen.fill(CAT["bg"])
    font_large = pygame.font.SysFont("consolas", 18, bold=True)
    font_small = pygame.font.SysFont("consolas", 14)
    
    # GAME (Left side)
    game_x, game_y = 20, 20
    grid_w = env.width * CELL_SIZE
    grid_h = env.height * CELL_SIZE
    
    for x in range(env.width):
        for y in range(env.height):
            rect = pygame.Rect(game_x + x * CELL_SIZE, game_y + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (40, 40, 48), rect, 1)
    
    for i, p in enumerate(env.snake):
        color = (200, 255, 190) if i == 0 else CAT["green"]
        rect = pygame.Rect(game_x + p.x * CELL_SIZE, game_y + p.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, color, rect)
    
    if env.food:
        rect = pygame.Rect(game_x + env.food.x * CELL_SIZE, game_y + env.food.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, CAT["peach"], rect)
    
    # NEURAL NETWORK (Right side)
    net_x = game_x + grid_w + 40
    net_y = 20
    net_w = 900
    net_h = 600
    
    pygame.draw.rect(screen, CAT["panel"], (net_x, net_y, net_w, net_h))
    screen.blit(font_large.render("Neural Network", True, CAT["text"]), (net_x + 10, net_y + 10))
    
    acts, weights = agent.get_activations(state)
    
    layer_names = ["Input\n(11)", "Hidden1\n(128)", "Hidden2\n(128)", "Output\n(4)"]
    layer_x = [net_x + 100, net_x + 350, net_x + 600, net_x + 850]
    
    max_neurons = 64
    layer_neurons = []
    layer_y_positions = []
    
    for i, act in enumerate(acts):
        n_neurons = min(len(act), max_neurons if i in [1, 2] else len(act))
        layer_neurons.append(n_neurons)
        
        if n_neurons == 1:
            layer_y_positions.append([net_y + net_h // 2])
        else:
            positions = np.linspace(net_y + 80, net_y + net_h - 80, n_neurons)
            layer_y_positions.append(list(positions.astype(int)))
    
    combined_max = max(np.max(np.abs(acts[i][:layer_neurons[i]])) for i in range(len(acts))) + 1e-9
    
    # DRAW EDGES - Draw ALL connections to make network visible
    for layer_idx in range(len(weights)):
        W = weights[layer_idx]
        from_neurons = layer_neurons[layer_idx]
        to_neurons = layer_neurons[layer_idx + 1]
        
        # Draw strongest connections for each output neuron
        for to_idx in range(to_neurons):
            actual_to = to_idx if layer_idx != 2 else to_idx  # Map to actual neuron index
            
            # Get all weights to this output neuron
            strengths = []
            for from_idx in range(from_neurons):
                actual_from = from_idx if layer_idx != 1 else from_idx
                try:
                    w = abs(W[actual_to, actual_from])
                    a = abs(acts[layer_idx][actual_from])
                    strengths.append((from_idx, w * (1 + a)))
                except:
                    strengths.append((from_idx, 0))
            
            # Sort and take top 15 connections
            strengths.sort(key=lambda x: x[1], reverse=True)
            top_k = min(15, len(strengths))
            max_strength = strengths[0][1] if strengths[0][1] > 0 else 1.0
            
            for from_idx, strength in strengths[:top_k]:
                if strength > 0:
                    from_x = layer_x[layer_idx]
                    from_y = layer_y_positions[layer_idx][from_idx]
                    to_x = layer_x[layer_idx + 1]
                    to_y = layer_y_positions[layer_idx + 1][to_idx]
                    
                    # Color based on strength
                    norm = strength / max_strength
                    color_val = int(60 + 140 * norm)
                    edge_color = (color_val, color_val - 20, color_val + 30)
                    
                    width = 1 if norm < 0.5 else 2
                    pygame.draw.line(screen, edge_color, (from_x, from_y), (to_x, to_y), width)
    
    # DRAW NODES
    for layer_idx in range(len(acts)):
        n_neurons = layer_neurons[layer_idx]
        
        for neuron_idx in range(n_neurons):
            x = layer_x[layer_idx]
            y = layer_y_positions[layer_idx][neuron_idx]
            
            activation = acts[layer_idx][neuron_idx]
            intensity = min(1.0, abs(activation) / combined_max)
            
            if activation >= 0:
                color = tuple(int(CAT["panel"][i] * (1 - intensity) + CAT["green"][i] * intensity) for i in range(3))
            else:
                color = tuple(int(CAT["panel"][i] * (1 - intensity) + CAT["mauve"][i] * intensity) for i in range(3))
            
            node_size = 8 if layer_idx in [0, 3] else 5
            
            pygame.draw.circle(screen, color, (x, y), node_size)
            pygame.draw.circle(screen, CAT["text"], (x, y), node_size, 1)
        
        screen.blit(font_small.render(layer_names[layer_idx], True, CAT["text"]), (layer_x[layer_idx] - 20, net_y + 40))
    
    # Q-VALUES (Bottom right)
    q_x = net_x
    q_y = net_y + net_h + 20
    
    screen.blit(font_large.render("Q-Values & Decision", True, CAT["text"]), (q_x, q_y))
    
    actions_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    q_min, q_max = np.min(q_values), np.max(q_values)
    q_range = q_max - q_min if q_max != q_min else 1.0
    
    for i, (name, q) in enumerate(zip(actions_names, q_values)):
        y = q_y + 30 + i * 30
        color = CAT["green"] if i == action else CAT["text"]
        
        if i == action:
            pygame.draw.rect(screen, (40, 60, 40), (q_x - 5, y - 5, 300, 25))
        
        text = f"{name:<6}: {q:+7.2f}"
        screen.blit(font_small.render(text, True, color), (q_x, y))
        
        bar_x = q_x + 150
        bar_w = 200
        q_norm = (q - q_min) / q_range
        bar_fill = int(bar_w * q_norm)
        
        pygame.draw.rect(screen, (40, 40, 50), (bar_x, y, bar_w, 15))
        if bar_fill > 0:
            pygame.draw.rect(screen, color, (bar_x, y, bar_fill, 15))
    
    # INFO (Bottom left)
    info_y = game_y + grid_h + 20
    screen.blit(font_large.render(f"Score: {score}", True, CAT["text"]), (game_x, info_y))
    screen.blit(font_small.render(f"Length: {len(env.snake)}", True, CAT["text"]), (game_x, info_y + 25))
    screen.blit(font_small.render(f"FPS: {fps:.0f}", True, CAT["text"]), (game_x, info_y + 45))
    screen.blit(font_small.render(f"Action: {actions_names[action]}", True, CAT["text"]), (game_x, info_y + 65))

def main():
    pygame.init()
    
    WIDTH, HEIGHT = 1400, 750
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NeuroSnake - Complete View")
    clock = pygame.time.Clock()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    env = SnakeEnv(width=20, height=20)
    state_dim = env.reset().shape[0]
    
    agent = DQNAgent(
        input_dim=state_dim,
        output_dim=4,
        device=device,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        memory_capacity=50000,
        use_prioritized_replay=False,
        target_update_steps=1000
    )
    
    agent.load(MODEL_PATH)
    print(f"Loaded: {MODEL_PATH}\n")
    
    state = env.reset(randomize_start=True)
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        
        with torch.no_grad():
            st = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = agent.policy(st).cpu().numpy()[0]
        
        action = int(np.argmax(q_values))
        next_state, reward, done = env.step(action)
        state = next_state
        
        if done:
            print(f"Episode end - Score: {env.score}")
            state = env.reset(randomize_start=True)
        
        draw_complete_visualization(screen, env, agent, state, q_values, action, env.score, clock.get_fps())
        
        pygame.display.flip()
        clock.tick(10)
    
    pygame.quit()

if __name__ == "__main__":
    main()
